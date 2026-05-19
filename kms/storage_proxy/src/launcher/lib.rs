//
// Copyright 2023 The Project Oak Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::{
    fmt::Display,
    net::{IpAddr, Ipv4Addr, SocketAddr},
};

use anyhow::Context;
use oak_proto_rust::oak::{
    attestation::v1::{endorsements, Endorsements, Evidence, OakContainersEndorsements},
    session::v1::EndorsedEvidence,
};
use tokio::{
    net::TcpListener,
    sync::{oneshot, watch},
    task::JoinHandle,
    time::{timeout, Duration},
};

/// The local IP address assigned to the VM guest.
pub const VM_LOCAL_ADDRESS: IpAddr = IpAddr::V4(Ipv4Addr::new(10, 0, 2, 15));

/// The local port that the Storage Proxy app listens on inside the VM.
pub const VM_LOCAL_PORT: u16 = 8008;

/// The local port that the Orchestrator should be listening on.
pub const VM_ORCHESTRATOR_LOCAL_PORT: u16 = 4000;

/// The local address that will be forwarded by the VMM to the guest's IP
/// address.
const PROXY_ADDRESS: Ipv4Addr = Ipv4Addr::LOCALHOST;

/// Number of seconds to wait for the VM to start up.
const VM_START_TIMEOUT: u64 = 300;

#[derive(Debug)]
pub struct Args {
    pub system_image: std::path::PathBuf,
    pub container_bundle: std::path::PathBuf,
    pub application_config: Vec<u8>,
    pub qemu_params: super::qemu::Params,
}

/// Validates that a path exists and is a file
pub fn path_exists(s: &str) -> Result<std::path::PathBuf, String> {
    let path = std::path::PathBuf::from(s);
    if !std::fs::metadata(s).map_err(|err| err.to_string())?.is_file() {
        Err(String::from("path does not represent a file"))
    } else {
        Ok(path)
    }
}

impl Args {
    pub fn parse() -> Result<Self, anyhow::Error> {
        let args: Vec<String> = std::env::args().collect();
        let mut system_image = None;
        let mut container_bundle = None;
        let mut vmm_binary = None;
        let mut stage0_binary = None;
        let mut kernel = None;
        let mut initrd = None;
        let mut memory_size = None;
        let mut num_cpus: u8 = 1;
        let mut ramdrive_size: u32 = 0;
        let mut telnet_console = None;
        let mut virtio_guest_cid = None;
        let mut vm_type = super::qemu::VmType::Default;
        let mut quiet = false;
        let mut storage_port = None;

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--system-image" | "--system_image" => {
                    i += 1;
                    system_image = Some(path_exists(&args[i]).map_err(|e| anyhow::anyhow!("--system-image: {}", e))?);
                }
                "--container-bundle" | "--container_bundle" => {
                    i += 1;
                    container_bundle = Some(path_exists(&args[i]).map_err(|e| anyhow::anyhow!("--container-bundle: {}", e))?);
                }
                "--vmm-binary" | "--vmm_binary" => {
                    i += 1;
                    vmm_binary = Some(path_exists(&args[i]).map_err(|e| anyhow::anyhow!("--vmm-binary: {}", e))?);
                }
                "--stage0-binary" | "--stage0_binary" => {
                    i += 1;
                    stage0_binary = Some(path_exists(&args[i]).map_err(|e| anyhow::anyhow!("--stage0-binary: {}", e))?);
                }
                "--kernel" => {
                    i += 1;
                    kernel = Some(path_exists(&args[i]).map_err(|e| anyhow::anyhow!("--kernel: {}", e))?);
                }
                "--initrd" => {
                    i += 1;
                    initrd = Some(path_exists(&args[i]).map_err(|e| anyhow::anyhow!("--initrd: {}", e))?);
                }
                "--memory-size" | "--memory_size" => {
                    i += 1;
                    memory_size = Some(args[i].clone());
                }
                "--num-cpus" | "--num_cpus" | "--vcpus" => {
                    i += 1;
                    num_cpus = args[i].parse().context("invalid --num-cpus")?;
                }
                "--ramdrive-size" | "--ramdrive_size" => {
                    i += 1;
                    ramdrive_size = args[i].parse().context("invalid --ramdrive-size")?;
                }
                "--telnet-console" | "--telnet_console" => {
                    i += 1;
                    telnet_console = Some(args[i].parse().context("invalid --telnet-console")?);
                }
                "--virtio-guest-cid" | "--virtio_guest_cid" => {
                    i += 1;
                    virtio_guest_cid = Some(args[i].parse().context("invalid --virtio-guest-cid")?);
                }
                "--vm-type" | "--vm_type" => {
                    i += 1;
                    vm_type = super::qemu::VmType::from_str(&args[i]).map_err(|e| anyhow::anyhow!(e))?;
                }
                "--quiet" => {
                    quiet = true;
                }
                "--storage-port" | "--storage_port" => {
                    i += 1;
                    storage_port = Some(args[i].parse().context("invalid --storage-port")?);
                }
                other => {
                    anyhow::bail!("unknown argument: {}", other);
                }
            }
            i += 1;
        }

        Ok(Args {
            system_image: system_image.context("--system-image is required")?,
            container_bundle: container_bundle.context("--container-bundle is required")?,
            application_config: Vec::new(),
            qemu_params: super::qemu::Params {
                vmm_binary: vmm_binary.context("--vmm-binary is required")?,
                stage0_binary: stage0_binary.context("--stage0-binary is required")?,
                kernel: kernel.context("--kernel is required")?,
                initrd: initrd.context("--initrd is required")?,
                memory_size,
                num_cpus,
                ramdrive_size,
                telnet_console,
                virtio_guest_cid,
                vm_type,
                quiet,
                storage_port,
            },
        })
    }
}

#[derive(Clone)]
pub enum Channel {
    Network { host_proxy_port: u16, trusted_app_address: Option<SocketAddr> },
}

/// Interface that is connected to the trusted application.
pub enum TrustedApplicationAddress {
    Network(SocketAddr),
}

impl TryFrom<Channel> for TrustedApplicationAddress {
    type Error = anyhow::Error;

    fn try_from(channel: Channel) -> Result<TrustedApplicationAddress, Self::Error> {
        match channel {
            Channel::Network { host_proxy_port: _, trusted_app_address } => {
                trusted_app_address.map(TrustedApplicationAddress::Network)
            }
        }
        .ok_or_else(|| anyhow::anyhow!("trusted application address not set"))
    }
}

impl Display for TrustedApplicationAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrustedApplicationAddress::Network(addr) => addr.fmt(f),
        }
    }
}

pub struct Launcher {
    vmm: super::qemu::Qemu,
    server: JoinHandle<Result<(), anyhow::Error>>,
    #[allow(dead_code)]
    host_orchestrator_proxy_port: u16,
    // Endorsed Attestation Evidence consists of Attestation Evidence (initialized by the
    // Orchestrator) and Attestation Endorsement (initialized by the Launcher).
    endorsed_evidence: Option<EndorsedEvidence>,
    // Receiver that is used to get the Attestation Evidence from the server implementation.
    evidence_receiver: Option<oneshot::Receiver<Evidence>>,
    app_ready_notifier: Option<oneshot::Receiver<()>>,
    trusted_app_channel: Channel,
    shutdown: Option<watch::Sender<()>>,
}

impl Launcher {
    pub async fn create(args: Args) -> Result<Self, anyhow::Error> {
        // Let the OS assign an open port for the launcher service.
        let sockaddr = SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0);
        let listener = TcpListener::bind(sockaddr).await?;
        let port = listener.local_addr()?.port();

        log::info!("Launcher service listening on port {port}");
        let (evidence_sender, evidence_receiver) = oneshot::channel::<Evidence>();
        let (shutdown_sender, mut shutdown_receiver) = watch::channel::<()>(());
        shutdown_receiver.mark_unchanged(); // Don't immediately notify on the initial value.
        let (app_notifier_sender, app_notifier_receiver) = oneshot::channel::<()>();
        let endorsements = get_endorsements();
        let server = tokio::spawn(super::server::new(
            listener,
            args.system_image,
            args.container_bundle,
            args.application_config,
            evidence_sender,
            app_notifier_sender,
            shutdown_receiver,
            endorsements,
        ));

        // Bind the host proxy to port 8008 so the workshop-server's KMS VMs
        // can reach the storage proxy via guestfwd 10.0.2.100:8008 → host:8008.
        // We bind and immediately drop to release the port for QEMU's hostfwd.
        let host_proxy_port: u16 = 8008;
        let proxy_sockaddr = SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), host_proxy_port);
        // Verify the port is available, then release it for QEMU.
        drop(TcpListener::bind(proxy_sockaddr).await?);
        let trusted_app_channel = Channel::Network { host_proxy_port, trusted_app_address: None };

        let host_orchestrator_proxy_port =
            { TcpListener::bind(sockaddr).await?.local_addr()?.port() };
        let vmm = super::qemu::Qemu::start(
            args.qemu_params,
            port,
            Some(host_proxy_port),
            host_orchestrator_proxy_port,
        )?;

        Ok(Self {
            vmm,
            server,
            host_orchestrator_proxy_port,
            endorsed_evidence: None,
            evidence_receiver: Some(evidence_receiver),
            app_ready_notifier: Some(app_notifier_receiver),
            trusted_app_channel,
            shutdown: Some(shutdown_sender),
        })
    }

    /// Gets the address that the untrusted application can use to connect to
    /// the trusted application.
    ///
    /// This call will wait until the trusted app has notified the launcher
    /// once that it is ready via the orchestrator.
    pub async fn get_trusted_app_address(
        &mut self,
    ) -> Result<TrustedApplicationAddress, anyhow::Error> {
        if let Some(receiver) = self.app_ready_notifier.take() {
            timeout(Duration::from_secs(VM_START_TIMEOUT), receiver).await??;
            match &mut self.trusted_app_channel {
                Channel::Network { host_proxy_port, trusted_app_address } => {
                    trusted_app_address
                        .replace(SocketAddr::new(IpAddr::V4(PROXY_ADDRESS), *host_proxy_port));
                }
            }
        }
        self.trusted_app_channel.clone().try_into()
    }

    /// Gets the endorsed attestation evidence.
    pub async fn get_endorsed_evidence(&mut self) -> anyhow::Result<EndorsedEvidence> {
        if let Some(receiver) = self.evidence_receiver.take() {
            let evidence = timeout(Duration::from_secs(VM_START_TIMEOUT), receiver)
                .await
                .context("couldn't get attestation evidence before timeout")?
                .context("no attestation evidence available")?;

            let endorsements = get_endorsements();

            let endorsed_evidence =
                EndorsedEvidence { evidence: Some(evidence), endorsements: Some(endorsements) };
            self.endorsed_evidence.replace(endorsed_evidence);
        }
        self.endorsed_evidence
            .clone()
            .ok_or_else(|| anyhow::anyhow!("endorsed evidence is not set"))
    }

    pub async fn wait(&mut self) -> Result<(), anyhow::Error> {
        self.vmm.wait().await?;
        Ok(())
    }

    pub async fn kill(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        let _ = self.vmm.kill().await;
        self.server.abort();
    }
}

fn get_endorsements() -> Endorsements {
    Endorsements {
        r#type: Some(endorsements::Type::OakContainers(OakContainersEndorsements {
            root_layer: None,
            kernel_layer: None,
            system_layer: None,
            container_layer: None,
        })),
        ..Default::default()
    }
}
