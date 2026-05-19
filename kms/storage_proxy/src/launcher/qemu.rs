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
//

use std::{
    io::{BufRead, BufReader},
    net::Ipv4Addr,
    os::{fd::AsRawFd, unix::net::UnixStream},
    path::PathBuf,
    process::Stdio,
};

use std::os::unix::process::CommandExt;

use anyhow::Result;

use super::path_exists;

/// Types of confidential VMs
#[derive(Clone, Debug, Default, PartialEq)]
pub enum VmType {
    #[default]
    Default,
    Sev,
    SevEs,
    SevSnp,
    Tdx,
}

impl VmType {
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "default" => Ok(VmType::Default),
            "sev" => Ok(VmType::Sev),
            "sev-es" | "seves" => Ok(VmType::SevEs),
            "sev-snp" | "sevsnp" => Ok(VmType::SevSnp),
            "tdx" => Ok(VmType::Tdx),
            _ => Err(format!("unknown VM type: {}", s)),
        }
    }
}

/// Represents parameters used for launching VM instances.
#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    pub vmm_binary: PathBuf,
    pub stage0_binary: PathBuf,
    pub kernel: PathBuf,
    pub initrd: PathBuf,
    pub memory_size: Option<String>,
    pub num_cpus: u8,
    pub ramdrive_size: u32,
    pub telnet_console: Option<u16>,
    pub virtio_guest_cid: Option<u32>,
    pub vm_type: VmType,
    pub quiet: bool,
    pub storage_port: Option<u16>,
}

pub struct Qemu {
    instance: tokio::process::Child,
    guest_cid: Option<u32>,
}

impl Qemu {
    pub fn start(
        params: Params,
        launcher_service_port: u16,
        host_proxy_port: Option<u16>,
        host_orchestrator_proxy_port: u16,
    ) -> Result<Self> {
        let mut cmd = tokio::process::Command::new(&params.vmm_binary);
        let (guest_socket, host_socket) = UnixStream::pair()?;
        cmd.kill_on_drop(true);

        // Configure QEMU output based on quiet flag
        if params.quiet {
            cmd.stderr(Stdio::null());
            cmd.stdout(Stdio::null());
        } else {
            cmd.stderr(Stdio::inherit());
            cmd.stdout(Stdio::inherit());
        }
        cmd.stdin(Stdio::null());

        // Extract the raw file descriptor number from the stream. We leak the
        // socket so the FD stays valid, then use pre_exec to clear the
        // close-on-exec flag so QEMU inherits it.
        let guest_socket_fd = guest_socket.as_raw_fd();
        std::mem::forget(guest_socket);

        // Safety: nix::fcntl calls are async-signal-safe.
        unsafe {
            cmd.pre_exec(move || {
                // Clear FD_CLOEXEC so the child process inherits this FD.
                let flags = nix::fcntl::fcntl(guest_socket_fd, nix::fcntl::FcntlArg::F_GETFD)
                    .map_err(|e| std::io::Error::from_raw_os_error(e as i32))?;
                let mut fd_flags = nix::fcntl::FdFlag::from_bits_truncate(flags);
                fd_flags.remove(nix::fcntl::FdFlag::FD_CLOEXEC);
                nix::fcntl::fcntl(
                    guest_socket_fd,
                    nix::fcntl::FcntlArg::F_SETFD(fd_flags),
                )
                .map_err(|e| std::io::Error::from_raw_os_error(e as i32))?;
                Ok(())
            });
        }

        // Construct the command-line arguments for `qemu`.
        cmd.arg("-enable-kvm");
        // SEV-SNP firmware requires a sanitized CPUID set; `-cpu host` exposes
        // host-specific bits (e.g. 0x80000021 EAX bits 27-28) that fail validation.
        // Use a versioned model for SNP, and `host` otherwise (for RDRAND etc.).
        let cpu_model = match params.vm_type {
            VmType::SevSnp => "EPYC-Milan-v2",
            _ => "host",
        };
        cmd.args(["-cpu", cpu_model]);
        if let Some(ref memory_size) = params.memory_size {
            cmd.args(["-m", memory_size]);
        };
        cmd.args(["-smp", format!("{}", params.num_cpus).as_str()]);
        cmd.arg("-nodefaults");
        cmd.arg("-nographic");
        cmd.arg("-no-reboot");

        // Platform specific machine setup
        let microvm_common = "microvm,acpi=on,pcie=on".to_string();
        let sev_machine_suffix = ",confidential-guest-support=sev0,memory-backend=ram1";
        let sev_common_object = format!(
            "memory-backend-memfd,id=ram1,size={},share=true,reserve=false",
            params.memory_size.clone().unwrap_or("8G".to_string())
        );
        let sev_config_object = "id=sev0,cbitpos=51,reduced-phys-bits=1";
        let tdx_machine_suffix = ",kernel_irqchip=split,memory-encryption=tdx,memory-backend=ram1";
        let tdx_common_object = format!(
            "memory-backend-ram,id=ram1,size={}",
            params.memory_size.unwrap_or("8G".to_string())
        );

        let (machine_arg, object_args) = match params.vm_type {
            VmType::Default => (microvm_common, vec![]),
            VmType::Sev => (
                microvm_common + sev_machine_suffix,
                vec![
                    sev_common_object,
                    "sev-guest,".to_string() + sev_config_object + ",policy=0x1",
                ],
            ),
            VmType::SevEs => (
                microvm_common + sev_machine_suffix,
                vec![
                    sev_common_object,
                    "sev-guest,".to_string() + sev_config_object + ",policy=0x5",
                ],
            ),
            VmType::SevSnp => (
                microvm_common + sev_machine_suffix,
                vec![
                    sev_common_object,
                    // Reference:
                    // https://lore.kernel.org/kvm/20240502231140.GC13783@ls.amr.corp.intel.com/T/
                    "sev-snp-guest,".to_string() + sev_config_object + ",id-auth=",
                ],
            ),
            VmType::Tdx => (
                microvm_common + tdx_machine_suffix,
                vec![
                    r#"{"qom-type":"tdx-guest","id":"tdx","sept-ve-disable":true, "quote-generation-socket":{"type": "vsock", "cid":"2","port":"4050"}}"#.to_string(),
                    tdx_common_object,
                ],
            ),
        };
        cmd.args(["-machine", &machine_arg]);
        for obj_arg in object_args {
            cmd.args(["-object", &obj_arg]);
        }

        // Route first serial port to console.
        if let Some(port) = params.telnet_console {
            cmd.args(["-serial", format!("telnet:localhost:{port},server").as_str()]);
        } else {
            cmd.args(["-chardev", format!("socket,id=consock,fd={guest_socket_fd}").as_str()]);
            cmd.args(["-serial", "chardev:consock"]);
        }

        // Set up the networking.
        let vm_address = super::VM_LOCAL_ADDRESS;
        let vm_orchestrator_port = super::VM_ORCHESTRATOR_LOCAL_PORT;
        let host_address = Ipv4Addr::LOCALHOST;

        // guestfwd: 10.0.2.100:8080 → launcher service (Oak Orchestrator protocol)
        let mut netdev_rules = vec![
            "user".to_string(),
            "id=netdev".to_string(),
            format!("guestfwd=tcp:10.0.2.100:8080-tcp:{host_address}:{launcher_service_port}"),
            format!("hostfwd=tcp:{host_address}:{host_orchestrator_proxy_port}-{vm_address}:{vm_orchestrator_port}"),
        ];
        if let Some(host_proxy_port) = host_proxy_port {
            let vm_port = super::VM_LOCAL_PORT;
            netdev_rules.push(format!(
                "hostfwd=tcp:0.0.0.0:{host_proxy_port}-{vm_address}:{vm_port}"
            ));
        };
        // guestfwd: 10.0.2.100:8008 → storage service on host
        if let Some(storage_port) = params.storage_port {
            netdev_rules.push(format!(
                "guestfwd=tcp:10.0.2.100:8008-tcp:{host_address}:{storage_port}"
            ));
        }
        cmd.args(["-netdev", netdev_rules.join(",").as_str()]);
        cmd.args([
            "-device",
            "virtio-net-pci,disable-legacy=on,iommu_platform=true,netdev=netdev,romfile=",
        ]);

        // vsock device
        let virtio_guest_cid = params
            .virtio_guest_cid
            .unwrap_or_else(|| nix::unistd::gettid().as_raw().unsigned_abs());
        cmd.args(["-device", &format!("vhost-vsock-pci,guest-cid={virtio_guest_cid},rombar=0")]);

        // BIOS, kernel, initrd
        cmd.args(["-bios", params.stage0_binary.to_str().unwrap_or("")]);
        cmd.args(["-kernel", params.kernel.to_str().unwrap_or("")]);
        cmd.args(["-initrd", params.initrd.to_str().unwrap_or("")]);

        // Kernel command line — use SLIRP gateway (10.0.2.2) to reach the host
        // directly, bypassing guestfwd which has issues with concurrent h2 connections.
        let ramdrive_size = params.ramdrive_size;
        let cmdline = vec![
            params.telnet_console.map_or_else(|| "", |_| "debug").to_string(),
            "console=ttyS0".to_string(),
            "panic=-1".to_string(),
            "brd.rd_nr=1".to_string(),
            format!("brd.rd_size={ramdrive_size}"),
            "brd.max_part=1".to_string(),
            format!("ip={vm_address}:::255.255.255.0::eth0:off"),
            "loglevel=7".to_string(),
            "--".to_string(),
            format!("--launcher-addr=http://10.0.2.2:{launcher_service_port}"),
        ];

        cmd.args(["-append", cmdline.join(" ").as_str()]);

        log::debug!("QEMU command line: {:?}", cmd);

        // Spit out everything we read, if we were not using telnet console.
        if params.telnet_console.is_none() {
            tokio::spawn(async {
                let mut reader = BufReader::new(host_socket);
                let mut line = String::new();
                while reader.read_line(&mut line).expect("couldn't read line") > 0 {
                    print!("{}", line);
                    line.clear();
                }
            });
        }

        let instance = cmd.spawn()?;

        Ok(Self { instance, guest_cid: params.virtio_guest_cid })
    }

    pub async fn kill(&mut self) -> Result<std::process::ExitStatus> {
        self.instance.start_kill()?;
        self.wait().await
    }

    pub async fn wait(&mut self) -> Result<std::process::ExitStatus> {
        self.instance.wait().await.map_err(anyhow::Error::from)
    }

    pub fn guest_cid(&self) -> Option<u32> {
        self.guest_cid
    }
}
