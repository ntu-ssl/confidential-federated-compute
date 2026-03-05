// Copyright 2025 Google LLC.
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

use crate::launcher_module::{Launcher, Args, ChannelType, TrustedApplicationAddress};
use crate::launcher_module::qemu::VmType;
use crate::models::{KmsService, ProcessState, ProcessStatus, ServiceType, TestConcatService, SessionId};
use crate::services::kms_client::ProstProtoConversionExt;
use anyhow::{Result, anyhow};
use chrono::Utc;
use clap::ValueEnum;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::info;
use uuid::Uuid;

/// Number of KMS instances in the pool
const KMS_POOL_SIZE: usize = 5;

/// Starting CID for VM instances (must be >= 3, as 0-2 are reserved by vsock)
const BASE_VIRTIO_CID: u32 = 100;

/// KMS pool entry containing launcher, service info, and session assignments
pub struct KmsPoolEntry {
    pub launcher: Launcher,
    pub service: KmsService,
    pub assigned_sessions: HashSet<SessionId>,
}

/// Manages launcher instances for KMS and TEE services
pub struct LauncherManager {
    system_image: PathBuf,
    kms_bundle: PathBuf,
    test_concat_bundle: PathBuf,
    vmm_binary: PathBuf,
    stage0_binary: PathBuf,
    kernel: PathBuf,
    initrd: PathBuf,
    ramdrive_size: u32,
    vm_type: VmType,
    quiet: bool,
    /// Pool of pre-started KMS instances (fixed size: 5)
    kms_pool: Arc<Mutex<Vec<KmsPoolEntry>>>,
    /// Store active TEE launchers to keep them alive
    tee_launchers: Arc<Mutex<Vec<Launcher>>>,
    /// Atomic counter for allocating unique virtio guest CIDs
    next_cid: AtomicU32,
}

impl LauncherManager {
    pub fn new(
        system_image: PathBuf,
        kms_bundle: PathBuf,
        test_concat_bundle: PathBuf,
        vmm_binary: PathBuf,
        stage0_binary: PathBuf,
        kernel: PathBuf,
        initrd: PathBuf,
        ramdrive_size: u32,
        vm_type_str: String,
        quiet: bool,
    ) -> Self {
        // Parse vm_type string to enum
        let vm_type = VmType::from_str(&vm_type_str, true).unwrap_or(VmType::SevSnp);

        LauncherManager {
            system_image,
            kms_bundle,
            test_concat_bundle,
            vmm_binary,
            stage0_binary,
            kernel,
            initrd,
            ramdrive_size,
            vm_type,
            quiet,
            kms_pool: Arc::new(Mutex::new(Vec::new())),
            tee_launchers: Arc::new(Mutex::new(Vec::new())),
            next_cid: AtomicU32::new(BASE_VIRTIO_CID),
        }
    }

    /// Allocate a unique virtio guest CID for a new VM
    fn allocate_cid(&self) -> u32 {
        let cid = self.next_cid.fetch_add(1, Ordering::SeqCst);
        info!("Allocated virtio_guest_cid: {}", cid);
        cid
    }

    /// Initialize the KMS pool with pre-started instances
    /// This should be called once at server startup
    pub async fn initialize_kms_pool(&self) -> Result<()> {
        info!("Initializing KMS pool with {} instances...", KMS_POOL_SIZE);

        let mut pool = self.kms_pool.lock().await;

        for i in 0..KMS_POOL_SIZE {
            info!("Starting KMS instance {} of {}...", i + 1, KMS_POOL_SIZE);

            let (launcher, service) = self.launch_kms_internal(Some("2G".to_string())).await?;

            pool.push(KmsPoolEntry {
                launcher,
                service,
                assigned_sessions: HashSet::new(),
            });

            info!("KMS instance {} started successfully", i + 1);
        }

        info!("✓ KMS pool initialized with {} instances", KMS_POOL_SIZE);
        Ok(())
    }

    /// Assign a session to the least-loaded KMS instance
    /// Returns the assigned KMS service and its index in the pool
    pub async fn assign_kms_to_session(&self, session_id: SessionId) -> Result<(usize, KmsService)> {
        let mut pool = self.kms_pool.lock().await;

        if pool.is_empty() {
            return Err(anyhow!("KMS pool is not initialized"));
        }

        // Find the KMS instance with the fewest assigned sessions (least-loaded)
        let (kms_index, _) = pool
            .iter()
            .enumerate()
            .min_by_key(|(_, entry)| entry.assigned_sessions.len())
            .ok_or_else(|| anyhow!("No KMS instances available"))?;

        // Assign the session
        pool[kms_index].assigned_sessions.insert(session_id);

        let kms_service = pool[kms_index].service.clone();

        info!(
            "Assigned session {} to KMS instance {} ({} sessions assigned)",
            session_id,
            kms_index,
            pool[kms_index].assigned_sessions.len()
        );

        Ok((kms_index, kms_service))
    }

    /// Unassign a session from its KMS instance
    pub async fn unassign_kms_from_session(&self, session_id: SessionId) -> Result<()> {
        let mut pool = self.kms_pool.lock().await;

        // Find and remove the session from whichever KMS it's assigned to
        for (index, entry) in pool.iter_mut().enumerate() {
            if entry.assigned_sessions.remove(&session_id) {
                info!(
                    "Unassigned session {} from KMS instance {} ({} sessions remaining)",
                    session_id,
                    index,
                    entry.assigned_sessions.len()
                );
                return Ok(());
            }
        }

        // Session wasn't found in any KMS assignment
        Ok(())
    }

    /// Get KMS instance for a specific session
    pub async fn get_kms_for_session(&self, session_id: SessionId) -> Result<KmsService> {
        let pool = self.kms_pool.lock().await;

        for entry in pool.iter() {
            if entry.assigned_sessions.contains(&session_id) {
                return Ok(entry.service.clone());
            }
        }

        Err(anyhow!("Session {} is not assigned to any KMS instance", session_id))
    }

    /// Internal method to launch a single KMS instance
    async fn launch_kms_internal(&self, memory_size: Option<String>) -> Result<(Launcher, KmsService)> {
        // Allocate a unique CID for this KMS instance
        let cid = self.allocate_cid();

        // Create launcher args for KMS
        let args = Args {
            system_image: self.system_image.clone(),
            container_bundle: self.kms_bundle.clone(),
            application_config: Vec::new(),
            qemu_params: crate::launcher_module::qemu::Params {
                vmm_binary: self.vmm_binary.clone(),
                stage0_binary: self.stage0_binary.clone(),
                kernel: self.kernel.clone(),
                initrd: self.initrd.clone(),
                memory_size,
                num_cpus: 2,
                ramdrive_size: self.ramdrive_size,
                telnet_console: None,
                virtio_guest_cid: Some(cid),
                pci_passthrough: None,
                vm_type: self.vm_type.clone(),
                quiet: self.quiet,
            },
            communication_channel: ChannelType::Network,
        };

        // Create the launcher
        let mut launcher = Launcher::create(args).await?;

        // Get trusted app address
        let trusted_app_address = launcher.get_trusted_app_address().await?;

        let (host, port) = match trusted_app_address {
            TrustedApplicationAddress::Network(addr) => {
                (addr.ip().to_string(), addr.port())
            }
            TrustedApplicationAddress::VirtioVsock(_addr) => {
                ("127.0.0.1".to_string(), 8080u16)
            }
        };

        let kms_service = KmsService {
            address: host,
            port,
            process_state: ProcessState {
                pid: 0,
                start_time: Utc::now(),
                timeout: 300,
                service_type: ServiceType::Kms,
                status: ProcessStatus::Running,
            },
        };

        Ok((launcher, kms_service))
    }

    /// Launch KMS service (kept for backward compatibility)
    pub async fn launch_kms(&self, memory_size: Option<String>) -> Result<(Launcher, KmsService)> {
        info!("Launching KMS service with memory_size: {:?}", memory_size);
        self.launch_kms_internal(memory_size).await
    }

    /// Launch Test Concat TEE service
    pub async fn launch_test_concat(
        &self,
        memory_size: Option<String>,
    ) -> Result<(Launcher, TestConcatService)> {
        info!("Launching Test Concat TEE service with memory_size: {:?}", memory_size);

        // Allocate a unique CID for this TEE instance
        // let cid = self.allocate_cid();

        // Create launcher args for Test Concat
        let args = Args {
            system_image: self.system_image.clone(),
            container_bundle: self.test_concat_bundle.clone(),
            application_config: Vec::new(),
            qemu_params: crate::launcher_module::qemu::Params {
                vmm_binary: self.vmm_binary.clone(),
                stage0_binary: self.stage0_binary.clone(),
                kernel: self.kernel.clone(),
                initrd: self.initrd.clone(),
                memory_size,
                num_cpus: 2,
                ramdrive_size: self.ramdrive_size,
                telnet_console: None,
                virtio_guest_cid: None,
                pci_passthrough: None,
                vm_type: self.vm_type.clone(),
                quiet: self.quiet,
            },
            communication_channel: ChannelType::Network,
        };

        // Create the launcher
        let mut launcher = Launcher::create(args).await?;

        // Get trusted app address
        let trusted_app_address = launcher.get_trusted_app_address().await?;

        let (host, port) = match trusted_app_address {
            TrustedApplicationAddress::Network(addr) => {
                info!("Test Concat TEE service at network address: {}", addr);
                (addr.ip().to_string(), addr.port())
            }
            TrustedApplicationAddress::VirtioVsock(addr) => {
                info!("Test Concat TEE service at vsock address: {:?}", addr);
                ("127.0.0.1".to_string(), 8081u16)
            }
        };

        // Fetch evidence and endorsements from the launcher
        let endorsed_evidence = launcher.get_endorsed_evidence().await.ok();

        // Convert from oak_proto_rust types to kms_proto types using the convert() method
        let evidence = endorsed_evidence.as_ref()
            .and_then(|ee| ee.evidence.as_ref())
            .and_then(|ev| ev.convert().ok());

        let endorsements = endorsed_evidence.as_ref()
            .and_then(|ee| ee.endorsements.as_ref())
            .and_then(|en| en.convert().ok());

        if evidence.is_some() {
            info!("✓ Retrieved TEE evidence and endorsements from launcher");
        } else {
            info!("⚠ No evidence/endorsements available from launcher");
        }

        let instance_id = Uuid::new_v4().to_string();
        let tee_service = TestConcatService {
            instance_id,
            address: host,
            port,
            process_state: ProcessState {
                pid: 0,
                start_time: Utc::now(),
                timeout: 300,
                service_type: ServiceType::TestConcat,
                status: ProcessStatus::Running,
            },
        };

        info!("Test Concat TEE service launched at {}:{}", tee_service.address, tee_service.port);

        Ok((launcher, tee_service))
    }

    /// Cleanup all launchers (TEE launchers only, KMS pool is kept alive)
    pub async fn cleanup(&self) -> Result<()> {
        // Clean up TEE launchers
        let mut tee_launchers = self.tee_launchers.lock().await;
        for mut launcher in tee_launchers.drain(..) {
            let _ = launcher.kill().await;
        }

        // Note: KMS pool is NOT cleaned up here as it's shared across sessions
        // To clean up the KMS pool, call cleanup_kms_pool() explicitly
        Ok(())
    }

    /// Cleanup the entire KMS pool (should only be called at server shutdown)
    pub async fn cleanup_kms_pool(&self) -> Result<()> {
        info!("Shutting down KMS pool...");
        let mut pool = self.kms_pool.lock().await;
        for mut entry in pool.drain(..) {
            let _ = entry.launcher.kill().await;
        }
        info!("KMS pool shut down");
        Ok(())
    }
}
