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

#![feature(let_chains)]

mod launcher_module;
mod models;
mod server;
mod services;

use anyhow::Result;
use clap::Parser;
use services::{LauncherManager, SessionRegistry};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tracing_subscriber;

#[derive(Parser, Debug)]
#[command(name = "Workshop Server")]
#[command(about = "REST API server for running KMS Flow Simulation experiments")]
struct Args {
    /// Path to the system image for QEMU
    #[arg(long, default_value = "/usr/share/oak/oak_system.img")]
    system_image: PathBuf,

    /// Path to the KMS container bundle
    #[arg(long, default_value = "/usr/share/oak/kms_bundle.tar")]
    kms_bundle: PathBuf,

    /// Path to the Test Concat TEE container bundle
    #[arg(long, default_value = "/usr/share/oak/test_concat_bundle.tar")]
    test_concat_bundle: PathBuf,

    /// Path to the VMM binary
    #[arg(long, default_value = "/usr/bin/qemu-system-x86_64")]
    vmm_binary: PathBuf,

    /// Path to the stage0 binary
    #[arg(long, default_value = "/usr/share/oak/stage0.bin")]
    stage0_binary: PathBuf,

    /// Path to the kernel
    #[arg(long, default_value = "/usr/share/oak/bzImage")]
    kernel: PathBuf,

    /// Path to the initrd
    #[arg(long, default_value = "/usr/share/oak/initrd")]
    initrd: PathBuf,

    /// Ramdrive size in kilobytes
    #[arg(long, default_value = "1000000")]
    ramdrive_size: u32,

    /// VM type (Default, Sev, SevEs, SevSnp, Tdx)
    #[arg(long, default_value = "sev-snp")]
    vm_type: String,

    /// Server bind address
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Server bind port
    #[arg(long, default_value = "3000")]
    port: u16,

    /// Suppress QEMU boot logs for all VMs (KMS and TEE)
    #[arg(long, default_value_t = false)]
    quiet: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let args = Args::parse();

    tracing::info!("Starting workshop server");
    tracing::info!(
        "System image: {}",
        args.system_image.display()
    );
    tracing::info!("KMS bundle: {}", args.kms_bundle.display());
    tracing::info!(
        "Test Concat bundle: {}",
        args.test_concat_bundle.display()
    );

    // Create session registry and launcher manager
    let registry = Arc::new(SessionRegistry::new());
    let launcher_manager = Arc::new(LauncherManager::new(
        args.system_image,
        args.kms_bundle,
        args.test_concat_bundle,
        args.vmm_binary,
        args.stage0_binary,
        args.kernel,
        args.initrd,
        args.ramdrive_size,
        args.vm_type,
        args.quiet,
    ));

    // Initialize KMS pool with 5 pre-started instances
    tracing::info!("Initializing KMS pool...");
    launcher_manager.initialize_kms_pool().await?;
    tracing::info!("KMS pool initialized successfully");

    // Create the router
    let app = server::create_router(registry, launcher_manager.clone());

    // Bind and run the server
    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    tracing::info!("Server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
