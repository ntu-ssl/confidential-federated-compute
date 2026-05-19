// Copyright 2026 Google LLC.
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

mod launcher;

use anyhow::Result;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = launcher::Args::parse()?;

    info!("Starting Storage Proxy Launcher");
    tracing::debug!("Arguments: {:?}", args);

    let mut launcher = launcher::Launcher::create(args).await?;

    // Wait for the VM to signal that the app is ready.
    let trusted_app_address = launcher.get_trusted_app_address().await?;
    info!("Storage Proxy is ready at {}", trusted_app_address);

    // Keep the launcher running until the VM exits or we are interrupted.
    launcher.wait().await?;

    Ok(())
}
