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

//! ProgramExecutorClient - gRPC client for communicating with ProgramExecutorTee.
//!
//! This client follows the protocol demonstrated in testing_base.h:
//! 1. Send WriteConfiguration requests (model files, private_state)
//! 2. Send InitializeRequest with ProgramExecutorTeeInitializeConfig
//! 3. Start Session with Configure → Finalize

use anyhow::{anyhow, Result};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use prost::Message;
use std::time::Duration;
use tokio::sync::mpsc;
use tonic::transport::Channel;
use tracing::{debug, error, info};

// Proto imports
use any_proto::google::protobuf::Any;
use confidential_transform_proto::fcp::confidentialcompute::{
    confidential_transform_client::ConfidentialTransformClient,
    session_request::Kind as SessionRequestKind, session_response::Kind as SessionResponseKind,
    stream_initialize_request::Kind as StreamInitializeRequestKind, ConfigurationMetadata,
    ConfigureRequest, FinalizeRequest, InitializeRequest, InitializeResponse, SessionRequest,
    StreamInitializeRequest, WriteConfigurationRequest,
};
use crypto_proto::oak::crypto::v1::EncryptedRequest;
use program_executor_tee_proto::fcp::confidentialcompute::ProgramExecutorTeeInitializeConfig;
use reference_value_proto::oak::attestation::v1::ReferenceValues;

/// The configuration ID used for KMS private state (decryption keys).
/// Must match kPrivateStateConfigId in fcp/confidentialcompute/private_state.h
const PRIVATE_STATE_CONFIG_ID: &str = "pipeline_private_state";

/// Maximum number of concurrent sessions.
const MAX_NUM_SESSIONS: u32 = 8;

/// ProgramExecutorClient manages communication with the ProgramExecutorTee container.
pub struct ProgramExecutorClient {
    client: ConfidentialTransformClient<Channel>,
}

impl ProgramExecutorClient {
    /// Creates a new ProgramExecutorClient and connects to the specified address.
    pub async fn new(address: &str) -> Result<Self> {
        info!("Connecting to ProgramExecutorTee at {}", address);
        let channel = Channel::from_shared(address.to_string())
            .map_err(|e| anyhow!("Invalid address '{}': {}", address, e))?
            .connect_timeout(Duration::from_secs(30))
            .connect()
            .await
            .map_err(|e| {
                anyhow!("Failed to connect to ProgramExecutorTee at '{}': {}", address, e)
            })?;

        Ok(ProgramExecutorClient { client: ConfidentialTransformClient::new(channel) })
    }

    /// Initializes the ProgramExecutorTee with a Python program and configuration.
    ///
    /// # Arguments
    /// * `protected_response` - Encrypted decryption keys from KMS (encrypted with TEE's public key)
    /// * `program` - Raw Python code (will be base64 encoded)
    /// * `client_ids` - List of client IDs for data access
    /// * `client_data_dir` - Directory path for client data
    /// * `outgoing_server_address` - FakeDataReadWriteService gRPC address
    /// * `worker_bns_addresses` - Worker TEE addresses for distributed execution (empty = single-node)
    /// * `worker_reference_values` - Reference values for worker attestation (required if workers provided)
    /// * `kms_private_state` - Serialized private state containing decryption keys
    /// * `model_files` - Model files to upload: vec![(config_id, file_content)]
    pub async fn initialize_with_program(
        &mut self,
        protected_response: EncryptedRequest,
        program: &str,
        client_ids: Vec<String>,
        client_data_dir: &str,
        outgoing_server_address: &str,
        worker_bns_addresses: Vec<String>,
        worker_reference_values: Option<ReferenceValues>,
        kms_private_state: Vec<u8>,
        model_files: Vec<(String, Vec<u8>)>,
    ) -> Result<InitializeResponse> {
        info!("Initializing ProgramExecutorTee");
        info!("  Program size: {} bytes", program.len());
        info!("  Client IDs: {:?}", client_ids);
        info!("  Client data dir: {}", client_data_dir);
        info!("  Outgoing server address: {}", outgoing_server_address);
        info!("  Worker BNS addresses: {:?}", worker_bns_addresses);
        info!("  Model files: {}", model_files.len());
        info!("  KMS private state: {} bytes", kms_private_state.len());

        let (tx, rx) = mpsc::unbounded_channel();

        // Step 1: Send WriteConfiguration requests for model files
        for (config_id, content) in &model_files {
            info!("  Uploading model file '{}': {} bytes", config_id, content.len());
            let write_config = WriteConfigurationRequest {
                first_request_metadata: Some(ConfigurationMetadata {
                    configuration_id: config_id.clone(),
                    total_size_bytes: content.len() as i64,
                }),
                data: content.clone(),
                commit: true,
            };

            tx.send(StreamInitializeRequest {
                kind: Some(StreamInitializeRequestKind::WriteConfiguration(write_config)),
            })
            .map_err(|e| anyhow!("Failed to send model file '{}': {}", config_id, e))?;
        }

        // Step 2: Send WriteConfiguration for private_state (KMS decryption keys)
        info!("  Uploading private_state: {} bytes", kms_private_state.len());
        let private_state_config = WriteConfigurationRequest {
            first_request_metadata: Some(ConfigurationMetadata {
                configuration_id: PRIVATE_STATE_CONFIG_ID.to_string(),
                total_size_bytes: kms_private_state.len() as i64,
            }),
            data: kms_private_state,
            commit: true,
        };

        tx.send(StreamInitializeRequest {
            kind: Some(StreamInitializeRequestKind::WriteConfiguration(private_state_config)),
        })
        .map_err(|e| anyhow!("Failed to send private_state: {}", e))?;

        // Step 3: Create ProgramExecutorTeeInitializeConfig
        let program_base64 = BASE64.encode(program.as_bytes());
        debug!("  Program base64 length: {} bytes", program_base64.len());

        let init_config = ProgramExecutorTeeInitializeConfig {
            program: program_base64.into_bytes(),
            client_ids,
            client_data_dir: client_data_dir.to_string(),
            outgoing_server_address: outgoing_server_address.to_string(),
            attester_id: "fake_attester".to_string(),
            worker_bns_addresses,
            reference_values: worker_reference_values,
            ..Default::default()
        };

        // Pack config into Any
        let config_any = Any {
            type_url:
                "type.googleapis.com/fcp.confidentialcompute.ProgramExecutorTeeInitializeConfig"
                    .to_string(),
            value: init_config.encode_to_vec(),
        };

        // Step 4: Create InitializeRequest with protected_response
        let initialize_request = InitializeRequest {
            max_num_sessions: MAX_NUM_SESSIONS,
            protected_response: Some(protected_response),
            configuration: Some(config_any),
            ..Default::default()
        };

        tx.send(StreamInitializeRequest {
            kind: Some(StreamInitializeRequestKind::InitializeRequest(initialize_request)),
        })
        .map_err(|e| anyhow!("Failed to send initialize request: {}", e))?;

        // Close sender to signal end of stream
        drop(tx);

        // Send the stream and get response
        let request_stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);
        let response = self
            .client
            .stream_initialize(request_stream)
            .await
            .map_err(|status| {
                let msg = format!(
                    "StreamInitialize failed - Code: {:?}, Message: {}",
                    status.code(),
                    status.message()
                );
                error!("{}", msg);
                anyhow!(msg)
            })?
            .into_inner();

        info!("ProgramExecutorTee initialized successfully");
        info!("  Public key: {} bytes", response.public_key.len());

        Ok(response)
    }

    /// Executes the session: Configure → Finalize.
    ///
    /// The Python program runs during Finalize, reading data from FakeDataReadWriteService
    /// and writing results back.
    ///
    /// # Returns
    /// * `(results, release_token)` - Results data and release token for KMS
    pub async fn execute_session(&mut self) -> Result<(Vec<u8>, Vec<u8>)> {
        info!("Starting session execution");

        let (tx, rx) = mpsc::unbounded_channel();
        let request_stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);

        // Send Configure request before calling session()
        let configure_request = ConfigureRequest { chunk_size: 4096, ..Default::default() };

        tx.send(SessionRequest { kind: Some(SessionRequestKind::Configure(configure_request)) })
            .map_err(|e| anyhow!("Failed to send configure request: {}", e))?;

        // Start session RPC
        let mut response_stream = self
            .client
            .session(request_stream)
            .await
            .map_err(|status| {
                let msg = format!(
                    "Session RPC failed - Code: {:?}, Message: {}",
                    status.code(),
                    status.message()
                );
                error!("{}", msg);
                anyhow!(msg)
            })?
            .into_inner();

        // Wait for Configure response
        info!("Waiting for Configure response...");
        loop {
            let response = response_stream
                .message()
                .await
                .map_err(|e| anyhow!("Failed to receive Configure response: {}", e))?
                .ok_or_else(|| anyhow!("Stream closed before Configure response"))?;

            match response.kind {
                Some(SessionResponseKind::Configure(config_resp)) => {
                    info!("Configure response received, nonce: {} bytes", config_resp.nonce.len());
                    break;
                }
                Some(SessionResponseKind::Read(read_resp)) => {
                    debug!(
                        "Skipping ReadResponse during Configure phase: {} bytes",
                        read_resp.data.len()
                    );
                }
                other => {
                    return Err(anyhow!("Expected Configure response, got: {:?}", other));
                }
            }
        }

        // Send Finalize request - this triggers Python program execution
        info!("Sending Finalize request (triggers program execution)...");
        let finalize_request = FinalizeRequest::default();

        tx.send(SessionRequest { kind: Some(SessionRequestKind::Finalize(finalize_request)) })
            .map_err(|e| anyhow!("Failed to send finalize request: {}", e))?;

        // Close sender
        drop(tx);

        // Collect results
        info!("Waiting for results...");
        let mut results = Vec::new();
        let mut release_token = Vec::new();

        loop {
            let response = response_stream
                .message()
                .await
                .map_err(|e| anyhow!("Failed to receive Finalize response: {}", e))?
                .ok_or_else(|| anyhow!("Stream closed before Finalize response"))?;

            match response.kind {
                Some(SessionResponseKind::Read(read_resp)) => {
                    info!(
                        "ReadResponse: {} bytes, finish_read={}",
                        read_resp.data.len(),
                        read_resp.finish_read
                    );
                    results.extend_from_slice(&read_resp.data);
                }
                Some(SessionResponseKind::Finalize(finalize_resp)) => {
                    info!("FinalizeResponse received");
                    release_token = finalize_resp.release_token;
                    info!("  Release token: {} bytes", release_token.len());
                    break;
                }
                other => {
                    return Err(anyhow!("Expected Read or Finalize response, got: {:?}", other));
                }
            }
        }

        info!(
            "Session complete: {} bytes results, {} bytes release_token",
            results.len(),
            release_token.len()
        );

        Ok((results, release_token))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base64_encoding() {
        let program = "def trusted_program(ip, esh): pass";
        let encoded = BASE64.encode(program.as_bytes());
        println!("Original: {}", program);
        println!("Base64: {}", encoded);
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_config_serialization() {
        let config = ProgramExecutorTeeInitializeConfig {
            program: BASE64.encode("test".as_bytes()).into_bytes(),
            client_ids: vec!["client1".to_string()],
            client_data_dir: "/data".to_string(),
            outgoing_server_address: "[::1]:8080".to_string(),
            attester_id: "test".to_string(),
            ..Default::default()
        };

        let encoded = config.encode_to_vec();
        println!("Config serialized: {} bytes", encoded.len());
        println!("Config hex: {:02x?}", &encoded[..encoded.len().min(32)]);
        assert!(!encoded.is_empty());
    }
}
