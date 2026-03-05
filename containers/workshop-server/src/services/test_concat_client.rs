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

use anyhow::{anyhow, Context, Result};
use crypto_proto::oak::crypto::v1::EncryptedRequest;
use prost::Message;
use std::time::Duration;
use tokio::sync::mpsc;
use tonic::transport::Channel;
use tracing::{debug, error, info};

use confidential_transform_proto::fcp::confidentialcompute::{
    confidential_transform_client::ConfidentialTransformClient,
    stream_initialize_request::Kind as StreamInitializeRequestKind,
    session_request::Kind as SessionRequestKind,
    BlobMetadata, ConfigureRequest, FinalizeRequest, InitializeRequest, InitializeResponse,
    SessionRequest, StreamInitializeRequest, WriteRequest,
};

/// TestConcatClient manages communication with the test_concat TEE container.
pub struct TestConcatClient {
    client: ConfidentialTransformClient<Channel>,
}

impl TestConcatClient {
    /// Creates a new TestConcatClient and connects to the specified address.
    pub async fn new(address: &str) -> Result<Self> {
        info!("Connecting to test_concat ConfidentialTransform at {}", address);
        let channel = Channel::from_shared(address.to_string())
            .map_err(|e| anyhow!("Invalid test_concat address '{}': {}", address, e))?
            .connect_timeout(Duration::from_secs(10))
            .connect()
            .await
            .map_err(|e| anyhow!(
                "Failed to connect to test_concat ConfidentialTransform at '{}': {}",
                address,
                e
            ))?;

        Ok(TestConcatClient {
            client: ConfidentialTransformClient::new(channel),
        })
    }

    /// Initializes the test_concat container with encrypted decryption keys from KMS.
    pub async fn initialize_with_protected_response(
        &mut self,
        protected_response: EncryptedRequest,
    ) -> Result<InitializeResponse> {
        info!(
            "Initializing test_concat with protected response: {} bytes encrypted message",
            protected_response
                .encrypted_message
                .as_ref()
                .map(|m| m.ciphertext.len())
                .unwrap_or(0)
        );

        // Create initialize request
        let initialize_request = InitializeRequest {
            max_num_sessions: 8,
            protected_response: Some(protected_response),
            ..Default::default()
        };

        // Create stream and send initialize request
        let (tx, rx) = mpsc::unbounded_channel();

        info!("Sending StreamInitializeRequest to test_concat");

        // Send the initialize request
        tx.send(StreamInitializeRequest {
            kind: Some(StreamInitializeRequestKind::InitializeRequest(
                initialize_request,
            )),
        })
        .map_err(|e| anyhow!("Failed to send initialize request to test_concat: {}", e))?;

        info!("Closing sender to signal end of initialize stream");
        // CRITICAL: Drop the sender to signal end of stream
        // This tells the server we're done sending requests
        drop(tx);

        // Convert channel to stream
        let request_stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);

        info!("Calling stream_initialize on test_concat");
        let response = self
            .client
            .stream_initialize(request_stream)
            .await
            .map_err(|status| {
                let error_msg = format!(
                    "Failed to call stream_initialize - Code: {:?}, Message: {}",
                    status.code(),
                    status.message()
                );
                error!("{}", error_msg);
                anyhow!(error_msg)
            })?
            .into_inner();

        info!("test_concat initialized successfully");

        Ok(response)
    }

    /// Sends multiple encrypted data chunks to test_concat in a single session.
    /// Returns (encrypted_results, release_token) tuple.
    pub async fn process_with_encrypted_data(
        &mut self,
        blob_metadatas: Vec<Option<BlobMetadata>>,
        encrypted_data_chunks: Vec<Vec<u8>>,
    ) -> Result<(Vec<u8>, Vec<u8>)> {
        info!(
            "Sending {} encrypted blob chunks to test_concat for processing",
            encrypted_data_chunks.len()
        );

        for (idx, chunk) in encrypted_data_chunks.iter().enumerate() {
            info!("  Chunk {}: {} bytes", idx + 1, chunk.len());
        }

        // Create a channel for bidirectional streaming
        let (tx, rx) = mpsc::unbounded_channel();
        info!("Created mpsc channel for session streaming");

        // Convert channel to stream
        let request_stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);
        info!("✓ Converted mpsc channel to UnboundedReceiverStream");

        // CRITICAL: Send the first message BEFORE awaiting on session()
        // The gRPC server won't start processing until it receives the first message,
        // so we must send Configure through the channel first
        info!("→ Step 1: Creating Configure request (BEFORE session() RPC)");
        let configure_request = ConfigureRequest {
            chunk_size: 4096,
            ..Default::default()
        };
        info!("✓ Configure request created");

        info!("→ Sending Configure request via tx.send() (BEFORE session() call)");
        tx.send(SessionRequest {
            kind: Some(SessionRequestKind::Configure(configure_request)),
        })
        .map_err(|e| {
            let error_msg = format!("Failed to send initial configure request: {}", e);
            error!("✗ {}", error_msg);
            anyhow!(error_msg)
        })?;
        info!("✓ Configure request sent - NOW safe to call session() RPC");

        // Now we can safely await on session() because the server has the first message
        info!("→ Calling session() RPC...");
        let mut response_stream = self
            .client
            .session(request_stream)
            .await
            .map_err(|status| {
                let error_msg = format!(
                    "Failed to call session - Code: {:?}, Message: {}",
                    status.code(),
                    status.message()
                );
                error!("✗ Session RPC failed: {}", error_msg);
                anyhow!(error_msg)
            })?
            .into_inner();
        info!("✓ Session RPC established successfully");

        // Wait for Configure response
        // According to proto spec, server may send multiple ReadResponses before ConfigureResponse
        info!("→ Waiting for Configure response from test_concat...");
        loop {
            let response = response_stream
                .message()
                .await
                .map_err(|e| {
                    let error_msg = format!("Failed to receive Configure response: {}", e);
                    error!("✗ {}", error_msg);
                    anyhow!(error_msg)
                })?
                .ok_or_else(|| {
                    let error_msg = "No configure response from test_concat (stream closed unexpectedly)";
                    error!("✗ {}", error_msg);
                    anyhow!(error_msg)
                })?;

            match response.kind {
                Some(confidential_transform_proto::fcp::confidentialcompute::session_response::Kind::Configure(config_resp)) => {
                    info!("✓ Configure response received");
                    break;
                }
                Some(confidential_transform_proto::fcp::confidentialcompute::session_response::Kind::Read(read_resp)) => {
                    info!("  Skipping ReadResponse during Configure phase ({} bytes)", read_resp.data.len());
                }
                _ => {
                    return Err(anyhow!("Expected Configure response, got unexpected type: {:?}", response.kind));
                }
            }
        }
        info!("✓ Configure response fully processed");

        // Step 2: Send write requests for each encrypted data chunk
        info!("→ Step 2: Sending {} Write requests with encrypted data", encrypted_data_chunks.len());
        for (idx, encrypted_data) in encrypted_data_chunks.iter().enumerate() {
            info!("  → Sending Write request {}: {} bytes", idx + 1, encrypted_data.len());

            let write_request = WriteRequest {
                first_request_metadata: blob_metadatas.get(idx).cloned().flatten(),
                commit: true,
                data: encrypted_data.to_vec(),
                ..Default::default()
            };

            tx.send(SessionRequest {
                kind: Some(SessionRequestKind::Write(write_request)),
            })
            .map_err(|e| {
                let error_msg = format!("Failed to send write request {}: {}", idx + 1, e);
                error!("✗ {}", error_msg);
                anyhow!(error_msg)
            })?;
            info!("  ✓ Write request {} sent", idx + 1);

            // Wait for Write response after each write
            info!("  → Waiting for Write response {} from test_concat...", idx + 1);
            loop {
                let response = response_stream
                    .message()
                    .await
                    .map_err(|e| {
                        let error_msg = format!("Failed to receive Write response {}: {}", idx + 1, e);
                        error!("✗ {}", error_msg);
                        anyhow!(error_msg)
                    })?
                    .ok_or_else(|| {
                        let error_msg = format!("No write response {} from test_concat (stream closed unexpectedly)", idx + 1);
                        error!("✗ {}", error_msg);
                        anyhow!(error_msg)
                    })?;

                match response.kind {
                    Some(confidential_transform_proto::fcp::confidentialcompute::session_response::Kind::Write(write_resp)) => {
                        info!("  ✓ Write response {} received", idx + 1);
                        info!(
                            "    Write response: {} bytes committed, status code: {:?}",
                            write_resp.committed_size_bytes, write_resp.status.as_ref().map(|s| s.code)
                        );
                        if let Some(status) = write_resp.status {
                            if status.code != 0 {
                                return Err(anyhow!(
                                    "Write operation {} failed with status code: {}, message: {}",
                                    idx + 1,
                                    status.code,
                                    status.message
                                ));
                            }
                        }
                        break;
                    }
                    Some(confidential_transform_proto::fcp::confidentialcompute::session_response::Kind::Read(read_resp)) => {
                        info!("    Skipping ReadResponse during Write {} phase ({} bytes)", idx + 1, read_resp.data.len());
                    }
                    _ => {
                        return Err(anyhow!("Expected Write response {}, got unexpected type: {:?}", idx + 1, response.kind));
                    }
                }
            }
            info!("  ✓ Write response {} fully processed", idx + 1);
        }
        info!("✓ All {} write requests processed", encrypted_data_chunks.len());

        // Step 3: Send finalize request to complete the session
        info!("→ Step 3: Creating Finalize request");
        let finalize_request = FinalizeRequest {
            ..Default::default()
        };
        info!("✓ Finalize request created");

        info!("→ About to send Finalize request via tx.send()");
        tx.send(SessionRequest {
            kind: Some(SessionRequestKind::Finalize(finalize_request)),
        })
        .map_err(|e| {
            let error_msg = format!("Failed to send finalize request: {}", e);
            error!("✗ {}", error_msg);
            anyhow!(error_msg)
        })?;
        info!("✓ Finalize request sent successfully");

        // Close the sender to signal end of stream
        // CRITICAL: Must close after final request so server knows we're done sending
        info!("→ Closing sender to signal end of session stream (dropping tx)");
        drop(tx);
        info!("✓ Sender dropped successfully");

        // Wait for Finalize response
        // According to proto spec, server sends multiple ReadResponses then FinalizeResponse
        info!("→ Waiting for Finalize responses (ReadResponses + FinalizeResponse) from test_concat...");
        let mut encrypted_results = Vec::new();
        let mut release_token = Vec::new();

        loop {
            let response = response_stream
                .message()
                .await
                .map_err(|e| {
                    let error_msg = format!("Failed to receive Finalize response: {}", e);
                    error!("✗ {}", error_msg);
                    anyhow!(error_msg)
                })?
                .ok_or_else(|| {
                    let error_msg = "No finalize response from test_concat (stream closed unexpectedly)";
                    error!("✗ {}", error_msg);
                    anyhow!(error_msg)
                })?;

            match response.kind {
                Some(confidential_transform_proto::fcp::confidentialcompute::session_response::Kind::Read(read_response)) => {
                    info!("  Received ReadResponse with {} bytes (finish_read={})",
                        read_response.data.len(), read_response.finish_read);
                    encrypted_results.extend_from_slice(&read_response.data);

                    if read_response.finish_read {
                        info!("  ReadResponse indicates end of data");
                    }
                }
                Some(confidential_transform_proto::fcp::confidentialcompute::session_response::Kind::Finalize(finalize_resp)) => {
                    info!("✓ FinalizeResponse received - session complete");
                    release_token = finalize_resp.release_token;
                    info!("  Release token size: {} bytes", release_token.len());
                    break;
                }
                _ => {
                    return Err(anyhow!("Expected Read or Finalize response, got unexpected type: {:?}", response.kind));
                }
            }
        }

        info!("✓ Finalize response fully processed");
        info!("✓ test_concat processing complete: {} bytes result, {} bytes release_token",
            encrypted_results.len(), release_token.len());

        // Return encrypted results and release token for KMS ReleaseResults call
        Ok((encrypted_results, release_token))
    }
}
