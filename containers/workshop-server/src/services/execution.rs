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

use crate::models::*;
use crate::services::{ClientSimulator, KmsClient, SessionRegistry, TestConcatClient};
use crate::services::kms_client::ProstProtoConversionExt;
use anyhow::{anyhow, Result};
use prost::Message;
use serde_json::json;
use sha2::Sha256;
use std::sync::Arc;
use tracing::{info, warn};

/// Execution engine for running simulation steps
pub struct ExecutionEngine;

impl ExecutionEngine {
    /// Rotate a keyset in KMS
    pub async fn rotate_keyset(
        context: &mut SessionContext,
        keyset_id: u64,
        ttl_seconds: i64,
    ) -> Result<()> {
        info!("Rotating keyset: {}", keyset_id);

        context.keyset_id = Some(keyset_id);

        // Only call KMS if it's running in this session
        if let Some(kms_service) = &context.kms_service {
            let kms_address = format!("http://{}:{}", kms_service.address, kms_service.port);
            match KmsClient::new(&kms_address).await {
                Ok(mut kms_client) => {
                    match kms_client.rotate_keyset(keyset_id, ttl_seconds).await {
                        Ok(_) => {
                            info!("✓ Keyset rotated successfully via KMS");
                            context.log_event(
                                "rotate_keyset".to_string(),
                                json!({"keyset_id": keyset_id, "ttl_seconds": ttl_seconds}),
                                json!({"status": "rotated"}),
                                ExecutionStatus::Success,
                                None,
                            );
                            Ok(())
                        }
                        Err(e) => {
                            let err_msg = format!("KMS rotate_keyset failed: {}", e);
                            context.log_event(
                                "rotate_keyset".to_string(),
                                json!({"keyset_id": keyset_id}),
                                json!({}),
                                ExecutionStatus::Failure,
                                Some(err_msg.clone()),
                            );
                            Err(anyhow!(err_msg))
                        }
                    }
                }
                Err(e) => {
                    let err_msg = format!("Failed to connect to KMS: {}", e);
                    context.log_event(
                        "rotate_keyset".to_string(),
                        json!({"keyset_id": keyset_id}),
                        json!({}),
                        ExecutionStatus::Failure,
                        Some(err_msg.clone()),
                    );
                    Err(anyhow!(err_msg))
                }
            }
        } else {
            info!("No KMS service running in this session, skipping KMS call");
            Ok(())
        }
    }

    /// Derive keys from a data access policy
    pub async fn derive_keys(
        context: &mut SessionContext,
        policy_name: String,
    ) -> Result<Vec<Vec<u8>>> {
        info!("Deriving keys for data access policy: {}", policy_name);

        // Retrieve the data access policy from cache
        let policy_cache = context
            .data_access_policies
            .get(&policy_name)
            .ok_or_else(|| anyhow!("Data access policy not found: {}", policy_name))?;

        // Get policy hash (already computed when policy was created)
        let policy_hash = policy_cache.policy_hash.clone();

        let public_keys = if let Some(kms_service) = &context.kms_service {
            let kms_address = format!("http://{}:{}", kms_service.address, kms_service.port);
            match KmsClient::new(&kms_address).await {
                Ok(mut kms_client) => {
                    let keyset_id = context.keyset_id.unwrap_or(1u64);
                    match kms_client.derive_keys(keyset_id, vec![policy_hash.clone()]).await {
                        Ok(keys) => {
                            info!("✓ Derived {} public keys from KMS", keys.len());
                            context.log_event(
                                "derive_keys".to_string(),
                                json!({"policy_name": policy_name, "policy_hash": hex::encode(&policy_hash)}),
                                json!({"public_keys_count": keys.len()}),
                                ExecutionStatus::Success,
                                None,
                            );
                            keys
                        }
                        Err(e) => {
                            let err_msg = format!("KMS derive_keys failed: {}", e);
                            context.log_event(
                                "derive_keys".to_string(),
                                json!({"policy_name": policy_name}),
                                json!({}),
                                ExecutionStatus::Failure,
                                Some(err_msg.clone()),
                            );
                            return Err(anyhow!(err_msg));
                        }
                    }
                }
                Err(e) => {
                    let err_msg = format!("Failed to connect to KMS: {}", e);
                    return Err(anyhow!(err_msg));
                }
            }
        } else {
            info!("No KMS service running, using dummy public key");
            vec![vec![1, 2, 3, 4, 5]]
        };

        context.public_keys = public_keys.clone();
        Ok(public_keys)
    }

    /// Register a pipeline invocation with KMS
    pub async fn register_pipeline(
        context: &mut SessionContext,
        invocation_name: String,
        logical_pipeline_name: String,
        data_access_policy_name: String,
        keyset_id: u64,
        ttl_seconds: i64,
    ) -> Result<Vec<u8>> {
        info!("Registering pipeline invocation: {}", invocation_name);

        // Get data access policy (contains authorized logical pipeline with variants)
        let data_access_policy_cache = context
            .data_access_policies
            .get(&data_access_policy_name)
            .ok_or_else(|| anyhow!("Data access policy not found: {}", data_access_policy_name))?;

        // Decode the DataAccessPolicy to extract the variant
        use prost::Message;
        use access_policy_proto::fcp::confidentialcompute::DataAccessPolicy;

        let data_access_policy = DataAccessPolicy::decode(data_access_policy_cache.proto_bytes.as_slice())
            .map_err(|e| anyhow!("Failed to decode data access policy: {}", e))?;

        // Extract the first variant from the logical pipeline
        // This ensures the variant bytes we pass match what's inside the DataAccessPolicy
        let variant_policy_bytes = data_access_policy
            .pipelines
            .get(&logical_pipeline_name)
            .and_then(|logical_policy| logical_policy.instances.first())
            .ok_or_else(|| anyhow!("No variant found in logical pipeline: {}", logical_pipeline_name))?
            .encode_to_vec();

        let invocation_id = if let Some(kms_service) = &context.kms_service {
            let kms_address = format!("http://{}:{}", kms_service.address, kms_service.port);
            match KmsClient::new(&kms_address).await {
                Ok(mut kms_client) => {
                    match kms_client
                        .register_pipeline_invocation(
                            logical_pipeline_name.clone(),
                            variant_policy_bytes,
                            vec![keyset_id],
                            vec![data_access_policy_cache.proto_bytes.clone()],
                            ttl_seconds,
                        )
                        .await
                    {
                        Ok((inv_id, public_keys)) => {
                            info!("✓ Pipeline registered with invocation_id: {}", hex::encode(&inv_id));
                            // Store public keys for encryption
                            context.public_keys = public_keys.clone();
                            context.log_event(
                                "register_pipeline".to_string(),
                                json!({"invocation_name": invocation_name, "logical_pipeline_name": logical_pipeline_name, "data_access_policy_name": data_access_policy_name}),
                                json!({"invocation_id": hex::encode(&inv_id), "public_keys_count": public_keys.len()}),
                                ExecutionStatus::Success,
                                None,
                            );
                            inv_id
                        }
                        Err(e) => {
                            let err_msg = format!("KMS register_pipeline_invocation failed: {}", e);
                            context.log_event(
                                "register_pipeline".to_string(),
                                json!({"invocation_name": invocation_name}),
                                json!({}),
                                ExecutionStatus::Failure,
                                Some(err_msg.clone()),
                            );
                            return Err(anyhow!(err_msg));
                        }
                    }
                }
                Err(e) => {
                    return Err(anyhow!("Failed to connect to KMS: {}", e));
                }
            }
        } else {
            info!("No KMS service running, generating dummy invocation ID");
            uuid::Uuid::new_v4().as_bytes().to_vec()
        };

        // Cache the invocation
        let invocation_cache = InvocationCache {
            name: invocation_name.clone(),
            logical_pipeline_name: logical_pipeline_name.clone(),
            data_access_policy_name: data_access_policy_name.clone(),
            invocation_id: invocation_id.clone(),
            keyset_id,
        };
        context
            .invocations
            .insert(invocation_cache.name.clone(), invocation_cache);

        Ok(invocation_id)
    }

    /// Authorize a transform with KMS
    /// Returns the encrypted protected response (EncryptedRequest) serialized to bytes
    pub async fn authorize_transform(
        context: &mut SessionContext,
        invocation_name: String,
        registry: Arc<SessionRegistry>,
        session_id: crate::models::SessionId,
    ) -> Result<Vec<u8>> {
        info!("Authorizing transform for invocation: {}", invocation_name);

        let invocation = context
            .invocations
            .get(&invocation_name)
            .ok_or_else(|| anyhow!("Invocation not found: {}", invocation_name))?;

        let protected_response_bytes = if let Some(kms_service) = &context.kms_service {
            let kms_address = format!("http://{}:{}", kms_service.address, kms_service.port);
            match KmsClient::new(&kms_address).await {
                Ok(mut kms_client) => {
                    // Get data access policy and extract the variant for this invocation
                    let data_access_policy_cache = context
                        .data_access_policies
                        .get(&invocation.data_access_policy_name)
                        .ok_or_else(|| anyhow!("Data access policy not found: {}", invocation.data_access_policy_name))?;

                    // Decode the DataAccessPolicy to extract the variant
                    use prost::Message;
                    use access_policy_proto::fcp::confidentialcompute::DataAccessPolicy;

                    let data_access_policy = DataAccessPolicy::decode(data_access_policy_cache.proto_bytes.as_slice())
                        .map_err(|e| anyhow!("Failed to decode data access policy: {}", e))?;

                    // Extract the variant from the logical pipeline (same as in register_pipeline)
                    let variant_policy_bytes = data_access_policy
                        .pipelines
                        .get(&invocation.logical_pipeline_name)
                        .and_then(|logical_policy| logical_policy.instances.first())
                        .ok_or_else(|| anyhow!("No variant found in logical pipeline: {}", invocation.logical_pipeline_name))?
                        .encode_to_vec();

                    let policy_bytes = variant_policy_bytes;

                    // Fetch real evidence and endorsements from the TEE launcher
                    let endorsed_evidence_opt = registry.get_tee_endorsed_evidence(session_id).await;

                    let (tee_evidence, tee_endorsements) = if let Some(endorsed_evidence) = endorsed_evidence_opt {
                        // Convert evidence if present using the ProstProtoConversionExt trait
                        let evidence_opt = if let Some(oak_evidence) = &endorsed_evidence.evidence {
                            match oak_evidence.convert() {
                                Ok(ev) => Some(ev),
                                Err(e) => {
                                    warn!("Failed to convert evidence: {}", e);
                                    None
                                }
                            }
                        } else {
                            None
                        };
                        // Convert endorsements if present using the ProstProtoConversionExt trait
                        let endorsements_opt = if let Some(oak_endorsements) = &endorsed_evidence.endorsements {
                            match oak_endorsements.convert() {
                                Ok(en) => Some(en),
                                Err(e) => {
                                    warn!("Failed to convert endorsements: {}", e);
                                    None
                                }
                            }
                        } else {
                            None
                        };
                        (evidence_opt, endorsements_opt)
                    } else {
                        (None, None)
                    };

                    // Use real evidence and endorsements from the launcher
                    match kms_client
                        .authorize_transform(
                            invocation.invocation_id.clone(),
                            policy_bytes,
                            tee_evidence.clone(),
                            tee_endorsements.clone(),
                        )
                        .await
                    {
                        Ok((encrypted_request, signing_key_endorsement)) => {
                            info!("✓ Transform authorized with real TEE evidence and endorsements");
                            // Serialize the EncryptedRequest to bytes
                            let response_bytes = encrypted_request.encode_to_vec();
                            context.log_event(
                                "authorize_transform".to_string(),
                                json!({"invocation_name": invocation_name, "has_tee_evidence": tee_evidence.is_some()}),
                                json!({"encrypted_request_size": response_bytes.len(), "signing_key_endorsement_size": signing_key_endorsement.len()}),
                                ExecutionStatus::Success,
                                None,
                            );
                            response_bytes
                        }
                        Err(e) => {
                            let err_msg = format!("KMS authorize_transform failed: {}", e);
                            context.log_event(
                                "authorize_transform".to_string(),
                                json!({"invocation_name": invocation_name}),
                                json!({}),
                                ExecutionStatus::Failure,
                                Some(err_msg.clone()),
                            );
                            return Err(anyhow!(err_msg));
                        }
                    }
                }
                Err(e) => {
                    return Err(anyhow!("Failed to connect to KMS: {}", e));
                }
            }
        } else {
            info!("No KMS service running, using dummy protected response");
            format!("protected_response_for_{}", invocation_name)
                .as_bytes()
                .to_vec()
        };

        Ok(protected_response_bytes)
    }

    /// Encrypt data using ClientSimulator
    pub async fn encrypt_data(
        context: &mut SessionContext,
        blob_name: String,
        plaintext: Vec<u8>,
        policy_name: String,
        public_key_index: usize,
    ) -> Result<Vec<u8>> {
        info!("Encrypting data: {} ({} bytes)", blob_name, plaintext.len());

        let policy = context
            .data_access_policies
            .get(&policy_name)
            .ok_or_else(|| anyhow!("Data access policy not found: {}", policy_name))?;

        // Get the public key
        if public_key_index >= context.public_keys.len() && context.public_keys.is_empty() {
            return Err(anyhow!("No public keys available. Call derive_keys first."));
        }

        let public_key = if !context.public_keys.is_empty() {
            &context.public_keys[public_key_index.min(context.public_keys.len() - 1)]
        } else {
            &vec![1, 2, 3, 4, 5]
        };

        // Use ClientSimulator to encrypt
        match ClientSimulator::encrypt_data(
            &blob_name,
            plaintext.clone(),
            public_key,
            &policy.policy_hash,
        ) {
            Ok(client_upload) => {
                // Store encrypted blob information
                let encrypted_data = client_upload.blob_data.data.clone();

                // Store BlobMetadata proto bytes for TEE decryption
                let blob_metadata_proto_bytes = client_upload.blob_data.metadata
                    .as_ref()
                    .map(|m| m.encode_to_vec())
                    .unwrap_or_default();

                let blob_cache = EncryptedBlobCache {
                    name: blob_name.clone(),
                    encrypted_data: encrypted_data.clone(),
                    blob_metadata: json!({
                        "blob_id": hex::encode(&client_upload.blob_header.blob_id),
                        "key_id": hex::encode(&client_upload.blob_header.key_id),
                        "access_policy_sha256": hex::encode(&client_upload.blob_header.access_policy_sha256),
                        "size": plaintext.len(),
                        "encrypted_size": encrypted_data.len()
                    }),
                    blob_metadata_proto_bytes,
                    policy_name,
                };
                context.encrypted_blobs.insert(blob_name.clone(), blob_cache);

                info!("✓ Data encrypted successfully: {} bytes → {} bytes",
                    plaintext.len(), encrypted_data.len());
                context.log_event(
                    "encrypt_data".to_string(),
                    json!({"blob_name": blob_name, "plaintext_size": plaintext.len()}),
                    json!({"encrypted_size": encrypted_data.len()}),
                    ExecutionStatus::Success,
                    None,
                );

                Ok(encrypted_data)
            }
            Err(e) => {
                let err_msg = format!("ClientSimulator encryption failed: {}", e);
                context.log_event(
                    "encrypt_data".to_string(),
                    json!({"blob_name": blob_name}),
                    json!({}),
                    ExecutionStatus::Failure,
                    Some(err_msg.clone()),
                );
                Err(anyhow!(err_msg))
            }
        }
    }

    /// Process encrypted data with TEE (test_concat)
    /// Supports processing multiple blobs in a single session
    pub async fn process_with_tee(
        context: &mut SessionContext,
        tee_instance_id: String,
        blob_names: Vec<String>,
        invocation_name: String,
    ) -> Result<Vec<u8>> {
        info!("Processing {} blob(s) with invocation {}", blob_names.len(), invocation_name);

        // Verify invocation exists
        let _invocation = context
            .invocations
            .get(&invocation_name)
            .ok_or_else(|| anyhow!("Invocation not found: {}", invocation_name))?;

        // Get protected response
        let protected_response_bytes = context
            .protected_responses
            .get(&invocation_name)
            .ok_or_else(|| anyhow!("Protected response not found for invocation: {}", invocation_name))?
            .clone();

        // Collect all blobs and their metadata
        use confidential_transform_proto::fcp::confidentialcompute::BlobMetadata;

        let mut blob_metadatas = Vec::new();
        let mut encrypted_data_chunks = Vec::new();

        for blob_name in &blob_names {
            let blob = context
                .encrypted_blobs
                .get(blob_name)
                .ok_or_else(|| anyhow!("Blob not found: {}", blob_name))?;

            // Decode BlobMetadata proto from stored bytes
            if !blob.blob_metadata_proto_bytes.is_empty() {
                match BlobMetadata::decode(blob.blob_metadata_proto_bytes.as_slice()) {
                    Ok(metadata) => {
                        blob_metadatas.push(Some(metadata));
                    }
                    Err(e) => {
                        warn!("Failed to decode BlobMetadata for blob {}: {}", blob_name, e);
                        blob_metadatas.push(None);
                    }
                }
            } else {
                info!("Warning: No BlobMetadata proto bytes available for blob {}", blob_name);
                blob_metadatas.push(None);
            }

            encrypted_data_chunks.push(blob.encrypted_data.clone());
        }

        // For now, if no TEE service is running, simulate the processing
        if context.tee_service.is_none() {
            info!("No TEE service running, simulating processing of {} blobs", blob_names.len());

            // Combine all encrypted data
            let mut combined_data = Vec::new();
            for chunk in &encrypted_data_chunks {
                combined_data.extend_from_slice(chunk);
            }

            // Simulate processing by reversing (placeholder)
            combined_data.reverse();

            context.log_event(
                "process_with_tee".to_string(),
                json!({"blob_count": blob_names.len(), "invocation_name": invocation_name}),
                json!({"status": "simulated", "results_size": combined_data.len()}),
                ExecutionStatus::Success,
                None,
            );

            return Ok(combined_data);
        }

        // Connect to running TEE service and process
        let tee_service = context
            .tee_service
            .as_ref()
            .filter(|s| s.instance_id == tee_instance_id)
            .ok_or_else(|| anyhow!("TEE instance not found: {}", tee_instance_id))?;

        let tee_address = format!("http://{}:{}", tee_service.address, tee_service.port);
        info!("Connecting to TEE at {}", tee_address);

        match TestConcatClient::new(&tee_address).await {
            Ok(mut test_concat_client) => {
                info!("Connected to test_concat TEE");

                // Deserialize protected response from bytes
                match crypto_proto::oak::crypto::v1::EncryptedRequest::decode(&protected_response_bytes[..]) {
                    Ok(encrypted_request) => {
                        // Step 1: Initialize test_concat with protected response
                        info!("Initializing test_concat with protected response");
                        match test_concat_client.initialize_with_protected_response(encrypted_request).await {
                            Ok(_init_response) => {
                                info!("✓ test_concat initialized successfully");

                                // Step 2: Process encrypted blobs
                                info!("Processing {} encrypted blob(s) with test_concat", encrypted_data_chunks.len());
                                match test_concat_client.process_with_encrypted_data(
                                    blob_metadatas,
                                    encrypted_data_chunks.clone(),
                                ).await {
                                    Ok((encrypted_results, _release_token)) => {
                                        info!("✓ test_concat processing complete: {} bytes results", encrypted_results.len());

                                        context.log_event(
                                            "process_with_tee".to_string(),
                                            json!({"blob_count": blob_names.len(), "invocation_name": invocation_name, "tee_instance_id": tee_instance_id}),
                                            json!({"results_size": encrypted_results.len(), "blobs_processed": blob_names.len()}),
                                            ExecutionStatus::Success,
                                            None,
                                        );

                                        Ok(encrypted_results)
                                    }
                                    Err(e) => {
                                        let err_msg = format!("test_concat processing failed: {}", e);
                                        context.log_event(
                                            "process_with_tee".to_string(),
                                            json!({"blob_count": blob_names.len()}),
                                            json!({}),
                                            ExecutionStatus::Failure,
                                            Some(err_msg.clone()),
                                        );
                                        Err(anyhow!(err_msg))
                                    }
                                }
                            }
                            Err(e) => {
                                let err_msg = format!("test_concat initialization failed: {}", e);
                                context.log_event(
                                    "process_with_tee".to_string(),
                                    json!({"blob_count": blob_names.len()}),
                                    json!({}),
                                    ExecutionStatus::Failure,
                                    Some(err_msg.clone()),
                                );
                                Err(anyhow!(err_msg))
                            }
                        }
                    }
                    Err(e) => {
                        Err(anyhow!("Failed to decode protected response: {}", e))
                    }
                }
            }
            Err(e) => {
                let err_msg = format!("Failed to connect to test_concat: {}", e);
                context.log_event(
                    "process_with_tee".to_string(),
                    json!({"blob_count": blob_names.len()}),
                    json!({}),
                    ExecutionStatus::Failure,
                    Some(err_msg.clone()),
                );
                Err(anyhow!(err_msg))
            }
        }
    }
}
