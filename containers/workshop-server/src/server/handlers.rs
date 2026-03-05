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
use crate::services::{ExecutionEngine, LauncherManager, SessionRegistry};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use prost::Message;
use sha2::Digest;
use std::sync::Arc;
use tracing::{info, warn};

type SharedState = (Arc<SessionRegistry>, Arc<LauncherManager>);

/// Create a new session
/// Automatically assigns a KMS instance from the pool
pub async fn create_session(
    State((registry, launcher)): State<SharedState>,
    Json(req): Json<CreateSessionRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let timeout_seconds = req.timeout_seconds.unwrap_or(3600);
    match registry.create_session(timeout_seconds).await {
        Ok(session_id) => {
            // Assign a KMS instance from the pool to this session
            match launcher.assign_kms_to_session(session_id).await {
                Ok((_kms_index, kms_service)) => {
                    // Update session context with assigned KMS
                    match registry.get_session(session_id).await {
                        Ok(mut context) => {
                            context.kms_service = Some(kms_service.clone());
                            if let Err(e) = registry.update_session(session_id, context).await {
                                tracing::error!("Failed to update session with KMS assignment: {}", e);
                            }

                            let now = chrono::Utc::now();
                            (
                                StatusCode::CREATED,
                                Json(serde_json::json!({
                                    "session_id": session_id.to_string(),
                                    "created_at": now,
                                    "timeout_seconds": timeout_seconds,
                                    "kms_service": {
                                        "address": kms_service.address,
                                        "port": kms_service.port,
                                        "auto_assigned": true,
                                    },
                                })),
                            )
                        }
                        Err(e) => {
                            tracing::error!("Failed to get session after KMS assignment: {}", e);
                            (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(serde_json::json!({
                                    "error": format!("Session created but failed to assign KMS: {}", e),
                                })),
                            )
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to assign KMS to session: {}", e);
                    // Delete the session since KMS assignment failed
                    let _ = registry.delete_session(session_id).await;
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({
                            "error": format!("Failed to assign KMS from pool: {}", e),
                        })),
                    )
                }
            }
        }
        Err(e) => {
            let error_msg = e.to_string();
            let is_limit_error = error_msg.contains("maximum") && error_msg.contains("concurrent sessions");

            if is_limit_error {
                tracing::warn!("Session creation rejected: {}", e);
                (
                    StatusCode::CONFLICT,
                    Json(serde_json::json!({
                        "error": error_msg,
                        "max_sessions": SessionRegistry::max_sessions(),
                    })),
                )
            } else {
                tracing::error!("Failed to create session: {}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": error_msg,
                    })),
                )
            }
        }
    }
}

/// Get session information
pub async fn get_session(
    State((registry, _launcher)): State<SharedState>,
    Path(session_id): Path<String>,
) -> (StatusCode, Json<serde_json::Value>) {
    let session_id = match session_id.parse::<SessionId>() {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "Invalid session ID"})),
            );
        }
    };

    match registry.get_session(session_id).await {
        Ok(context) => (StatusCode::OK, Json(serde_json::to_value(&context).unwrap())),
        Err(_) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Session not found"})),
        ),
    }
}

/// Start KMS service for a session
/// DEPRECATED: Start KMS service for a session
/// KMS is now automatically assigned from a pool when the session is created.
/// This endpoint returns the already-assigned KMS information.
pub async fn start_kms(
    State((registry, _launcher)): State<SharedState>,
    Path(session_id): Path<String>,
    Json(_req): Json<StartKmsRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let session_id = match session_id.parse::<SessionId>() {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "Invalid session ID"})),
            );
        }
    };

    match registry.get_session(session_id).await {
        Ok(context) => {
            // KMS is already assigned automatically when session was created
            if let Some(kms_service) = context.kms_service {
                (
                    StatusCode::OK,
                    Json(serde_json::json!({
                        "status": "already_assigned",
                        "address": kms_service.address,
                        "port": kms_service.port,
                        "message": "KMS is automatically assigned from pool on session creation. This endpoint is deprecated."
                    })),
                )
            } else {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": "KMS not assigned to session. This should not happen.",
                    })),
                )
            }
        }
        Err(_) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Session not found"})),
        ),
    }
}

/// Start TEE service for a session
pub async fn start_tee(
    State((registry, launcher)): State<SharedState>,
    Path(session_id): Path<String>,
    Json(req): Json<StartTeeRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let session_id = match session_id.parse::<SessionId>() {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "Invalid session ID"})),
            );
        }
    };

    match registry.get_session(session_id).await {
        Ok(mut context) => {
            // If there's an existing TEE, clean it up first
            if context.tee_service.is_some() {
                let mut launchers = registry.launchers().lock().await;
                if let Some(launcher_handles) = launchers.get_mut(&session_id) {
                    if let Some(mut old_launcher) = launcher_handles.tee_launcher.take() {
                        info!("Killing existing TEE launcher for session {}", session_id);
                        old_launcher.kill().await;
                    }
                }
                drop(launchers);
            }

            // Get KMS address if available
            let kms_address = context
                .kms_service
                .as_ref()
                .map(|kms| format!("{}:{}", kms.address, kms.port))
                .unwrap_or_else(|| "localhost:8080".to_string());

            match launcher.launch_test_concat(req.memory_size).await {
                Ok((tee_launcher, tee_service)) => {
                    // Replace any existing TEE service (one TEE per session)
                    context.tee_service = Some(tee_service.clone());

                    if let Err(e) = registry.update_session(session_id, context).await {
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": e.to_string()})),
                        );
                    }

                    // Store launcher for cleanup on session deletion
                    let mut launchers = registry.launchers().lock().await;
                    if let Some(launcher_handles) = launchers.get_mut(&session_id) {
                        launcher_handles.tee_launcher = Some(tee_launcher);
                    } else {
                        // LauncherHandles don't exist (KMS pool session), create them for TEE
                        use crate::services::session_registry::LauncherHandles;
                        let handles = LauncherHandles {
                            kms_launcher: None,  // KMS is in the pool
                            tee_launcher: Some(tee_launcher),
                        };
                        launchers.insert(session_id, handles);
                        info!("Created LauncherHandles for session {} to store TEE launcher", session_id);
                    }

                    (
                        StatusCode::OK,
                        Json(serde_json::json!({
                            "status": "started",
                            "instances": vec![serde_json::json!({
                                "instance_id": tee_service.instance_id,
                                "address": tee_service.address,
                                "port": tee_service.port
                            })]
                        })),
                    )
                }
                Err(e) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": format!("Failed to start TEE: {}", e)})),
                ),
            }
        }
        Err(_) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Session not found"})),
        ),
    }
}

/// Rotate KMS keyset
pub async fn rotate_keyset(
    State((registry, _launcher)): State<SharedState>,
    Path(session_id): Path<String>,
    Json(req): Json<RotateKeysetRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let session_id = match session_id.parse::<SessionId>() {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "Invalid session ID"})),
            );
        }
    };

    match registry.get_session(session_id).await {
        Ok(mut context) => {
            if let Err(e) =
                ExecutionEngine::rotate_keyset(&mut context, req.keyset_id, req.ttl_seconds).await
            {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": e.to_string()})),
                );
            }

            if let Err(e) = registry.update_session(session_id, context).await {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": e.to_string()})),
                );
            }

            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "keyset_id": req.keyset_id,
                    "rotated": true
                })),
            )
        }
        Err(_) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Session not found"})),
        ),
    }
}

/// Derive encryption keys from KMS
pub async fn derive_keys(
    State((registry, _launcher)): State<SharedState>,
    Path(session_id): Path<String>,
    Json(req): Json<DeriveKeysRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let session_id = match session_id.parse::<SessionId>() {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "Invalid session ID"})),
            );
        }
    };

    match registry.get_session(session_id).await {
        Ok(mut context) => {
            match ExecutionEngine::derive_keys(&mut context, req.policy_name).await {
                Ok(public_keys) => {
                    if let Err(e) = registry.update_session(session_id, context).await {
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": e.to_string()})),
                        );
                    }

                    let hex_keys: Vec<String> =
                        public_keys.iter().map(|k| hex::encode(k)).collect();
                    (
                        StatusCode::OK,
                        Json(serde_json::json!({
                            "public_keys": hex_keys
                        })),
                    )
                }
                Err(e) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": e.to_string()})),
                ),
            }
        }
        Err(_) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Session not found"})),
        ),
    }
}

/// Register a pipeline invocation with KMS
pub async fn register_pipeline(
    State((registry, _launcher)): State<SharedState>,
    Path(session_id): Path<String>,
    Json(req): Json<RegisterPipelineRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let session_id = match session_id.parse::<SessionId>() {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "Invalid session ID"})),
            );
        }
    };

    match registry.get_session(session_id).await {
        Ok(mut context) => {
            match ExecutionEngine::register_pipeline(
                &mut context,
                req.invocation_name.clone(),
                req.logical_pipeline_name,
                req.data_access_policy_name,
                req.keyset_id,
                req.ttl_seconds,
            )
            .await
            {
                Ok(invocation_id) => {
                    if let Err(e) = registry.update_session(session_id, context).await {
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": e.to_string()})),
                        );
                    }

                    (
                        StatusCode::OK,
                        Json(serde_json::json!({
                            "invocation_id": hex::encode(&invocation_id),
                            "invocation_name": req.invocation_name
                        })),
                    )
                }
                Err(e) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": e.to_string()})),
                ),
            }
        }
        Err(_) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Session not found"})),
        ),
    }
}

/// Authorize a transform with KMS
pub async fn authorize_transform(
    State((registry, _launcher)): State<SharedState>,
    Path(session_id): Path<String>,
    Json(req): Json<AuthorizeTransformRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let session_id = match session_id.parse::<SessionId>() {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "Invalid session ID"})),
            );
        }
    };

    match registry.get_session(session_id).await {
        Ok(mut context) => {
            match ExecutionEngine::authorize_transform(&mut context, req.invocation_name.clone(), registry.clone(), session_id)
                .await
            {
                Ok(protected_response) => {
                    // Store protected response for later use in process_with_tee
                    context.protected_responses.insert(req.invocation_name.clone(), protected_response.clone());

                    if let Err(e) = registry.update_session(session_id, context).await {
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": e.to_string()})),
                        );
                    }

                    (
                        StatusCode::OK,
                        Json(serde_json::json!({
                            "protected_response": hex::encode(&protected_response)
                        })),
                    )
                }
                Err(e) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": e.to_string()})),
                ),
            }
        }
        Err(_) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Session not found"})),
        ),
    }
}

/// Encrypt data for a session
pub async fn encrypt_data(
    State((registry, _launcher)): State<SharedState>,
    Path(session_id): Path<String>,
    Json(req): Json<EncryptDataRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let session_id = match session_id.parse::<SessionId>() {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "Invalid session ID"})),
            );
        }
    };

    match registry.get_session(session_id).await {
        Ok(mut context) => {
            // Decode plaintext from hex
            let plaintext = match hex::decode(&req.plaintext) {
                Ok(data) => data,
                Err(e) => {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(serde_json::json!({"error": format!("Invalid plaintext hex: {}", e)})),
                    );
                }
            };

            match ExecutionEngine::encrypt_data(
                &mut context,
                req.blob_name.clone(),
                plaintext,
                req.policy_name,
                req.public_key_index,
            )
            .await
            {
                Ok(encrypted_blob) => {
                    if let Err(e) = registry.update_session(session_id, context).await {
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": e.to_string()})),
                        );
                    }

                    (
                        StatusCode::OK,
                        Json(serde_json::json!({
                            "blob_name": req.blob_name,
                            "encrypted_blob": hex::encode(&encrypted_blob)
                        })),
                    )
                }
                Err(e) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": e.to_string()})),
                ),
            }
        }
        Err(_) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Session not found"})),
        ),
    }
}

/// Process encrypted data with TEE
pub async fn process_with_tee(
    State((registry, _launcher)): State<SharedState>,
    Path(session_id): Path<String>,
    Json(req): Json<ProcessWithTeeRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let session_id = match session_id.parse::<SessionId>() {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "Invalid session ID"})),
            );
        }
    };

    match registry.get_session(session_id).await {
        Ok(mut context) => {
            match ExecutionEngine::process_with_tee(
                &mut context,
                req.tee_instance_id,
                req.blob_names,
                req.invocation_name,
            )
            .await
            {
                Ok(results) => {
                    if let Err(e) = registry.update_session(session_id, context).await {
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": e.to_string()})),
                        );
                    }

                    (
                        StatusCode::OK,
                        Json(serde_json::json!({
                            "status": "processed",
                            "encrypted_results": hex::encode(&results)
                        })),
                    )
                }
                Err(e) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": e.to_string()})),
                ),
            }
        }
        Err(_) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Session not found"})),
        ),
    }
}

/// Create a PipelineVariantPolicy
pub async fn create_variant_policy(
    State((registry, _launcher)): State<SharedState>,
    Path(session_id): Path<String>,
    Json(req): Json<CreateVariantPolicyRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    use oak_attestation_verification::extract_evidence;
    use crate::services::attestation_factory::create_reference_values_for_extracted_evidence;
    use crate::services::kms_client::ProstProtoConversionExt;
    use reference_value_proto::oak::attestation::v1::ReferenceValues;
    use access_policy_proto::fcp::confidentialcompute::{ApplicationMatcher, PipelineVariantPolicy};
    use access_policy_proto::fcp::confidentialcompute::pipeline_variant_policy::Transform;

    let session_id = match session_id.parse::<SessionId>() {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "Invalid session ID"})),
            );
        }
    };

    match registry.get_session(session_id).await {
        Ok(mut context) => {
            // Fetch evidence from the TEE launcher
            let endorsed_evidence_opt = registry.get_tee_endorsed_evidence(session_id).await;

            if endorsed_evidence_opt.is_none() {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({
                        "error": "TEE evidence not available. Start TEE service first (POST /sessions/:id/tee/start)"
                    })),
                );
            }

            // Extract and convert real reference values from TEE evidence
            let reference_values = match endorsed_evidence_opt {
                Some(endorsed_evidence) => {
                    if let Some(evidence) = endorsed_evidence.evidence {
                        match extract_evidence(&evidence) {
                            Ok(extracted_evidence) => {
                                let rv_oak = create_reference_values_for_extracted_evidence(extracted_evidence);
                                match rv_oak.convert() {
                                    Ok(rv) => {
                                        info!("✓ Created reference values from TEE evidence");
                                        Some(rv)
                                    }
                                    Err(e) => {
                                        warn!("Failed to convert reference values: {}", e);
                                        None
                                    }
                                }
                            }
                            Err(e) => {
                                warn!("Failed to extract evidence: {}", e);
                                None
                            }
                        }
                    } else {
                        warn!("Evidence not in endorsed evidence");
                        None
                    }
                }
                None => None,
            };

            // Decode config_value if provided
            let config_bytes = req.config_value.as_ref().and_then(|hex_str| {
                hex::decode(hex_str).ok()
            });

            // Create the transform with reference values
            let transform = Transform {
                src_node_ids: req.src_node_ids.clone(),
                dst_node_ids: req.dst_node_ids.clone(),
                application: reference_values.map(|rv| ApplicationMatcher {
                    reference_values: Some(rv),
                    ..Default::default()
                }),
                // Config constraints would be set if config_bytes are provided, but leaving it None for now
                config_constraints: None,
                ..Default::default()
            };

            // Create the PipelineVariantPolicy with the transform
            let variant_policy = PipelineVariantPolicy {
                transforms: vec![transform],
                ..Default::default()
            };

            // Serialize to bytes
            let variant_policy_bytes = variant_policy.encode_to_vec();

            // Store in session context
            let variant_cache = VariantPolicyCache {
                name: req.variant_policy_name.clone(),
                proto_bytes: variant_policy_bytes.clone(),
            };

            context.variant_policies.insert(req.variant_policy_name.clone(), variant_cache);

            if let Err(e) = registry.update_session(session_id, context).await {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": e.to_string()})),
                );
            }

            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "status": "created",
                    "variant_policy_name": req.variant_policy_name,
                    "variant_policy_bytes_size": variant_policy_bytes.len(),
                    "src_node_ids": req.src_node_ids,
                    "dst_node_ids": req.dst_node_ids
                })),
            )
        }
        Err(_) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Session not found"})),
        ),
    }
}

/// Create a DataAccessPolicy from variant policies
pub async fn create_data_access_policy(
    State((registry, _launcher)): State<SharedState>,
    Path(session_id): Path<String>,
    Json(req): Json<CreateDataAccessPolicyRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    use access_policy_proto::fcp::confidentialcompute::{DataAccessPolicy, LogicalPipelinePolicy};
    use std::collections::HashMap;

    let session_id = match session_id.parse::<SessionId>() {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "Invalid session ID"})),
            );
        }
    };

    match registry.get_session(session_id).await {
        Ok(mut context) => {
            // Collect the variant policies
            let mut variant_policies = Vec::new();
            for variant_name in &req.variant_policy_names {
                match context.variant_policies.get(variant_name) {
                    Some(variant_cache) => {
                        // Decode bytes back to PipelineVariantPolicy
                        use prost::Message;
                        use access_policy_proto::fcp::confidentialcompute::PipelineVariantPolicy;

                        match PipelineVariantPolicy::decode(variant_cache.proto_bytes.as_slice()) {
                            Ok(policy) => variant_policies.push(policy),
                            Err(e) => {
                                return (
                                    StatusCode::INTERNAL_SERVER_ERROR,
                                    Json(serde_json::json!({
                                        "error": format!("Failed to decode variant policy {}: {}", variant_name, e)
                                    })),
                                );
                            }
                        }
                    }
                    None => {
                        return (
                            StatusCode::NOT_FOUND,
                            Json(serde_json::json!({
                                "error": format!("Variant policy not found: {}", variant_name)
                            })),
                        );
                    }
                }
            }

            // Create LogicalPipelinePolicy with the variants
            let logical_policy = LogicalPipelinePolicy {
                instances: variant_policies,
            };

            // Create DataAccessPolicy with the logical pipeline
            let mut pipelines = HashMap::new();
            pipelines.insert(req.logical_pipeline_name.clone(), logical_policy);

            let data_access_policy = DataAccessPolicy {
                pipelines,
                ..Default::default()
            };

            // Serialize to bytes
            let policy_bytes = data_access_policy.encode_to_vec();

            // Compute policy hash
            let mut hasher = sha2::Sha256::new();
            hasher.update(&policy_bytes);
            let policy_hash = hasher.finalize().to_vec();

            // Store in session context
            let access_policy_cache = DataAccessPolicyCache {
                name: req.policy_name.clone(),
                proto_bytes: policy_bytes.clone(),
                policy_hash: policy_hash.clone(),
            };

            context.data_access_policies.insert(req.policy_name.clone(), access_policy_cache);

            if let Err(e) = registry.update_session(session_id, context).await {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": e.to_string()})),
                );
            }

            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "status": "created",
                    "policy_name": req.policy_name,
                    "logical_pipeline_name": req.logical_pipeline_name,
                    "variant_policy_count": req.variant_policy_names.len(),
                    "policy_hash": hex::encode(&policy_hash),
                    "policy_bytes_size": policy_bytes.len()
                })),
            )
        }
        Err(_) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Session not found"})),
        ),
    }
}

/// Get TEE evidence and endorsements for a session
pub async fn get_tee_evidence(
    State((registry, _launcher)): State<SharedState>,
    Path((session_id, instance_id)): Path<(String, String)>,
) -> (StatusCode, Json<serde_json::Value>) {
    let session_id = match session_id.parse::<SessionId>() {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "Invalid session ID"})),
            );
        }
    };

    match registry.get_session(session_id).await {
        Ok(context) => {
            // Check if the TEE instance exists and matches
            let tee_service = context
                .tee_service
                .as_ref()
                .filter(|s| s.instance_id == instance_id);

            match tee_service {
                Some(tee) => {
                    // Fetch evidence and endorsements from the TEE launcher
                    match registry.get_tee_endorsed_evidence(session_id).await {
                        Some(endorsed_evidence) => {
                            let evidence_bytes = endorsed_evidence.evidence
                                .as_ref()
                                .map(|e| e.encode_to_vec())
                                .unwrap_or_default();
                            let endorsements_bytes = endorsed_evidence.endorsements
                                .as_ref()
                                .map(|e| e.encode_to_vec())
                                .unwrap_or_default();

                            info!("✓ Returning TEE evidence and endorsements for instance {}", instance_id);
                            (
                                StatusCode::OK,
                                Json(serde_json::json!({
                                    "instance_id": instance_id,
                                    "address": tee.address,
                                    "port": tee.port,
                                    "evidence_hex": hex::encode(&evidence_bytes),
                                    "evidence_size_bytes": evidence_bytes.len(),
                                    "evidence_structure": format!("{:#?}", endorsed_evidence.evidence.as_ref()),
                                    "endorsements_hex": hex::encode(&endorsements_bytes),
                                    "endorsements_size_bytes": endorsements_bytes.len(),
                                    "endorsements_structure": format!("{:#?}", endorsed_evidence.endorsements.as_ref()),
                                    "status": "available"
                                })),
                            )
                        }
                        None => {
                            warn!("TEE launcher not available for instance {}", instance_id);
                            (
                                StatusCode::NOT_FOUND,
                                Json(serde_json::json!({
                                    "error": "TEE evidence and endorsements not yet available",
                                    "hint": "Evidence is populated when TEE service starts via POST /sessions/:id/tee/start"
                                })),
                            )
                        }
                    }
                }
                None => (
                    StatusCode::NOT_FOUND,
                    Json(serde_json::json!({"error": "TEE instance not found"})),
                ),
            }
        }
        Err(_) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Session not found"})),
        ),
    }
}

/// Get reference values for a TEE instance
pub async fn get_tee_reference_values(
    State((registry, _launcher)): State<SharedState>,
    Path((session_id, instance_id)): Path<(String, String)>,
) -> (StatusCode, Json<serde_json::Value>) {
    use oak_attestation_verification::extract_evidence;
    use crate::services::attestation_factory::create_reference_values_for_extracted_evidence;
    use crate::services::kms_client::ProstProtoConversionExt;
    use reference_value_proto::oak::attestation::v1::ReferenceValues;

    let session_id = match session_id.parse::<SessionId>() {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "Invalid session ID"})),
            );
        }
    };

    match registry.get_session(session_id).await {
        Ok(context) => {
            // Check if the TEE instance exists and matches
            let tee_service = context
                .tee_service
                .as_ref()
                .filter(|s| s.instance_id == instance_id);

            match tee_service {
                Some(tee) => {
                    // Fetch evidence from the TEE launcher
                    match registry.get_tee_endorsed_evidence(session_id).await {
                        Some(endorsed_evidence) => {
                            // Extract reference values from fetched evidence
                            if let Some(evidence) = endorsed_evidence.evidence {
                                // Convert from oak_proto_rust Evidence to reference values
                                match extract_evidence(&evidence) {
                                    Ok(extracted_evidence) => {
                                        // Create reference values from extracted evidence
                                        let rv_oak = create_reference_values_for_extracted_evidence(extracted_evidence);
                                        // Convert to reference_value_proto
                                        let reference_values: ReferenceValues = match rv_oak.convert() {
                                            Ok(rv) => rv,
                                            Err(e) => {
                                                warn!("Failed to convert reference values: {}", e);
                                                return (
                                                    StatusCode::INTERNAL_SERVER_ERROR,
                                                    Json(serde_json::json!({
                                                        "error": "Failed to convert reference values"
                                                    })),
                                                );
                                            }
                                        };

                                        // Serialize reference values as proto bytes
                                        let rv_bytes = reference_values.encode_to_vec();

                                        info!("✓ Extracted reference values for instance {}", instance_id);
                                        (
                                            StatusCode::OK,
                                            Json(serde_json::json!({
                                                "instance_id": instance_id,
                                                "address": tee.address,
                                                "port": tee.port,
                                                "reference_values_hex": hex::encode(&rv_bytes),
                                                "reference_values_size_bytes": rv_bytes.len(),
                                                "reference_values_structure": format!("{:#?}", reference_values),
                                                "status": "available"
                                            })),
                                        )
                                    }
                                    Err(e) => {
                                        warn!("Failed to extract evidence: {}", e);
                                        (
                                            StatusCode::INTERNAL_SERVER_ERROR,
                                            Json(serde_json::json!({
                                                "error": "Failed to extract evidence from TEE"
                                            })),
                                        )
                                    }
                                }
                            } else {
                                warn!("Evidence not in endorsed evidence for instance {}", instance_id);
                                (
                                    StatusCode::NOT_FOUND,
                                    Json(serde_json::json!({
                                        "error": "TEE evidence not available",
                                        "hint": "Evidence is populated when TEE service starts via POST /sessions/:id/tee/start"
                                    })),
                                )
                            }
                        }
                        None => {
                            warn!("TEE launcher not available for instance {}", instance_id);
                            (
                                StatusCode::NOT_FOUND,
                                Json(serde_json::json!({
                                    "error": "TEE evidence not available",
                                    "hint": "Evidence is populated when TEE service starts via POST /sessions/:id/tee/start"
                                })),
                            )
                        }
                    }
                }
                None => (
                    StatusCode::NOT_FOUND,
                    Json(serde_json::json!({"error": "TEE instance not found"})),
                ),
            }
        }
        Err(_) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Session not found"})),
        ),
    }
}

/// Delete a session
pub async fn delete_session(
    State((registry, launcher)): State<SharedState>,
    Path(session_id): Path<String>,
) -> StatusCode {
    let session_id = match session_id.parse::<SessionId>() {
        Ok(id) => id,
        Err(_) => return StatusCode::BAD_REQUEST,
    };

    // Unassign KMS from pool before deleting session
    if let Err(e) = launcher.unassign_kms_from_session(session_id).await {
        warn!("Failed to unassign KMS from session {}: {}", session_id, e);
        // Continue with session deletion even if unassignment fails
    }

    match registry.delete_session(session_id).await {
        Ok(_) => {
            info!("Deleted session: {}", session_id);
            StatusCode::OK
        }
        Err(_) => StatusCode::NOT_FOUND,
    }
}

/// Create a policy for a session using real TEE evidence
pub async fn create_policy(
    State((registry, _launcher)): State<SharedState>,
    Path(session_id): Path<String>,
    Json(req): Json<CreatePolicyRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    use oak_attestation_verification::extract_evidence;
    use crate::services::attestation_factory::create_reference_values_for_extracted_evidence;
    use crate::services::kms_client::ProstProtoConversionExt;
    use reference_value_proto::oak::attestation::v1::ReferenceValues;

    let session_id = match session_id.parse::<SessionId>() {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "Invalid session ID"})),
            );
        }
    };

    match registry.get_session(session_id).await {
        Ok(mut context) => {
            // Fetch evidence from the TEE launcher
            let endorsed_evidence_opt = registry.get_tee_endorsed_evidence(session_id).await;

            if endorsed_evidence_opt.is_none() {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({
                        "error": "TEE evidence not available. Start TEE service first (POST /sessions/:id/tee/start)"
                    })),
                );
            }

            // Extract and convert real reference values from TEE evidence
            let reference_values = match endorsed_evidence_opt {
                Some(endorsed_evidence) => {
                    if let Some(evidence) = endorsed_evidence.evidence {
                        match extract_evidence(&evidence) {
                            Ok(extracted_evidence) => {
                                // Create reference values from extracted evidence (returns oak_proto_rust type)
                                let rv_oak = create_reference_values_for_extracted_evidence(extracted_evidence);
                                // Convert from oak_proto_rust to reference_value_proto
                                match rv_oak.convert() {
                                    Ok(rv) => {
                                        info!("✓ Created reference values from TEE evidence");
                                        rv
                                    }
                                    Err(e) => {
                                        warn!("Failed to convert reference values: {}", e);
                                        info!("Using default reference values instead");
                                        ReferenceValues::default()
                                    }
                                }
                            }
                            Err(e) => {
                                warn!("Failed to extract evidence: {}", e);
                                info!("Using default reference values instead");
                                ReferenceValues::default()
                            }
                        }
                    } else {
                        warn!("Evidence not in endorsed evidence");
                        info!("Using default reference values instead");
                        ReferenceValues::default()
                    }
                }
                None => ReferenceValues::default(),
            };

            // Decode config_value if provided
            let config_bytes = req.config_value.as_ref().and_then(|hex_str| {
                hex::decode(hex_str).ok()
            });

            // Create policy using PolicyBuilder with real reference values
            match crate::services::PolicyBuilder::create_policy(
                &req.pipeline_name,
                req.src_node_id,
                req.dst_node_id,
                reference_values,
                req.config_type_url,
                config_bytes,
            ) {
                Ok(policy) => {
                    // Serialize policy to bytes for storage
                    let policy_bytes = policy.encode_to_vec();

                    // Compute policy hash
                    let mut hasher = sha2::Sha256::new();
                    hasher.update(&policy_bytes);
                    let policy_hash = hasher.finalize().to_vec();

                    // Store policy in session context
                    let policy_cache = PolicyCache {
                        name: req.policy_name.clone(),
                        json: serde_json::json!({
                            "pipeline_name": req.pipeline_name,
                            "src_node_id": req.src_node_id.unwrap_or(0),
                            "dst_node_id": req.dst_node_id.unwrap_or(1)
                        }),
                        proto_bytes: policy_bytes.clone(),
                        policy_hash: policy_hash.clone(),
                    };

                    context.policies.insert(req.policy_name.clone(), policy_cache);

                    if let Err(e) = registry.update_session(session_id, context).await {
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": e.to_string()})),
                        );
                    }

                    (
                        StatusCode::OK,
                        Json(serde_json::json!({
                            "status": "created",
                            "policy_name": req.policy_name,
                            "pipeline_name": req.pipeline_name,
                            "policy_hash": hex::encode(&policy_hash),
                            "policy_bytes_size": policy_bytes.len()
                        })),
                    )
                }
                Err(e) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": format!("Failed to create policy: {}", e)})),
                ),
            }
        }
        Err(_) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Session not found"})),
        ),
    }
}

/// Get workshop server status and session metrics
pub async fn get_status(
    State((registry, _launcher)): State<SharedState>,
) -> (StatusCode, Json<serde_json::Value>) {
    let session_count = registry.session_count().await;
    let max_sessions = SessionRegistry::max_sessions();
    let is_at_limit = registry.is_at_limit().await;

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "healthy",
            "sessions": {
                "active": session_count,
                "max": max_sessions,
                "at_limit": is_at_limit,
                "available": max_sessions - session_count,
            },
        })),
    )
}
