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

/// This module contains route definitions for the workshop server.
/// Routes are organized by resource type:
/// - /sessions: Session lifecycle management
/// - /sessions/{id}/kms/*: KMS operations
/// - /sessions/{id}/tee/*: TEE operations
/// - /sessions/{id}/*: Data encryption and processing

pub const ROUTES: &[(&str, &str, &str)] = &[
    // Session management
    ("POST", "/sessions", "create_session"),
    ("GET", "/sessions/:session_id", "get_session"),
    ("DELETE", "/sessions/:session_id", "delete_session"),
    // Policy operations
    ("POST", "/sessions/:session_id/policies", "create_policy"),
    ("POST", "/sessions/:session_id/variant-policies", "create_variant_policy"),
    ("POST", "/sessions/:session_id/data-access-policies", "create_data_access_policy"),
    // KMS operations
    ("POST", "/sessions/:session_id/kms/start", "start_kms"),
    ("POST", "/sessions/:session_id/kms/rotate-keyset", "rotate_keyset"),
    ("POST", "/sessions/:session_id/kms/derive-keys", "derive_keys"),
    (
        "POST",
        "/sessions/:session_id/kms/register-pipeline",
        "register_pipeline",
    ),
    (
        "POST",
        "/sessions/:session_id/kms/authorize-transform",
        "authorize_transform",
    ),
    // TEE operations
    ("POST", "/sessions/:session_id/tee/start", "start_tee"),
    ("GET", "/sessions/:session_id/tee/:instance_id/evidence", "get_tee_evidence"),
    ("GET", "/sessions/:session_id/tee/:instance_id/reference-values", "get_tee_reference_values"),
    ("POST", "/sessions/:session_id/tee/process", "process_with_tee"),
    // Data operations
    ("POST", "/sessions/:session_id/encrypt-data", "encrypt_data"),
];
