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

use chrono::{DateTime, Utc};
use kms_proto::{
    endorsement_proto::oak::attestation::v1::Endorsements,
    evidence_proto::oak::attestation::v1::Evidence,
};
use prost::Message;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Session identifier for all operations
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct SessionId(pub Uuid);

impl SessionId {
    pub fn new() -> Self {
        SessionId(Uuid::new_v4())
    }
}

impl Default for SessionId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::str::FromStr for SessionId {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(SessionId(Uuid::parse_str(s).map_err(|e| e.to_string())?))
    }
}

/// State of a managed process (KMS or TEE)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessState {
    pub pid: u32,
    pub start_time: DateTime<Utc>,
    pub timeout: u64,  // seconds
    pub service_type: ServiceType,
    pub status: ProcessStatus,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ServiceType {
    Kms,
    TestConcat,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ProcessStatus {
    Running,
    Stopped,
    Failed(String),
}

/// KMS Service information
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KmsService {
    pub address: String,
    pub port: u16,
    pub process_state: ProcessState,
}

/// Test Concat TEE Service information
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TestConcatService {
    pub instance_id: String,
    pub address: String,
    pub port: u16,
    pub process_state: ProcessState,
}

/// Session context - stores all resources for a session
#[derive(Clone, Debug)]
pub struct SessionContext {
    pub session_id: SessionId,
    pub created_at: DateTime<Utc>,
    pub timeout_seconds: u64,

    // Services
    pub kms_service: Option<KmsService>,
    pub tee_service: Option<TestConcatService>,

    // Cached execution state
    pub keyset_id: Option<u64>,
    pub policies: HashMap<String, PolicyCache>,
    pub variant_policies: HashMap<String, VariantPolicyCache>,
    pub data_access_policies: HashMap<String, DataAccessPolicyCache>,
    pub invocations: HashMap<String, InvocationCache>,
    pub encrypted_blobs: HashMap<String, EncryptedBlobCache>,
    pub protected_responses: HashMap<String, Vec<u8>>,
    pub public_keys: Vec<Vec<u8>>,

    // Execution history
    pub execution_log: Vec<ExecutionEvent>,
}

impl SessionContext {
    pub fn new(session_id: SessionId, timeout_seconds: u64) -> Self {
        SessionContext {
            session_id,
            created_at: Utc::now(),
            timeout_seconds,
            kms_service: None,
            tee_service: None,
            keyset_id: None,
            policies: HashMap::new(),
            variant_policies: HashMap::new(),
            data_access_policies: HashMap::new(),
            invocations: HashMap::new(),
            encrypted_blobs: HashMap::new(),
            protected_responses: HashMap::new(),
            public_keys: Vec::new(),
            execution_log: Vec::new(),
        }
    }

    pub fn is_expired(&self) -> bool {
        let elapsed = Utc::now()
            .signed_duration_since(self.created_at)
            .num_seconds() as u64;
        elapsed > self.timeout_seconds
    }

    pub fn log_event(
        &mut self,
        step: String,
        input: serde_json::Value,
        output: serde_json::Value,
        status: ExecutionStatus,
        error: Option<String>,
    ) {
        self.execution_log.push(ExecutionEvent {
            timestamp: Utc::now(),
            step,
            input,
            output,
            status,
            error,
        });
    }
}

impl Serialize for SessionContext {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("SessionContext", 10)?;
        state.serialize_field("session_id", &self.session_id)?;
        state.serialize_field("created_at", &self.created_at)?;
        state.serialize_field("timeout_seconds", &self.timeout_seconds)?;
        state.serialize_field("kms_service", &self.kms_service)?;
        state.serialize_field("tee_service", &self.tee_service)?;
        state.serialize_field("keyset_id", &self.keyset_id)?;
        state.serialize_field("policies", &self.policies)?;
        state.serialize_field("invocations", &self.invocations)?;
        state.serialize_field("encrypted_blobs", &self.encrypted_blobs)?;
        state.serialize_field("execution_log", &self.execution_log)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for SessionContext {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, Visitor};
        use std::fmt;

        #[allow(non_camel_case_types)]
        enum Field {
            session_id,
            created_at,
            timeout_seconds,
            kms_service,
            tee_service,
            keyset_id,
            policies,
            invocations,
            encrypted_blobs,
            execution_log,
        }

        impl<'de> Deserialize<'de> for Field {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct FieldVisitor;

                impl<'de> Visitor<'de> for FieldVisitor {
                    type Value = Field;

                    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                        formatter.write_str("field identifier")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
                    where
                        E: de::Error,
                    {
                        match value {
                            "session_id" => Ok(Field::session_id),
                            "created_at" => Ok(Field::created_at),
                            "timeout_seconds" => Ok(Field::timeout_seconds),
                            "kms_service" => Ok(Field::kms_service),
                            "tee_service" | "tee_services" => Ok(Field::tee_service),
                            "keyset_id" => Ok(Field::keyset_id),
                            "policies" => Ok(Field::policies),
                            "invocations" => Ok(Field::invocations),
                            "encrypted_blobs" => Ok(Field::encrypted_blobs),
                            "execution_log" => Ok(Field::execution_log),
                            _ => Err(de::Error::unknown_field(value, &[])),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct SessionContextVisitor;

        impl<'de> Visitor<'de> for SessionContextVisitor {
            type Value = SessionContext;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct SessionContext")
            }

            fn visit_map<V>(self, mut map: V) -> Result<SessionContext, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut session_id = None;
                let mut created_at = None;
                let mut timeout_seconds = None;
                let mut kms_service = None;
                let mut tee_service = None;
                let mut keyset_id = None;
                let mut policies = None;
                let mut invocations = None;
                let mut encrypted_blobs = None;
                let mut execution_log = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::session_id => {
                            if session_id.is_some() {
                                return Err(de::Error::duplicate_field("session_id"));
                            }
                            session_id = Some(map.next_value()?);
                        }
                        Field::created_at => {
                            if created_at.is_some() {
                                return Err(de::Error::duplicate_field("created_at"));
                            }
                            created_at = Some(map.next_value()?);
                        }
                        Field::timeout_seconds => {
                            if timeout_seconds.is_some() {
                                return Err(de::Error::duplicate_field("timeout_seconds"));
                            }
                            timeout_seconds = Some(map.next_value()?);
                        }
                        Field::kms_service => {
                            if kms_service.is_some() {
                                return Err(de::Error::duplicate_field("kms_service"));
                            }
                            kms_service = Some(map.next_value()?);
                        }
                        Field::tee_service => {
                            if tee_service.is_some() {
                                return Err(de::Error::duplicate_field("tee_service"));
                            }
                            tee_service = Some(map.next_value()?);
                        }
                        Field::keyset_id => {
                            if keyset_id.is_some() {
                                return Err(de::Error::duplicate_field("keyset_id"));
                            }
                            keyset_id = Some(map.next_value()?);
                        }
                        Field::policies => {
                            if policies.is_some() {
                                return Err(de::Error::duplicate_field("policies"));
                            }
                            policies = Some(map.next_value()?);
                        }
                        Field::invocations => {
                            if invocations.is_some() {
                                return Err(de::Error::duplicate_field("invocations"));
                            }
                            invocations = Some(map.next_value()?);
                        }
                        Field::encrypted_blobs => {
                            if encrypted_blobs.is_some() {
                                return Err(de::Error::duplicate_field("encrypted_blobs"));
                            }
                            encrypted_blobs = Some(map.next_value()?);
                        }
                        Field::execution_log => {
                            if execution_log.is_some() {
                                return Err(de::Error::duplicate_field("execution_log"));
                            }
                            execution_log = Some(map.next_value()?);
                        }
                    }
                }

                Ok(SessionContext {
                    session_id: session_id.ok_or_else(|| de::Error::missing_field("session_id"))?,
                    created_at: created_at.ok_or_else(|| de::Error::missing_field("created_at"))?,
                    timeout_seconds: timeout_seconds
                        .ok_or_else(|| de::Error::missing_field("timeout_seconds"))?,
                    kms_service: kms_service.unwrap_or(None),
                    tee_service: tee_service.unwrap_or(None),
                    keyset_id: keyset_id.unwrap_or(None),
                    policies: policies.unwrap_or_default(),
                    variant_policies: HashMap::new(),
                    data_access_policies: HashMap::new(),
                    invocations: invocations.unwrap_or_default(),
                    encrypted_blobs: encrypted_blobs.unwrap_or_default(),
                    protected_responses: HashMap::new(),
                    public_keys: Vec::new(),
                    execution_log: execution_log.unwrap_or_default(),
                })
            }
        }

        const FIELDS: &[&str] = &[
            "session_id",
            "created_at",
            "timeout_seconds",
            "kms_service",
            "tee_service",
            "keyset_id",
            "policies",
            "invocations",
            "encrypted_blobs",
            "execution_log",
        ];
        deserializer.deserialize_struct("SessionContext", FIELDS, SessionContextVisitor)
    }
}

/// Cached policy information
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolicyCache {
    pub name: String,
    pub json: serde_json::Value,
    pub proto_bytes: Vec<u8>,
    pub policy_hash: Vec<u8>,
}

/// Cached Pipeline Variant Policy
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VariantPolicyCache {
    pub name: String,
    pub proto_bytes: Vec<u8>,  // Serialized PipelineVariantPolicy
}

/// Cached Data Access Policy
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataAccessPolicyCache {
    pub name: String,
    pub proto_bytes: Vec<u8>,  // Serialized DataAccessPolicy
    pub policy_hash: Vec<u8>,  // SHA256 hash of proto_bytes
}

/// Cached invocation information
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InvocationCache {
    pub name: String,
    pub logical_pipeline_name: String,
    pub data_access_policy_name: String,
    pub invocation_id: Vec<u8>,
    pub keyset_id: u64,
}

/// Cached encrypted blob
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncryptedBlobCache {
    pub name: String,
    pub encrypted_data: Vec<u8>,
    pub blob_metadata: serde_json::Value,  // JSON for logging/inspection
    pub blob_metadata_proto_bytes: Vec<u8>,  // BlobMetadata proto bytes for TEE
    pub policy_name: String,
}

/// Execution event for audit trail
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExecutionEvent {
    pub timestamp: DateTime<Utc>,
    pub step: String,
    pub input: serde_json::Value,
    pub output: serde_json::Value,
    pub status: ExecutionStatus,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ExecutionStatus {
    Success,
    Failure,
}

// ============================================================================
// API Request/Response Models
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateSessionRequest {
    pub timeout_seconds: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SessionResponse {
    pub session_id: String,
    pub created_at: DateTime<Utc>,
    pub timeout_seconds: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StartKmsRequest {
    pub communication_channel: Option<String>,
    pub vm_type: Option<String>,
    pub memory_size: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct KmsStatusResponse {
    pub status: String,
    pub kms_address: Option<String>,
    pub port: Option<u16>,
    pub launcher_pid: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StartTeeRequest {
    pub communication_channel: Option<String>,
    pub vm_type: Option<String>,
    pub memory_size: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TeeStatusResponse {
    pub instances: Vec<TeeInstanceResponse>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TeeInstanceResponse {
    pub instance_id: String,
    pub status: String,
    pub address: Option<String>,
    pub port: Option<u16>,
    pub launcher_pid: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PolicyRequest {
    pub pipeline_name: String,
    pub src_node_id: Option<u32>,
    pub dst_node_id: Option<u32>,
    pub config_type_url: Option<String>,
    pub config_value: Option<String>,  // UTF-8 string or base64-encoded bytes
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PolicyResponse {
    pub policy_name: String,
    pub policy_hash: String,  // hex-encoded
    pub proto_bytes: String,  // hex-encoded
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ValidatePolicyRequest {
    pub json_policy: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ValidatePolicyResponse {
    pub valid: bool,
    pub errors: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BuildPolicyRequest {
    pub name: String,
    pub json_policy: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BuildPolicyResponse {
    pub policy_name: String,
    pub proto_bytes: String,  // hex-encoded
    pub policy_hash: String,  // hex-encoded
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RotateKeysetRequest {
    pub keyset_id: u64,
    pub ttl_seconds: i64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DeriveKeysRequest {
    pub policy_name: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DeriveKeysResponse {
    pub public_keys: Vec<String>,  // hex-encoded
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RegisterPipelineRequest {
    pub invocation_name: String,
    pub logical_pipeline_name: String,
    pub data_access_policy_name: String,
    pub keyset_id: u64,
    pub ttl_seconds: i64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RegisterPipelineResponse {
    pub invocation_id: String,  // hex-encoded
    pub invocation_name: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AuthorizeTransformRequest {
    pub invocation_name: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AuthorizeTransformResponse {
    pub protected_response: String,  // hex-encoded
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EncryptDataRequest {
    pub blob_name: String,
    pub plaintext: String,
    pub policy_name: String,
    pub public_key_index: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EncryptDataResponse {
    pub blob_name: String,
    pub encrypted_blob: String,  // hex-encoded
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CreatePolicyRequest {
    pub policy_name: String,
    pub pipeline_name: String,
    pub src_node_id: Option<u32>,
    pub dst_node_id: Option<u32>,
    pub config_type_url: Option<String>,
    pub config_value: Option<String>,  // hex-encoded bytes
}

/// Request to create a PipelineVariantPolicy
#[derive(Debug, Serialize, Deserialize)]
pub struct CreateVariantPolicyRequest {
    pub variant_policy_name: String,
    pub src_node_ids: Vec<u32>,
    pub dst_node_ids: Vec<u32>,
    pub config_type_url: Option<String>,
    pub config_value: Option<String>,  // hex-encoded bytes
}

/// Request to create a DataAccessPolicy from variant policies
#[derive(Debug, Serialize, Deserialize)]
pub struct CreateDataAccessPolicyRequest {
    pub policy_name: String,
    pub logical_pipeline_name: String,
    pub variant_policy_names: Vec<String>,  // List of variant policy names to include
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProcessWithTeeRequest {
    pub tee_instance_id: String,
    pub blob_names: Vec<String>,  // Support multiple blobs in single session
    pub invocation_name: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProcessWithTeeResponse {
    pub status: String,
    pub encrypted_results: Option<String>,  // hex-encoded
    pub error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExecutionLogResponse {
    pub events: Vec<ExecutionEvent>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TeeEvidenceResponse {
    pub evidence: String,  // JSON-serialized evidence
    pub endorsements: String,  // JSON-serialized endorsements
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TeeReferenceValuesResponse {
    pub reference_values: String,  // JSON-serialized reference values
}
