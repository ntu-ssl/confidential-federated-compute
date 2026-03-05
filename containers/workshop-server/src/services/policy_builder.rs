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

use access_policy_proto::{
    any_proto::google::protobuf::Any,
    fcp::confidentialcompute::{
        DataAccessPolicy, LogicalPipelinePolicy, PipelineVariantPolicy,
        pipeline_variant_policy::Transform,
        ApplicationMatcher, AccessBudget,
    },
};
use reference_value_proto::oak::attestation::v1::ReferenceValues;
use anyhow::{anyhow, Result};
use prost::Message;
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// Builder for creating DataAccessPolicy for KMS and TEE operations
pub struct PolicyBuilder;

impl PolicyBuilder {
    /// Create a DataAccessPolicy from structured request parameters with real reference values
    ///
    /// # Arguments
    /// * `pipeline_name` - Name of the logical pipeline (e.g., "sql_processing_pipeline")
    /// * `src_node_id` - Source node ID (defaults to 0 for client input)
    /// * `dst_node_id` - Destination node ID (defaults to 1 for results)
    /// * `reference_values` - Real TEE reference values from attestation (REQUIRED for KMS)
    /// * `config_type_url` - Type URL for config constraints (optional)
    /// * `config_value` - Config value (optional, as bytes)
    pub fn create_policy(
        pipeline_name: &str,
        src_node_id: Option<u32>,
        dst_node_id: Option<u32>,
        reference_values: ReferenceValues,
        config_type_url: Option<String>,
        config_value: Option<Vec<u8>>,
    ) -> Result<DataAccessPolicy> {
        // Validate pipeline name
        if pipeline_name.is_empty() {
            return Err(anyhow!("pipeline_name cannot be empty"));
        }

        let src_id = src_node_id.unwrap_or(0);
        let dst_id = dst_node_id.unwrap_or(1);

        // Create a Transform with the specified parameters
        let transform = Transform {
            src_node_ids: vec![src_id],
            dst_node_ids: vec![dst_id],
            application: Some(ApplicationMatcher {
                reference_values: Some(reference_values),
                ..Default::default()
            }),
            config_constraints: config_type_url.map(|type_url| Any {
                type_url,
                value: config_value.unwrap_or_default(),
                ..Default::default()
            }),
            access_budget: Some(AccessBudget::default()),
            ..Default::default()
        };

        // Create PipelineVariantPolicy containing the transform
        let pipeline_variant_policy = PipelineVariantPolicy {
            transforms: vec![transform],
            ..Default::default()
        };

        // Create LogicalPipelinePolicy with one variant
        let logical_pipeline_policy = LogicalPipelinePolicy {
            instances: vec![pipeline_variant_policy],
            ..Default::default()
        };

        // Create DataAccessPolicy with the named pipeline
        let mut pipelines = HashMap::new();
        pipelines.insert(pipeline_name.to_string(), logical_pipeline_policy);

        Ok(DataAccessPolicy {
            pipelines,
            ..Default::default()
        })
    }

    /// Serialize a DataAccessPolicy to protobuf bytes
    pub fn serialize_policy(policy: &DataAccessPolicy) -> Result<Vec<u8>> {
        Ok(policy.encode_to_vec())
    }

    /// Compute SHA-256 hash of policy bytes
    /// Used for policy authorization with KMS
    pub fn compute_hash(policy_bytes: &[u8]) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(policy_bytes);
        hasher.finalize().to_vec()
    }

    /// Create a policy hash from a DataAccessPolicy
    pub fn hash_policy(policy: &DataAccessPolicy) -> Result<Vec<u8>> {
        let policy_bytes = Self::serialize_policy(policy)?;
        Ok(Self::compute_hash(&policy_bytes))
    }
}
