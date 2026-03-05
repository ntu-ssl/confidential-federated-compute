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

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use prost::Message;
use sha2::{Digest, Sha256};
use std::collections::HashMap;

use access_policy_proto::{
    any_proto::google::protobuf::Any,
    fcp::confidentialcompute::{
        pipeline_variant_policy::Transform, ApplicationMatcher, DataAccessPolicy,
        LogicalPipelinePolicy, PipelineVariantPolicy,
    },
};
use program_executor_tee_proto::fcp::confidentialcompute::ProgramExecutorTeeConfigConstraints;
use reference_value_proto::oak::attestation::v1::{
    binary_reference_value, kernel_binary_reference_value, reference_values, text_reference_value,
    BinaryReferenceValue, ContainerLayerReferenceValues, InsecureReferenceValues,
    KernelBinaryReferenceValue, KernelLayerReferenceValues, OakContainersReferenceValues,
    ReferenceValues, RootLayerReferenceValues, SkipVerification, SystemLayerReferenceValues,
    TextReferenceValue,
};

/// The hardcoded access policy hash used by FakeDataReadWriteService
/// Must match kAccessPolicyHash in fake_data_read_write_service.h
pub const ACCESS_POLICY_HASH: &str = "access_policy_hash";

/// Maximum number of runs allowed for the program executor
pub const MAX_NUM_RUNS: i32 = 5;

/// Computes the SHA-256 hash of the access policy
pub fn compute_policy_hash(policy: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(policy);
    hasher.finalize().to_vec()
}

/// Creates test reference values with skip verification for all layers
/// This is used for testing purposes where attestation verification is not required
pub fn get_test_reference_values() -> ReferenceValues {
    let skip = SkipVerification {};

    ReferenceValues {
        r#type: Some(reference_values::Type::OakContainers(OakContainersReferenceValues {
            root_layer: Some(RootLayerReferenceValues {
                insecure: Some(InsecureReferenceValues {}),
                ..Default::default()
            }),
            kernel_layer: Some(KernelLayerReferenceValues {
                kernel: Some(KernelBinaryReferenceValue {
                    r#type: Some(kernel_binary_reference_value::Type::Skip(skip.clone())),
                }),
                kernel_cmd_line_text: Some(TextReferenceValue {
                    r#type: Some(text_reference_value::Type::Skip(skip.clone())),
                }),
                init_ram_fs: Some(BinaryReferenceValue {
                    r#type: Some(binary_reference_value::Type::Skip(skip.clone())),
                }),
                memory_map: Some(BinaryReferenceValue {
                    r#type: Some(binary_reference_value::Type::Skip(skip.clone())),
                }),
                acpi: Some(BinaryReferenceValue {
                    r#type: Some(binary_reference_value::Type::Skip(skip.clone())),
                }),
                ..Default::default()
            }),
            system_layer: Some(SystemLayerReferenceValues {
                system_image: Some(BinaryReferenceValue {
                    r#type: Some(binary_reference_value::Type::Skip(skip.clone())),
                }),
            }),
            container_layer: Some(ContainerLayerReferenceValues {
                binary: Some(BinaryReferenceValue {
                    r#type: Some(binary_reference_value::Type::Skip(skip.clone())),
                }),
                configuration: Some(BinaryReferenceValue {
                    r#type: Some(binary_reference_value::Type::Skip(skip.clone())),
                }),
            }),
        })),
    }
}

/// Creates ProgramExecutorTeeConfigConstraints for the given program
///
/// # Arguments
/// * `program` - The Python program code (will be base64 encoded)
/// * `worker_reference_values` - Optional reference values for distributed workers
///
/// # Returns
/// ProgramExecutorTeeConfigConstraints proto message
pub fn create_program_executor_config_constraints(
    program: &str,
    worker_reference_values: Option<ReferenceValues>,
) -> ProgramExecutorTeeConfigConstraints {
    let mut config_constraints = ProgramExecutorTeeConfigConstraints {
        program: BASE64.encode(program.as_bytes()).into_bytes(),
        num_runs: MAX_NUM_RUNS,
        ..Default::default()
    };

    if let Some(ref_values) = worker_reference_values {
        config_constraints.worker_reference_values =
            BASE64.encode(ref_values.encode_to_vec()).into_bytes();
    }

    config_constraints
}

/// Creates a DataAccessPolicy for ProgramExecutorTee
///
/// # Arguments
/// * `reference_values` - Optional reference values derived from TEE evidence.
///   If None, uses test reference values with skip verification.
/// * `program` - The Python program code (for config_constraints)
///
/// # Returns
/// DataAccessPolicy configured for program_executor pipeline
pub fn create_program_executor_access_policy(
    reference_values: Option<ReferenceValues>,
    program: &str,
) -> DataAccessPolicy {
    // Use provided reference values or create test reference values
    let ref_values = reference_values.unwrap_or_else(get_test_reference_values);

    // Create ProgramExecutorTeeConfigConstraints and pack into Any
    let config_constraints = create_program_executor_config_constraints(program, None);
    let config_constraints_any = Any {
        type_url: "type.googleapis.com/fcp.confidentialcompute.ProgramExecutorTeeConfigConstraints"
            .to_string(),
        value: config_constraints.encode_to_vec(),
    };

    // Create a transform that processes client data
    let transform = Transform {
        src_node_ids: vec![0], // Read from node 0 (client uploads/input data)
        dst_node_ids: vec![1], // Write to node 1 (results/output data)
        application: Some(ApplicationMatcher {
            reference_values: Some(ref_values),
            ..Default::default()
        }),
        config_constraints: Some(config_constraints_any),
        ..Default::default()
    };

    // Create the pipeline variant policy containing the transform
    let pipeline_variant_policy =
        PipelineVariantPolicy { transforms: vec![transform], ..Default::default() };

    // Create a logical pipeline policy with one variant
    let logical_pipeline_policy =
        LogicalPipelinePolicy { instances: vec![pipeline_variant_policy], ..Default::default() };

    // Create the data access policy with the named logical pipeline
    let mut pipelines = HashMap::new();
    pipelines.insert("program_executor_pipeline".to_string(), logical_pipeline_policy);

    DataAccessPolicy { pipelines, ..Default::default() }
}

// /// Creates and serializes a sample DataAccessPolicy for ProgramExecutorTee
// ///
// /// # Arguments
// /// * `reference_values` - Optional reference values derived from TEE evidence
// /// * `program` - The Python program code
// pub fn create_sample_access_policy(
//     reference_values: Option<ReferenceValues>,
//     program: &str,
// ) -> Vec<u8> {
//     let policy = create_program_executor_access_policy(reference_values, program);
//     policy.encode_to_vec()
// }
