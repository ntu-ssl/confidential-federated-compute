# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_rust//rust:defs.bzl", "rust_binary", "rust_library")
load("@rules_rust_prost//:defs.bzl", "rust_prost_library")

package(features = ["layering_check"])

# ============================================================================
# Reuse proto libraries from federated-compute (same as kms-flow-simulation)
# ============================================================================

rust_prost_library(
    name = "access_policy_prost_proto",
    proto = "@federated-compute//fcp/protos/confidentialcompute:access_policy_proto",
)

rust_prost_library(
    name = "kms_prost_proto",
    proto = "@federated-compute//fcp/protos/confidentialcompute:kms_proto",
)

rust_prost_library(
    name = "key_prost_proto",
    proto = "@federated-compute//fcp/protos/confidentialcompute:key_proto",
)

rust_prost_library(
    name = "blob_header_prost_proto",
    proto = "@federated-compute//fcp/protos/confidentialcompute:blob_header_proto",
)

rust_prost_library(
    name = "blob_data_prost_proto",
    proto = "@federated-compute//fcp/protos/confidentialcompute:blob_data_proto",
)

rust_prost_library(
    name = "confidential_transform_prost_proto",
    proto = "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_proto",
)

rust_prost_library(
    name = "crypto_prost_proto",
    proto = "@oak//proto/crypto:crypto_proto",
)

rust_prost_library(
    name = "any_prost_proto",
    proto = "@com_google_protobuf//:any_proto",
)

rust_prost_library(
    name = "reference_value_prost_proto",
    proto = "@oak//proto/attestation:reference_value_proto",
)

rust_prost_library(
    name = "duration_prost_proto",
    proto = "@com_google_protobuf//:duration_proto",
)

# ============================================================================
# Core server libraries
# ============================================================================

# Models and data structures
rust_library(
    name = "models",
    srcs = ["src/models/mod.rs"],
    deps = [
        ":access_policy_prost_proto",
        ":kms_prost_proto",
        "@oak_crates_index//:serde",
        "@oak_crates_index//:serde_json",
        "@oak_crates_index//:uuid",
        "@oak_crates_index//:chrono",
        "@oak_crates_index//:anyhow",
    ],
)

# Session and launcher management
rust_library(
    name = "services",
    srcs = [
        "src/services/mod.rs",
        "src/services/key_derivation.rs",
        "src/services/kms_client.rs",
        "src/services/test_concat_client.rs",
        "src/services/client_simulator.rs",
        "src/services/session_registry.rs",
        "src/services/policy_builder.rs",
        "src/services/execution.rs",
        "src/services/launcher_manager.rs",
        "src/services/attestation_factory.rs",
    ],
    crate_root = "src/services/mod.rs",
    deps = [
        ":access_policy_prost_proto",
        ":reference_value_prost_proto",
        ":kms_prost_proto",
        ":duration_prost_proto",
        ":confidential_transform_prost_proto",
        ":crypto_prost_proto",
        ":blob_header_prost_proto",
        ":blob_data_prost_proto",
        ":models",
        "@boringssl//rust/bssl-crypto",
        "@oak//oak_proto_rust",
        "@oak//oak_proto_rust/grpc",
        "@oak//oak_attestation_verification",
        "@oak_crates_index//:tokio",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:serde",
        "@oak_crates_index//:serde_json",
        "@oak_crates_index//:tracing",
        "@oak_crates_index//:prost",
        "@oak_crates_index//:sha2",
        "@oak_crates_index//:hex",
        "@oak_crates_index//:chrono",
        "@oak_crates_index//:uuid",
        "@oak_crates_index//:tonic",
        "@oak_crates_index//:tokio-stream",
        "@oak_crates_index//:rand",
        "@oak_crates_index//:coset",
    ],
)

# HTTP routes and handlers
rust_library(
    name = "server",
    srcs = [
        "src/server/mod.rs",
        "src/server/routes.rs",
        "src/server/handlers.rs",
    ],
    crate_root = "src/server/mod.rs",
    deps = [
        ":models",
        ":services",
        "@oak//oak_attestation_verification",
        "@oak//oak_proto_rust",
        "@oak_crates_index//:axum",
        "@oak_crates_index//:chrono",
        "@oak_crates_index//:tokio",
        "@oak_crates_index//:tower",
        "@oak_crates_index//:tower-http",
        "@oak_crates_index//:serde",
        "@oak_crates_index//:serde_json",
        "@oak_crates_index//:tracing",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:prost",
        "@oak_crates_index//:sha2",
        "@oak_crates_index//:hex",
    ],
)

# Main binary
rust_binary(
    name = "workshop_server",
    srcs = [
        "src/main.rs",
        "src/models/mod.rs",
        "src/server/mod.rs",
        "src/server/routes.rs",
        "src/server/handlers.rs",
        "src/services/mod.rs",
        "src/services/kms_client.rs",
        "src/services/key_derivation.rs",
        "src/services/test_concat_client.rs",
        "src/services/client_simulator.rs",
        "src/services/session_registry.rs",
        "src/services/policy_builder.rs",
        "src/services/execution.rs",
        "src/services/launcher_manager.rs",
        "src/services/attestation_factory.rs",
        "src/launcher_module/mod.rs",
        "src/launcher_module/lib.rs",
        "src/launcher_module/qemu.rs",
        "src/launcher_module/server.rs",
    ],
    deps = [
        ":access_policy_prost_proto",
        ":reference_value_prost_proto",
        ":kms_prost_proto",
        ":duration_prost_proto",
        ":crypto_prost_proto",
        ":confidential_transform_prost_proto",
        ":blob_header_prost_proto",
        ":blob_data_prost_proto",
        "@oak//oak_proto_rust",
        "@oak//oak_proto_rust/grpc",
        "@oak//oak_attestation_verification",
        "@oak//oak_sdk/containers:oak_sdk_containers",
        "@oak_crates_index//:aes-gcm-siv",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:axum",
        "@oak_crates_index//:async-stream",
        "@oak_crates_index//:bytes",
        "@oak_crates_index//:chrono",
        "@oak_crates_index//:clap",
        "@oak_crates_index//:command-fds",
        "@oak_crates_index//:futures",
        "@oak_crates_index//:hex",
        "@oak_crates_index//:hpke",
        "@oak_crates_index//:log",
        "@oak_crates_index//:nix",
        "@oak_crates_index//:opentelemetry-proto",
        "@oak_crates_index//:prost",
        "@oak_crates_index//:serde",
        "@oak_crates_index//:serde_json",
        "@oak_crates_index//:sha2",
        "@oak_crates_index//:tokio",
        "@oak_crates_index//:tokio-stream",
        "@oak_crates_index//:tonic",
        "@oak_crates_index//:tokio-vsock",
        "@oak_crates_index//:tower",
        "@oak_crates_index//:tower-http",
        "@oak_crates_index//:tracing",
        "@oak_crates_index//:tracing-subscriber",
        "@oak_crates_index//:uuid",
        "@oak_crates_index//:which",
        "@oak_crates_index//:rand",
        "@oak_crates_index//:coset",
        "@boringssl//rust/bssl-crypto",
    ],
)
