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

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("@rules_python//python:defs.bzl", "py_test")
load("@rules_rust//rust:defs.bzl", "rust_library", "rust_test")
load("@rules_rust_prost//:defs.bzl", "rust_prost_library")

package(features = ["layering_check"])

# ===== Proto Libraries =====

rust_prost_library(
    name = "confidential_transform_prost_proto",
    proto = "@federated-compute//fcp/protos/confidentialcompute:confidential_transform_proto",
)

rust_prost_library(
    name = "program_executor_tee_prost_proto",
    proto = "@federated-compute//fcp/protos/confidentialcompute:program_executor_tee_proto",
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
    name = "kms_prost_proto",
    proto = "@federated-compute//fcp/protos/confidentialcompute:kms_proto",
)

rust_prost_library(
    name = "duration_prost_proto",
    proto = "@com_google_protobuf//:duration_proto",
)

rust_prost_library(
    name = "access_policy_prost_proto",
    proto = "@federated-compute//fcp/protos/confidentialcompute:access_policy_proto",
)

rust_prost_library(
    name = "reference_value_prost_proto",
    proto = "@oak//proto/attestation:reference_value_proto",
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
    name = "data_read_write_prost_proto",
    proto = "@federated-compute//fcp/protos/confidentialcompute:data_read_write_proto",
)

rust_prost_library(
    name = "executor_prost_proto",
    proto = "@org_tensorflow_federated//tensorflow_federated/proto/v0:executor_proto",
)

# ===== Key Derivation Library =====

rust_library(
    name = "key_derivation",
    srcs = ["src/key_derivation.rs"],
    deps = [
        "@boringssl//rust/bssl-crypto",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:coset",
        "@oak_crates_index//:rand",
        "@oak_crates_index//:hpke",
        "@oak_crates_index//:aes-gcm-siv",
    ],
)

rust_test(
    name = "key_derivation_test",
    crate = ":key_derivation",
)

# ===== Data Read Write Service =====

rust_library(
    name = "data_read_write_service",
    srcs = ["src/data_read_write_service.rs"],
    visibility = ["//visibility:public"],
    deps = [
        ":key_derivation",
        ":blob_header_prost_proto",
        ":confidential_transform_prost_proto",
        ":data_read_write_prost_proto",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:coset",
        "@oak_crates_index//:prost",
        "@oak_crates_index//:rand",
        "@oak_crates_index//:tokio",
        "@oak_crates_index//:tokio-stream",
        "@oak_crates_index//:tonic",
        "@oak_crates_index//:tracing",
    ],
)

rust_test(
    name = "data_read_write_service_test",
    crate = ":data_read_write_service",
)

# ===== Local Checkpoint Generation (extended with float/multi-tensor support) =====

cc_library(
    name = "generate_checkpoint",
    srcs = ["cc/generate_checkpoint.cc"],
    hdrs = ["cc/generate_checkpoint.h"],
    deps = [
        "@com_google_absl//absl/log:check",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:mutable_string_data",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/core:tensor",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:checkpoint_builder",
        "@org_tensorflow_federated//tensorflow_federated/cc/core/impl/aggregation/protocol:federated_compute_checkpoint_builder",
    ],
)

# ===== Checkpoint FFI (TensorFlow Federated) =====

cc_library(
    name = "checkpoint_ffi",
    srcs = ["cc/checkpoint_ffi.cc"],
    hdrs = ["cc/checkpoint_ffi.h"],
    deps = [
        ":generate_checkpoint",
    ],
)

rust_library(
    name = "checkpoint_rs",
    srcs = ["src/checkpoint_ffi.rs"],
    visibility = ["//visibility:public"],
    deps = [
        ":checkpoint_ffi",
    ],
)

rust_test(
    name = "checkpoint_rs_test",
    crate = ":checkpoint_rs",
)

# ===== C++ FFI Wrapper for FakeDataReadWriteService =====
# Uses FakeDataReadWriteService from confidential-federated-compute-program directly

cc_library(
    name = "fake_data_read_write_ffi",
    srcs = ["cc/fake_data_read_write_ffi.cc"],
    hdrs = ["cc/fake_data_read_write_ffi.h"],
    deps = [
        "@confidential-federated-compute-program//program_executor_tee/program_context/cc:fake_data_read_write_service",
        "@confidential-federated-compute-program//program_executor_tee/program_context/cc:kms_helper",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
    ],
)

# Test to verify FFI wrapper works
cc_test(
    name = "fake_data_read_write_ffi_test",
    srcs = ["cc/fake_data_read_write_ffi_test.cc"],
    deps = [
        ":fake_data_read_write_ffi",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)

# ===== Rust FFI Bindings =====

rust_library(
    name = "fake_data_read_write_rs",
    srcs = ["src/fake_data_read_write_ffi.rs"],
    visibility = ["//visibility:public"],
    deps = [
        ":fake_data_read_write_ffi",
    ],
)

rust_test(
    name = "fake_data_read_write_rs_test",
    crate = ":fake_data_read_write_rs",
)

# ===== ProgramExecutorClient =====

rust_library(
    name = "program_executor_client",
    srcs = ["src/program_executor_client.rs"],
    visibility = ["//visibility:public"],
    deps = [
        ":confidential_transform_prost_proto",
        ":program_executor_tee_prost_proto",
        ":reference_value_prost_proto",
        ":crypto_prost_proto",
        ":any_prost_proto",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:base64",
        "@oak_crates_index//:prost",
        "@oak_crates_index//:prost-types",
        "@oak_crates_index//:tokio",
        "@oak_crates_index//:tokio-stream",
        "@oak_crates_index//:tonic",
        "@oak_crates_index//:tracing",
    ],
)

rust_test(
    name = "program_executor_client_test",
    crate = ":program_executor_client",
)

# ===== Main Demo Binary =====

load("@rules_rust//rust:defs.bzl", "rust_binary")

# ===== MNIST Training Program Test =====
# Tests the Python computations from PROGRAM_MNIST_TRAINING_TEMPLATE locally
# without requiring a TEE or computation_runner binary.

py_test(
    name = "test_mnist_program",
    srcs = ["test_mnist_program.py"],
    tags = ["noasan"],
    deps = [
        "@federated_language//federated_language",
        "@org_tensorflow_federated//tensorflow_federated",
        "@pypi//numpy",
        "@pypi//tensorflow",
    ],
)

py_test(
    name = "run_mnist_local",
    srcs = ["run_mnist_local.py"],
    tags = ["noasan"],
    deps = [
        "@federated_language//federated_language",
        "@org_tensorflow_federated//tensorflow_federated",
        "@pypi//numpy",
        "@pypi//tensorflow",
    ],
)

# ===== Main Demo Binary =====

rust_binary(
    name = "demo_cluster",
    srcs = [
        "src/main.rs",
        "src/kms_client.rs",
        "src/program_executor_client.rs",
        "src/attestation_factory.rs",
        "src/client_simulator.rs",
        "src/key_derivation.rs",
        "src/launcher_module/mod.rs",
        "src/launcher_module/lib.rs",
        "src/launcher_module/qemu.rs",
        "src/launcher_module/server.rs",
        ],
    deps = [
        ":checkpoint_rs",
        ":fake_data_read_write_rs",
        ":key_derivation",
        ":data_read_write_service",
        ":program_executor_client",
        ":confidential_transform_prost_proto",
        ":program_executor_tee_prost_proto",
        ":access_policy_prost_proto",
        ":reference_value_prost_proto",
        ":crypto_prost_proto",
        ":kms_prost_proto",
        ":any_prost_proto",
        ":duration_prost_proto",
        ":blob_header_prost_proto",
        ":blob_data_prost_proto",
        ":data_read_write_prost_proto",
        ":executor_prost_proto",
        "@oak_crates_index//:hpke",
        "@oak_crates_index//:aes-gcm-siv",
        "@boringssl//rust/bssl-crypto",
        "@oak//oak_attestation_verification",
        "@oak//oak_crypto",
        "@oak//oak_proto_rust",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:base64",
        "@oak_crates_index//:clap",
        "@oak_crates_index//:tokio",
        "@oak_crates_index//:tokio-stream",
        "@oak_crates_index//:tokio-vsock",
        "@oak_crates_index//:tonic",
        "@oak_crates_index//:tracing",
        "@oak_crates_index//:tracing-subscriber",
        "@oak//oak_proto_rust/grpc",
        "@oak_crates_index//:bytes",
        "@oak_crates_index//:command-fds",
        "@oak_crates_index//:futures",
        "@oak_crates_index//:opentelemetry-proto",
        "@oak_crates_index//:prost",
        "@oak_crates_index//:prost-types",
        "@oak_crates_index//:sha2",
        "@oak_crates_index//:log",
        "@oak_crates_index//:nix",
        "@oak_crates_index//:async-stream",
        "@oak_crates_index//:coset",
        "@oak_crates_index//:rand",
    ],
)
