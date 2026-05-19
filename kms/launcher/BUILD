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

load("@rules_rust//rust:defs.bzl", "rust_binary")

package(features = ["layering_check"])

rust_binary(
    name = "launcher",
    srcs = ["main.rs"],
    deps = [
        # KMS storage actor and proto dependencies
        "//storage_service:storage_actor",
        "//storage_service:session_v1_service_prost_proto",
        "//key_management_service:access_policy_prost_proto",
        "//key_management_service:kms_prost_proto",
        "//:prost_proto_conversion",

        # TCP runtime and proto
        "@trusted_computations_platform//runtime:tcp_runtime",
        "@trusted_computations_platform//proto:tcp_proto",

        # Oak dependencies
        "@oak//oak_proto_rust",
        "@oak//oak_attestation_types",
        "@oak//oak_session",
        "@oak//oak_time",
        "@oak//oak_sdk/common:oak_sdk_common",
        "@oak//oak_sdk/standalone:oak_sdk_standalone",

        # Crate dependencies
        "@oak_crates_index//:p256",
        "@oak_crates_index//:prost",
        "@oak_crates_index//:rand_core",
        "@oak_crates_index//:tokio",
        "@oak_crates_index//:tokio-stream",
        "@oak_crates_index//:tonic",
    ],
)
