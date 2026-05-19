# Copyright 2026 Google LLC.
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

load("@oak//bazel:defs.bzl", "oci_runtime_bundle")
load("@rules_oci//oci:defs.bzl", "oci_image")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load("@rules_rust//rust:defs.bzl", "rust_binary")

package(features = ["layering_check"])

pkg_tar(
    name = "tar",
    testonly = True,
    srcs = [":main"],
)

oci_image(
    name = "oci_image",
    testonly = True,
    base = "@distroless_cc_debian12_base",
    entrypoint = ["/main"],
    tars = [":tar"],
)

oci_runtime_bundle(
    name = "oci_runtime_bundle",
    testonly = True,
    image = ":oci_image",
    visibility = ["//visibility:public"],
)

rust_binary(
    name = "main",
    testonly = True,
    srcs = ["src/main.rs"],
    compile_data = [
        "//:insecure_reference_values",
    ],
    rustc_env = {
        "INSECURE_REFERENCE_VALUES": "$(location //:insecure_reference_values)",
    },
    deps = [
        "//storage_service:session_config",
        "//storage_service:session_v1_service_prost_proto",
        "//storage_service:storage",
        "//storage_service:storage_prost_proto",
        "//:prost_proto_conversion",
        "@oak//oak_attestation_types",
        "@oak//oak_proto_rust",
        "@oak//oak_session",
        "@oak//oak_time",
        "@oak//oak_time:oak_time_std",
        "@oak//oak_sdk/common:oak_sdk_common",
        "@oak//oak_sdk/containers:oak_sdk_containers",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:prost",
        "@oak_crates_index//:tokio",
        "@oak_crates_index//:tokio-stream",
        "@oak_crates_index//:tonic",
        "@oak_crates_index//:tracing",
        "@oak_crates_index//:tracing-subscriber",
    ],
)

rust_binary(
    name = "storage_proxy_launcher",
    srcs = [
        "src/launcher_main.rs",
        "src/launcher/lib.rs",
        "src/launcher/mod.rs",
        "src/launcher/qemu.rs",
        "src/launcher/server.rs",
    ],
    crate_root = "src/launcher_main.rs",
    deps = [
        "@oak//oak_proto_rust",
        "@oak//oak_proto_rust/grpc",
        "@oak_crates_index//:anyhow",
        "@oak_crates_index//:bytes",
        "@oak_crates_index//:futures",
        "@oak_crates_index//:log",
        "@oak_crates_index//:nix",
        "@oak_crates_index//:prost",
        "@oak_crates_index//:tokio",
        "@oak_crates_index//:tokio-stream",
        "@oak_crates_index//:tonic",
        "@oak_crates_index//:tracing",
        "@oak_crates_index//:tracing-subscriber",
    ],
)
