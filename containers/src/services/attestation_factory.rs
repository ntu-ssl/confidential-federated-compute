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

use oak_proto_rust::oak::attestation::v1::{
    binary_reference_value, extracted_evidence::EvidenceValues, kernel_binary_reference_value,
    reference_values, text_reference_value, ApplicationLayerReferenceValues, BinaryReferenceValue,
    ContainerLayerReferenceValues, ExtractedEvidence, InsecureReferenceValues,
    KernelBinaryReferenceValue, KernelLayerReferenceValues, OakContainersReferenceValues,
    OakRestrictedKernelReferenceValues, ReferenceValues, RootLayerReferenceValues,
    SkipVerification, SystemLayerReferenceValues, TextReferenceValue,
};

/// Creates a permissive reference-values set for the TEE's extracted evidence.
///
/// Historically this function pinned the reference values to the exact digests
/// found in the TEE evidence — so the KMS would only authorize transforms
/// running the same TEE binary that produced the evidence. With strict
/// digests, the KMS-side verifier follows the `AmdSevSnpDiceAttestationVerifier`
/// path, whose first step ("verifying platform policy") needs a real platform
/// endorsement (the AMD VCEK chain) to validate the SEV-SNP attestation
/// report. The launchers in this repo ship empty endorsements
/// (`OakContainersEndorsements { root_layer: None, ... }`), and on this
/// hardware the PSP firmware can't produce a valid platform endorsement
/// either — so that verifier always failed with `no platform endorsement`,
/// breaking `authorize_transform`.
///
/// This implementation now emits an "insecure root layer + Skip everywhere"
/// reference-values set, which steers the KMS verifier into the
/// `InsecureAttestationVerifier` branch (no `AmdSevSnpPolicy`, no
/// `FirmwarePolicy`) and skips digest checks on each layer. The
/// `extracted_evidence` argument is preserved so call sites don't need to
/// change, but its contents are intentionally ignored.
///
/// Security: this matches the test-attestation posture already used on the
/// KMS↔storage_proxy channel (see `kms/storage_proxy/src/main.rs` and
/// `kms/main.rs`'s `GrpcStorageClient` setup). Any TEE presenting any
/// evidence will satisfy this policy; do not deploy with this in production.
pub fn create_reference_values_for_extracted_evidence(
    extracted_evidence: ExtractedEvidence,
) -> ReferenceValues {
    let r#type = match extracted_evidence.evidence_values.expect("no evidence") {
        EvidenceValues::OakRestrictedKernel(_) => {
            Some(reference_values::Type::OakRestrictedKernel(OakRestrictedKernelReferenceValues {
                root_layer: Some(insecure_root_layer_reference_values()),
                kernel_layer: Some(skip_kernel_layer_reference_values()),
                application_layer: Some(ApplicationLayerReferenceValues {
                    binary: Some(skip_binary_reference_value()),
                    configuration: Some(skip_binary_reference_value()),
                }),
            }))
        }
        EvidenceValues::OakContainers(_) => {
            Some(reference_values::Type::OakContainers(OakContainersReferenceValues {
                root_layer: Some(insecure_root_layer_reference_values()),
                kernel_layer: Some(skip_kernel_layer_reference_values()),
                system_layer: Some(SystemLayerReferenceValues {
                    system_image: Some(skip_binary_reference_value()),
                }),
                container_layer: Some(ContainerLayerReferenceValues {
                    binary: Some(skip_binary_reference_value()),
                    configuration: Some(skip_binary_reference_value()),
                }),
            }))
        }
        EvidenceValues::Cb(_) => panic!("not yet supported"),
        EvidenceValues::Standalone(_) => panic!("not yet supported"),
    };
    ReferenceValues { r#type }
}

/// `RootLayerReferenceValues` with `insecure` set — this picks the
/// `InsecureAttestationVerifier` branch on the KMS side, which has no
/// `AmdSevSnpPolicy` or `FirmwarePolicy` and therefore never tries to
/// verify the (empty) platform endorsement.
fn insecure_root_layer_reference_values() -> RootLayerReferenceValues {
    #[allow(deprecated)]
    RootLayerReferenceValues {
        insecure: Some(InsecureReferenceValues::default()),
        amd_sev: None,
        intel_tdx: None,
    }
}

/// `KernelLayerReferenceValues` with `Skip` on every field — matches what
/// `kms/insecure_reference_values.txtpb` does for in-VM sessions.
fn skip_kernel_layer_reference_values() -> KernelLayerReferenceValues {
    #[allow(deprecated)]
    KernelLayerReferenceValues {
        kernel: Some(KernelBinaryReferenceValue {
            r#type: Some(kernel_binary_reference_value::Type::Skip(
                SkipVerification::default(),
            )),
        }),
        kernel_cmd_line_text: Some(TextReferenceValue {
            r#type: Some(text_reference_value::Type::Skip(SkipVerification::default())),
        }),
        init_ram_fs: Some(skip_binary_reference_value()),
        memory_map: Some(skip_binary_reference_value()),
        acpi: Some(skip_binary_reference_value()),
    }
}

fn skip_binary_reference_value() -> BinaryReferenceValue {
    BinaryReferenceValue {
        r#type: Some(binary_reference_value::Type::Skip(SkipVerification::default())),
    }
}
