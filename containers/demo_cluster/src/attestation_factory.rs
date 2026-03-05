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

use oak_proto_rust::oak::{
    attestation::v1::{
        binary_reference_value, extracted_evidence::EvidenceValues, kernel_binary_reference_value,
        reference_values, root_layer_data::Report, tcb_version_reference_value,
        text_reference_value, AmdSevReferenceValues, ApplicationLayerReferenceValues,
        BinaryReferenceValue, ContainerLayerReferenceValues, Digests, ExtractedEvidence,
        InsecureReferenceValues, KernelBinaryReferenceValue, KernelDigests, KernelLayerData,
        KernelLayerReferenceValues, OakContainersReferenceValues,
        OakRestrictedKernelReferenceValues, ReferenceValues, RootLayerData,
        RootLayerReferenceValues, SkipVerification, StringLiterals, SystemLayerReferenceValues,
        TcbVersionReferenceValue, TextReferenceValue,
    },
    RawDigest,
};

/// Creates digest-based reference values for extracted evidence.
pub fn create_reference_values_for_extracted_evidence(
    extracted_evidence: ExtractedEvidence,
) -> ReferenceValues {
    let r#type = match extracted_evidence.evidence_values.expect("no evidence") {
        EvidenceValues::OakRestrictedKernel(rk) => {
            let application = rk.application_layer.expect("no application layer evidence");
            let config = application.config.expect("no application config digest");
            Some(reference_values::Type::OakRestrictedKernel(OakRestrictedKernelReferenceValues {
                root_layer: Some(root_layer_reference_values_from_evidence(
                    rk.root_layer.expect("no root layer evidence"),
                )),
                kernel_layer: Some(kernel_layer_reference_values_from_evidence(
                    rk.kernel_layer.expect("no kernel layer evidence"),
                )),
                application_layer: Some(ApplicationLayerReferenceValues {
                    binary: Some(BinaryReferenceValue {
                        r#type: Some(binary_reference_value::Type::Digests(Digests {
                            digests: vec![application
                                .binary
                                .expect("no application binary digest")],
                        })),
                    }),
                    // We don't currently specify configuration values for Oak Containers
                    // applications, so skip for now if the sha2_256 value is empty.
                    configuration: if config.sha2_256.is_empty() {
                        Some(BinaryReferenceValue {
                            r#type: Some(binary_reference_value::Type::Skip(SkipVerification {})),
                        })
                    } else {
                        Some(BinaryReferenceValue {
                            r#type: Some(binary_reference_value::Type::Digests(Digests {
                                digests: vec![config],
                            })),
                        })
                    },
                }),
            }))
        }
        EvidenceValues::OakContainers(oc) => {
            let system = oc.system_layer.expect("no system layer evidence");
            let container = oc.container_layer.expect("no container layer evidence");
            Some(reference_values::Type::OakContainers(OakContainersReferenceValues {
                root_layer: Some(root_layer_reference_values_from_evidence(
                    oc.root_layer.expect("no root layer evidence"),
                )),
                kernel_layer: Some(kernel_layer_reference_values_from_evidence(
                    oc.kernel_layer.expect("no kernel layer evidence"),
                )),
                system_layer: Some(SystemLayerReferenceValues {
                    system_image: Some(BinaryReferenceValue {
                        r#type: Some(binary_reference_value::Type::Digests(Digests {
                            digests: vec![system.system_image.expect("no system image digest")],
                        })),
                    }),
                }),
                container_layer: Some(ContainerLayerReferenceValues {
                    binary: Some(BinaryReferenceValue {
                        r#type: Some(binary_reference_value::Type::Digests(Digests {
                            digests: vec![container.bundle.expect("no container bundle digest")],
                        })),
                    }),
                    configuration: Some(BinaryReferenceValue {
                        r#type: Some(binary_reference_value::Type::Digests(Digests {
                            digests: vec![container.config.expect("no container config digest")],
                        })),
                    }),
                }),
            }))
        }
        EvidenceValues::Cb(_) => panic!("not yet supported"),
        EvidenceValues::Standalone(_) => panic!("not yet supported"),
    };
    ReferenceValues { r#type }
}

fn root_layer_reference_values_from_evidence(
    root_layer: RootLayerData,
) -> RootLayerReferenceValues {
    #[allow(deprecated)]
    let amd_sev = root_layer.report.clone().and_then(|report| match report {
        Report::SevSnp(r) => {
            let tcb = r.reported_tcb.unwrap();
            let rv = TcbVersionReferenceValue {
                r#type: Some(tcb_version_reference_value::Type::Minimum(tcb)),
            };

            Some(AmdSevReferenceValues {
                min_tcb_version: Some(tcb),
                milan: Some(rv),
                genoa: Some(rv),
                turin: Some(rv),
                stage0: Some(BinaryReferenceValue {
                    r#type: Some(binary_reference_value::Type::Digests(Digests {
                        digests: vec![RawDigest {
                            sha2_384: r.initial_measurement,
                            ..Default::default()
                        }],
                    })),
                }),
                allow_debug: r.debug,
                // check_vcek_cert_expiry: true,
            })
        }
        _ => None,
    });
    let intel_tdx = if let Some(Report::Tdx(_)) = root_layer.report.clone() {
        panic!("not yet supported");
    } else {
        None
    };
    let insecure = root_layer.report.and_then(|report| match report {
        Report::Fake(_) => Some(InsecureReferenceValues {}),
        _ => None,
    });
    RootLayerReferenceValues { amd_sev, intel_tdx, insecure }
}

fn kernel_layer_reference_values_from_evidence(
    kernel_layer: KernelLayerData,
) -> KernelLayerReferenceValues {
    #[allow(deprecated)]
    KernelLayerReferenceValues {
        kernel: Some(KernelBinaryReferenceValue {
            r#type: Some(kernel_binary_reference_value::Type::Digests(KernelDigests {
                image: Some(Digests {
                    digests: vec![kernel_layer.kernel_image.expect("no kernel image digest")],
                }),
                setup_data: Some(Digests {
                    digests: vec![kernel_layer
                        .kernel_setup_data
                        .expect("no kernel setup data digest")],
                }),
            })),
        }),
        kernel_cmd_line_text: Some(TextReferenceValue {
            r#type: Some(text_reference_value::Type::StringLiterals(StringLiterals {
                value: vec![kernel_layer.kernel_raw_cmd_line.expect("no kernel command-line")],
            })),
        }),
        init_ram_fs: Some(BinaryReferenceValue {
            r#type: Some(binary_reference_value::Type::Digests(Digests {
                digests: vec![kernel_layer.init_ram_fs.expect("no initial ram disk digest")],
            })),
        }),
        memory_map: Some(BinaryReferenceValue {
            r#type: Some(binary_reference_value::Type::Digests(Digests {
                digests: vec![kernel_layer.memory_map.expect("no memory map digest")],
            })),
        }),
        acpi: Some(BinaryReferenceValue {
            r#type: Some(binary_reference_value::Type::Digests(Digests {
                digests: vec![kernel_layer.acpi.expect("no acpi digest")],
            })),
        }),
    }
}
