// Copyright 2025 The Trusted Computations Platform Authors.
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
use anyhow::{Context, Result};
use oak_attestation_types::{attester::Attester, endorser::Endorser};
use oak_crypto::signer::Signer;
use oak_proto_rust::oak::Variant;
use oak_proto_rust::oak::attestation::v1::{
    ApplicationKeys, Endorsements, EventLog, Evidence,
};
use oak_sdk_common::{StaticAttester, StaticEndorser};
use oak_sdk_containers::{InstanceSessionBinder, OrchestratorClient};
use oak_session::session_binding::SessionBinder;
use oak_session::{attestation::AttestationType, config::SessionConfig, handshake::HandshakeType};
use oak_session_endorsed_evidence::EndorsedEvidenceBindableAssertionGenerator;
use once_cell::sync::Lazy;
use p256::ecdsa::SigningKey;
use std::ffi::c_void;
use std::sync::Arc;
use tokio::runtime::{Builder, Runtime};

const ASSERTION_ID: &str = "cfc_program_worker";

// Constants matching computation_runner.cc's fake attestation mode.
const FAKE_ATTESTER_ID: &str = "fake_attester";
const FAKE_EVENT: &[u8] = b"fake event";
const FAKE_PLATFORM: &[u8] = b"fake platform";

/// Generate a static fake signing key (deterministic for reproducibility).
static FAKE_SIGNING_KEY: Lazy<SigningKey> = Lazy::new(|| {
    SigningKey::random(&mut rand_core::OsRng)
});

/// A simple attester that returns fake evidence with a real signing key,
/// matching the C++ fake verifier expectations.
struct FakeAttester;

impl Attester for FakeAttester {
    fn extend(&mut self, _encoded_event: &[u8]) -> anyhow::Result<()> {
        Ok(())
    }

    fn quote(&self) -> anyhow::Result<Evidence> {
        let verifying_key_bytes =
            FAKE_SIGNING_KEY.verifying_key().to_sec1_bytes().to_vec();
        #[allow(deprecated)]
        Ok(Evidence {
            root_layer: None,
            application_keys: Some(ApplicationKeys {
                signing_public_key_certificate: verifying_key_bytes,
                ..Default::default()
            }),
            layers: vec![],
            event_log: Some(EventLog {
                encoded_events: vec![FAKE_EVENT.to_vec()],
            }),
            transparent_event_log: None,
            signed_user_data_certificate: vec![],
        })
    }
}

/// A simple endorser that returns fake endorsements matching the C++ fake verifier.
struct FakeEndorser;

impl Endorser for FakeEndorser {
    fn endorse(&self, _evidence: Option<&Evidence>) -> anyhow::Result<Endorsements> {
        Ok(Endorsements {
            events: vec![],
            initial: None,
            platform: Some(Variant {
                id: b"fake".to_vec(),
                value: FAKE_PLATFORM.to_vec(),
            }),
            r#type: None,
        })
    }
}

/// A session binder that signs with the fake signing key.
struct FakeSessionBinder;

impl SessionBinder for FakeSessionBinder {
    fn bind(&self, bound_data: &[u8]) -> Vec<u8> {
        FAKE_SIGNING_KEY.sign(bound_data)
    }
}

pub static RUNTIME: Lazy<Runtime> =
    Lazy::new(|| Builder::new_multi_thread().enable_all().build().unwrap());

/// Initializes the Tokio runtime context.
#[no_mangle]
pub extern "C" fn init_tokio_runtime() {
    Lazy::force(&RUNTIME);
    println!("Rust runtime is running");
}

/// Enters the Tokio runtime context and returns an opaque handle.
///
/// The caller is responsible for calling `exit_tokio_runtime` with this
/// handle to exit the context. Failure to do so will lead to resource leaks.
#[no_mangle]
pub extern "C" fn enter_tokio_runtime() -> *mut c_void {
    println!("Rust: enter_tokio_runtime called");
    // Create an EnterGuard and keep it alive on the heap.
    let guard = Box::new(RUNTIME.enter());
    // Return a raw, type-erased pointer to the C++ caller.
    Box::into_raw(guard) as *mut c_void
}

/// Exits the Tokio runtime context using the handle from `enter_tokio_runtime`.
#[no_mangle]
pub extern "C" fn exit_tokio_runtime(guard_ptr: *mut c_void) {
    println!("Rust: exit_tokio_runtime called");
    if guard_ptr.is_null() {
        return;
    }
    // Reconstruct the Box from the raw pointer. The guard is dropped at the
    // end of this scope, which exits the Tokio context.
    unsafe {
        let _ = Box::from_raw(guard_ptr as *mut tokio::runtime::EnterGuard);
    }
}

fn create_session_config_internal() -> Result<*mut SessionConfig> {
    let channel = RUNTIME
        .block_on(oak_sdk_containers::default_orchestrator_channel())
        .context("Failed to create orchestrator channel")?;
    let mut orchestrator_client = OrchestratorClient::create(&channel);
    let endorsed_evidence = RUNTIME
        .block_on(orchestrator_client.get_endorsed_evidence())
        .context("failed to get endorsed evidence")?;
    let evidence = endorsed_evidence.evidence.context("EndorsedEvidence.evidence not set")?;
    let endorsements =
        endorsed_evidence.endorsements.context("EndorsedEvidence.endorsements not set")?;

    // If platform endorsement is missing (Oak Launcher doesn't populate it yet),
    // fall back to fake attestation that matches computation_runner.cc's fake mode.
    // Uses the legacy APIs (add_self_attester/endorser/session_binder) to match
    // the C++ client's legacy AddPeerVerifier.
    if endorsements.platform.is_none() {
        println!("Platform endorsement missing, falling back to fake attestation");
        let builder =
            SessionConfig::builder(AttestationType::SelfUnidirectional, HandshakeType::NoiseNN)
                .add_self_attester(String::from(FAKE_ATTESTER_ID), Box::new(FakeAttester))
                .add_self_endorser(String::from(FAKE_ATTESTER_ID), Box::new(FakeEndorser))
                .add_session_binder(String::from(FAKE_ATTESTER_ID), Box::new(FakeSessionBinder));
        return Ok(Box::into_raw(Box::new(builder.build())));
    }

    let attester: Arc<dyn Attester> = Arc::new(StaticAttester::new(evidence.clone()));
    let endorser: Arc<dyn Endorser> = Arc::new(StaticEndorser::new(endorsements.clone()));
    let session_binder: Arc<dyn SessionBinder> = Arc::new(InstanceSessionBinder::create(&channel));

    let builder =
        SessionConfig::builder(AttestationType::SelfUnidirectional, HandshakeType::NoiseNN)
            .add_self_assertion_generator(
                String::from(ASSERTION_ID),
                Box::new(EndorsedEvidenceBindableAssertionGenerator::new(
                    attester,
                    endorser,
                    session_binder,
                )),
            );
    Ok(Box::into_raw(Box::new(builder.build())))
}

/// Creates a SessionConfig for the program worker.
///
/// This function is exported to C so that it can be called from the program
/// worker binary.
///
/// # Arguments
///
/// * `endorsed_evidence_bytes`: A serialized EndorsedEvidence proto.
/// * `endorsed_evidence_len`: The length of the serialized EndorsedEvidence
///   proto.
///
/// # Returns
///
/// A raw pointer to a SessionConfig. The caller is responsible for managing the
/// memory of this object.
///
/// # Safety
///
/// The returned config is an opaque raw pointer to a `SessionConfig` object.
/// The handle is intended to be reclaimed by passing it the FFI factory of an
/// OakSession object like `oak_session::ffi::new_client_session`.
///
/// `endorsed_evidence_bytes` and `endorsed_evidence_len` must describe a valid
/// buffer. Data must not be modified during this function call. It may be
/// modified or discarded after, as this function will make its own copy.
#[no_mangle]
pub unsafe extern "C" fn create_session_config() -> *mut SessionConfig {
    match create_session_config_internal() {
        Ok(config) => {
            println!("Session config created successfully");
            config
        }
        Err(err) => {
            eprintln!("Error creating session config: {:?}", err);
            std::ptr::null_mut()
        }
    }
}
