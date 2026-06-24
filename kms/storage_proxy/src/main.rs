// Copyright 2026 Google LLC.
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

use std::pin::Pin;
use std::sync::{Arc, Mutex, OnceLock};
use std::io::Write;

use anyhow::Context;

use oak_attestation_types::{attester::Attester, endorser::Endorser};
use oak_proto_rust::oak::attestation::v1::ReferenceValues;
use oak_proto_rust::oak::session::v1::PlaintextMessage;
use oak_sdk_containers::OrchestratorClient;
use oak_session::{session_binding::SessionBinder, ProtocolEngine, ServerSession, Session};
use oak_time::Clock;
use prost::Message;
use prost_proto_conversion::ProstProtoConversionExt;
use session_config::create_session_config;
use session_test_utils::{
    get_test_attester, get_test_endorser, get_test_reference_values, get_test_session_binder,
};
use session_v1_service_proto::oak::services::oak_session_v1_service_server::{
    OakSessionV1Service, OakSessionV1ServiceServer,
};
use session_v1_service_proto::session_proto::oak::session::v1::{SessionRequest, SessionResponse};
use storage::Storage;
use storage_proto::confidential_federated_compute::kms::{
    storage_request, storage_response, StorageRequest, StorageResponse,
};
use tokio::sync::mpsc;
use tokio_stream::{wrappers::ReceiverStream, Stream, StreamExt};
use tonic::transport::Server;
use tracing::info;

/// Diagnostic sink. The storage_proxy runs inside an Oak Containers VM where:
///   * the container's stderr is captured by oak-syslogd, which only forwards
///     via OTLP and that endpoint returns UNIMPLEMENTED in this environment;
///   * /dev/kmsg, /dev/console, /dev/ttyS0 are not present in the distroless
///     container's /dev.
/// The remaining path that works without rebuilding the OCI bundle is outbound
/// TCP from the guest. SLIRP routes the VM's traffic to the host as 10.0.2.2,
/// so we try to open a TCP connection to that host on a fixed port and use it
/// as the diagnostic sink. To collect logs on the host, run before launching:
///
///     nc -lk 0.0.0.0 6655 | tee storage_proxy.log
///
/// (or any tool that prints whatever arrives). If the TCP attempt fails we
/// still try the device-node fallbacks, and finally stderr.
const DIAG_HOST: &str = "10.0.2.2";
const DIAG_PORT: u16 = 6655;

enum DiagSink {
    Tcp(std::net::TcpStream),
    File(std::fs::File),
    None,
}

static DIAG_OUT: OnceLock<Mutex<DiagSink>> = OnceLock::new();

fn diag_writer() -> &'static Mutex<DiagSink> {
    DIAG_OUT.get_or_init(|| {
        // 1) outbound TCP to host:6655 (works whenever the user starts
        //    `nc -lk 6655` before launching).
        match std::net::TcpStream::connect_timeout(
            &format!("{DIAG_HOST}:{DIAG_PORT}").parse().unwrap(),
            std::time::Duration::from_secs(2),
        ) {
            Ok(stream) => {
                let _ = stream.set_nodelay(true);
                eprintln!("StorageProxy: diag logger opened TCP {DIAG_HOST}:{DIAG_PORT}");
                return Mutex::new(DiagSink::Tcp(stream));
            }
            Err(e) => {
                eprintln!(
                    "StorageProxy: diag TCP {DIAG_HOST}:{DIAG_PORT} unavailable ({e}); trying device fallbacks"
                );
            }
        }
        // 2) kernel ring buffer / serial console fallbacks (often missing).
        for path in ["/dev/kmsg", "/dev/console", "/dev/ttyS0"] {
            if let Ok(f) = std::fs::OpenOptions::new().write(true).open(path) {
                eprintln!("StorageProxy: diag logger opened {path}");
                return Mutex::new(DiagSink::File(f));
            }
        }
        eprintln!(
            "StorageProxy: WARNING no diag sink available (TCP {DIAG_HOST}:{DIAG_PORT}, \
             /dev/kmsg, /dev/console, /dev/ttyS0 all failed). Diagnostics will only \
             appear in stderr (likely lost to oak-syslogd/OTLP)."
        );
        Mutex::new(DiagSink::None)
    })
}

/// Emit a diagnostic line that survives even when oak-syslogd/OTLP is broken.
/// Always also writes to stderr; on top of that, sends the line to whichever
/// sink `diag_writer()` was able to open.
macro_rules! diag {
    ($($arg:tt)*) => {{
        let line = format!($($arg)*);
        eprintln!("{line}");
        if let Ok(mut guard) = diag_writer().lock() {
            match &mut *guard {
                DiagSink::Tcp(s) => {
                    let _ = writeln!(s, "storage_proxy: {line}");
                    let _ = s.flush();
                }
                DiagSink::File(f) => {
                    let _ = writeln!(f, "storage_proxy: {line}");
                    let _ = f.flush();
                }
                DiagSink::None => {}
            }
        }
    }};
}

struct StorageProxy {
    storage: Arc<tokio::sync::Mutex<Storage>>,
    attester: Arc<dyn Attester>,
    endorser: Arc<dyn Endorser>,
    session_binder: Arc<dyn SessionBinder>,
    reference_values: ReferenceValues,
    clock: Arc<dyn Clock>,
}

#[tonic::async_trait]
impl OakSessionV1Service for StorageProxy {
    type StreamStream = Pin<Box<dyn Stream<Item = Result<SessionResponse, tonic::Status>> + Send>>;

    async fn stream(
        &self,
        request: tonic::Request<tonic::Streaming<SessionRequest>>,
    ) -> Result<tonic::Response<Self::StreamStream>, tonic::Status> {
        diag!("StorageProxy: Received new gRPC stream request");
        info!("Received new gRPC stream request");
        let session_result = create_session_config(
            &self.attester,
            &self.endorser,
            &self.session_binder,
            &self.reference_values,
            self.clock.clone(),
        )
        .and_then(ServerSession::create);
        let mut session = match session_result {
            Ok(s) => {
                diag!("StorageProxy: Session created successfully");
                s
            }
            Err(e) => {
                diag!("StorageProxy: FAILED to create session: {:?}", e);
                return Err(tonic::Status::internal(format!("failed to create session: {:?}", e)));
            }
        };

        let mut in_stream = request.into_inner();
        let (tx, rx) = mpsc::channel(128);
        let storage = self.storage.clone();
        let clock = self.clock.clone();

        tokio::spawn(async move {
            // Diagnostics: log every reason the spawn task can exit. The
            // previous code used `break` without a visible log, so the tx
            // channel would drop silently and the gRPC stream would close
            // with trailers-only — looks identical at the wire level to
            // create_session_config() failing.
            let mut incoming_seq: u64 = 0;
            let mut session_open_logged = false;
            let exit_reason: &'static str = loop {
                let msg = match in_stream.next().await {
                    Some(Ok(m)) => m,
                    Some(Err(e)) => {
                        diag!("StorageProxy: gRPC stream errored: {e:?}");
                        break "grpc stream error";
                    }
                    None => break "client closed stream",
                };
                incoming_seq += 1;
                let raw_len = msg.encode_to_vec().len();
                diag!(
                    "StorageProxy: incoming message #{incoming_seq} ({raw_len} bytes), session.is_open={}",
                    session.is_open()
                );

                let session_req = match oak_proto_rust::oak::session::v1::SessionRequest::decode(
                    msg.encode_to_vec().as_slice(),
                ) {
                    Ok(r) => r,
                    Err(e) => {
                        diag!("StorageProxy: failed to decode SessionRequest: {e:?}");
                        break "session request decode failure";
                    }
                };

                if let Err(e) = session.put_incoming_message(session_req) {
                    diag!("StorageProxy: put_incoming_message failed: {e:?}");
                    break "put_incoming_message failed";
                }
                if session.is_open() && !session_open_logged {
                    diag!(
                        "StorageProxy: session.is_open() became true after incoming #{incoming_seq}"
                    );
                    session_open_logged = true;
                }

                if session.is_open() {
                    loop {
                        let plaintext = match session.read() {
                            Ok(Some(m)) => m,
                            Ok(None) => break,
                            Err(e) => {
                                diag!("StorageProxy: session.read() errored: {e:?}");
                                break;
                            }
                        };
                        diag!(
                            "StorageProxy: decrypted storage payload ({} bytes)",
                            plaintext.plaintext.len()
                        );

                        let request = match StorageRequest::decode(plaintext.plaintext.as_slice()) {
                            Ok(r) => r,
                            Err(e) => {
                                diag!("StorageProxy: failed to decode StorageRequest: {e:?}");
                                break;
                            }
                        };
                        let kind_label = match &request.kind {
                            Some(storage_request::Kind::Read(_)) => "Read",
                            Some(storage_request::Kind::Update(_)) => "Update",
                            None => "<missing>",
                        };
                        diag!(
                            "StorageProxy: StorageRequest corr_id={} kind={kind_label}",
                            request.correlation_id
                        );

                        let mut storage_lock = storage.lock().await;
                        let response_kind = match request.kind {
                            Some(storage_request::Kind::Read(read_req)) => {
                                storage_lock.read(&read_req).map(storage_response::Kind::Read)
                            }
                            Some(storage_request::Kind::Update(update_req)) => {
                                let now = storage_proto::timestamp_proto::google::protobuf::Timestamp {
                                    seconds: clock.get_time().into_timestamp().seconds,
                                    nanos: 0,
                                };
                                storage_lock
                                    .update(&now, update_req)
                                    .map(storage_response::Kind::Update)
                            }
                            None => Err(anyhow::anyhow!("missing request kind")),
                        };

                        let response = match response_kind {
                            Ok(kind) => StorageResponse {
                                correlation_id: request.correlation_id,
                                kind: Some(kind),
                            },
                            Err(e) => {
                                // Preserve the gRPC Code attached to the error
                                // (e.g. FailedPrecondition for unsatisfied
                                // preconditions). The KMS storage_client and
                                // the KMS's own retry logic in rotate_keyset
                                // depend on this code to distinguish e.g.
                                // "already initialized" / "key collision"
                                // from real internal failures. Mirrors
                                // storage_actor::convert_error.
                                let code = e
                                    .downcast_ref::<tonic::Code>()
                                    .copied()
                                    .unwrap_or(tonic::Code::Internal);
                                StorageResponse {
                                    correlation_id: request.correlation_id,
                                    kind: Some(storage_response::Kind::Error(
                                        storage_proto::status_proto::google::rpc::Status {
                                            code: code as i32,
                                            message: format!("{e:#}"),
                                            ..Default::default()
                                        },
                                    )),
                                }
                            }
                        };

                        let response_kind_label = match &response.kind {
                            Some(storage_response::Kind::Read(_)) => "Read".to_string(),
                            Some(storage_response::Kind::Update(_)) => "Update".to_string(),
                            Some(storage_response::Kind::Error(s)) => {
                                format!("Error(code={}, msg={})", s.code, s.message)
                            }
                            None => "<missing>".to_string(),
                        };
                        diag!(
                            "StorageProxy: prepared StorageResponse corr_id={} kind={response_kind_label}",
                            response.correlation_id
                        );

                        if let Err(e) = session.write(PlaintextMessage {
                            plaintext: response.encode_to_vec(),
                        }) {
                            diag!("StorageProxy: session.write() errored: {e:?}");
                            break;
                        }
                    }
                }

                let mut outgoing_seq: u64 = 0;
                loop {
                    let response_msg = match session.get_outgoing_message() {
                        Ok(Some(m)) => m,
                        Ok(None) => {
                            diag!(
                                "StorageProxy: no more outgoing messages after incoming #{incoming_seq} (sent {outgoing_seq})"
                            );
                            break;
                        }
                        Err(e) => {
                            diag!(
                                "StorageProxy: session.get_outgoing_message() errored: {e:?}"
                            );
                            break;
                        }
                    };
                    outgoing_seq += 1;
                    let response = match SessionResponse::decode(
                        response_msg.encode_to_vec().as_slice(),
                    ) {
                        Ok(r) => r,
                        Err(e) => {
                            diag!("StorageProxy: failed to re-decode SessionResponse: {e:?}");
                            break;
                        }
                    };
                    let encoded_len = response.encode_to_vec().len();
                    if let Err(e) = tx.send(Ok(response)).await {
                        // The receiver was dropped: client gave up. Not fatal.
                        diag!("StorageProxy: outbound mpsc send failed (client gone): {e:?}");
                        break;
                    }
                    diag!(
                        "StorageProxy: sent outgoing message #{outgoing_seq} ({encoded_len} bytes) for incoming #{incoming_seq}"
                    );
                }
            };
            diag!("StorageProxy: stream task exiting: {exit_reason}");
        });

        Ok(tonic::Response::new(Box::pin(ReceiverStream::new(rx))))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    // First diag! call also triggers OnceLock init of the writer, which logs
    // which device (/dev/kmsg vs /dev/console vs /dev/ttyS0) ended up open.
    diag!("StorageProxy: main() starting (pid={})", std::process::id());
    info!("Storage Proxy starting...");

    let args: Vec<String> = std::env::args().collect();
    let port: u16 = if args.len() > 1 && args[1] == "--port" && args.len() > 2 {
        args[2].parse().context("failed to parse port")?
    } else {
        8008
    };

    let addr_str = format!("0.0.0.0:{}", port);
    let addr = addr_str.parse()?;

    // We still need the orchestrator channel for `notify_app_ready` so the
    // launcher (and any waiter on `get_trusted_app_address`) knows we are
    // serving. We deliberately do NOT use the orchestrator's attestation
    // components for the session below — see comment above the proxy
    // initialisation.
    let channel = oak_sdk_containers::default_orchestrator_channel()
        .await
        .context("failed to create orchestrator channel")?;
    let mut orchestrator_client = OrchestratorClient::create(&channel);

    // Use the `session_test_utils` (oak_sdk_standalone-based) attestation
    // path for the session. The orchestrator-provided alternative
    // (StaticAttester::new(evidence) + StaticEndorser::new(endorsements) +
    // InstanceSessionBinder::create) fails handshake with
    //   "verification failed: no platform endorsement"
    // because the launcher's `get_endorsements()` returns
    // `OakContainersEndorsements { root_layer: None, kernel_layer: None, ... }`
    // and `EndorsedEvidenceBoundAssertionVerifier` cannot extract the
    // session-binding key without a populated root-layer endorsement.
    // `Standalone` produces evidence + endorsements + signing key that are
    // self-consistent for the insecure root layer used here.
    //
    // IMPORTANT: the KMS side (kms/main.rs) must apply the same change —
    // i.e. build its `KeyManagementService::new(GrpcStorageClient::new(..))`
    // with `get_test_attester() / get_test_endorser() /
    // get_test_session_binder()` and pass `get_test_reference_values()` —
    // otherwise the asymmetry will produce the same handshake failure in
    // the opposite direction.
    let attester: Arc<dyn Attester> = get_test_attester();
    let endorser: Arc<dyn Endorser> = get_test_endorser();
    let session_binder: Arc<dyn SessionBinder> = get_test_session_binder();
    let reference_values: ReferenceValues =
        get_test_reference_values().convert().unwrap();
    let clock = Arc::new(oak_time_std::clock::SystemTimeClock {});

    let storage = Arc::new(tokio::sync::Mutex::new(Storage::default()));
    let proxy = StorageProxy {
        storage,
        attester,
        endorser,
        session_binder,
        reference_values,
        clock,
    };

    diag!("StorageProxy: bound proxy state, about to notify_app_ready and serve on {addr}");
    info!("Starting Storage Proxy gRPC server on {}", addr);

    // 4. Notify Orchestrator that app is ready
    orchestrator_client.notify_app_ready().await.context("failed to notify that app is ready")?;
    diag!("StorageProxy: notify_app_ready returned, calling Server::serve");

    Server::builder()
        .max_frame_size(1024 * 1024) // 1MB
        .add_service(
            OakSessionV1ServiceServer::new(proxy)
                .max_encoding_message_size(10 * 1024 * 1024)
                // Without this, tonic defaults to 4 MiB for the decoded size of
                // each incoming SessionRequest. Bidirectional Oak Containers
                // attestation can carry endorsements that bump a single
                // handshake message past that cap; when it does, tonic rejects
                // the frame and the stream closes with a trailers-only error
                // before the spawn task ever sees the message. Matches the
                // 10 MiB cap the KMS-side client sets in kms/main.rs.
                .max_decoding_message_size(10 * 1024 * 1024),
        )
        .serve(addr)
        .await?;

    Ok(())
}

