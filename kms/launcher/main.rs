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

//! Untrusted Launcher for the CFC KMS.
//!
//! This binary hosts a TCP RAFT node with a StorageActor in-process and
//! provides an OakSessionV1Service on port 8008 for the KMS to connect to.

use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use oak_attestation_types::{attester::Attester, endorser::Endorser};
use oak_sdk_common::{StaticAttester, StaticEndorser};
use oak_sdk_standalone::Standalone;
use oak_session::session_binding::{SessionBinder, SignatureBinder};
use oak_time::{Clock, Instant};
use p256::ecdsa::{SigningKey, VerifyingKey};
use prost::Message;
use prost_proto_conversion::ProstProtoConversionExt;
use rand_core::OsRng;

// Use the "ppl" (prost proto library) versions to match StorageActor.
use access_policy_proto::reference_value_proto::oak::attestation::v1::ReferenceValues;
use session_v1_service_proto::{
    oak::services::oak_session_v1_service_server::{OakSessionV1Service, OakSessionV1ServiceServer},
    session_proto::oak::session::v1::{
        SessionRequest as GrpcSessionRequest, SessionResponse as GrpcSessionResponse,
    },
};
use tcp_proto::runtime::endpoint::*;
use tcp_runtime::service::ApplicationService;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::{wrappers::ReceiverStream, Stream, StreamExt};
use tonic::transport::Server;

// Re-export the KMS proto for SessionResponseWithStatus.
use kms_proto::fcp::confidentialcompute::SessionResponseWithStatus;

/// Port on which the OakSessionV1Service will listen.
const SERVICE_PORT: u16 = 8008;

/// Tick interval in milliseconds for the RAFT driver.
const TICK_INTERVAL_MS: u64 = 100;

// ---------------------------------------------------------------------------
// Insecure attestation stubs
// ---------------------------------------------------------------------------

struct InsecureCredentials {
    standalone: Standalone,
    session_binding_key: SigningKey,
}

impl InsecureCredentials {
    fn new() -> Self {
        let signing_key = SigningKey::random(&mut OsRng);
        let session_binding_key = SigningKey::random(&mut OsRng);
        let standalone = Standalone::builder()
            .signing_key_pair(Some((signing_key.clone(), VerifyingKey::from(&signing_key))))
            .session_binding_key_pair(Some((
                session_binding_key.clone(),
                VerifyingKey::from(&session_binding_key),
            )))
            .build()
            .expect("failed to build Standalone");
        InsecureCredentials { standalone, session_binding_key }
    }

    fn attester(&self) -> Arc<dyn Attester> {
        Arc::new(StaticAttester::new(
            self.standalone.endorsed_evidence().evidence.unwrap(),
        ))
    }

    fn endorser(&self) -> Arc<dyn Endorser> {
        Arc::new(StaticEndorser::new(
            self.standalone.endorsed_evidence().endorsements.unwrap(),
        ))
    }

    fn session_binder(&self) -> Arc<dyn SessionBinder> {
        Arc::new(SignatureBinder::new(Box::new(self.session_binding_key.clone())))
    }

    fn reference_values(&self) -> ReferenceValues {
        use access_policy_proto::reference_value_proto::oak::attestation::v1::{
            binary_reference_value, kernel_binary_reference_value, reference_values,
            text_reference_value, BinaryReferenceValue, ContainerLayerReferenceValues,
            InsecureReferenceValues, KernelBinaryReferenceValue, KernelLayerReferenceValues,
            OakContainersReferenceValues, RootLayerReferenceValues, SkipVerification,
            SystemLayerReferenceValues, TextReferenceValue,
        };

        let skip = BinaryReferenceValue {
            r#type: Some(binary_reference_value::Type::Skip(SkipVerification::default())),
        };
        ReferenceValues {
            r#type: Some(reference_values::Type::OakContainers(
                OakContainersReferenceValues {
                    root_layer: Some(RootLayerReferenceValues {
                        insecure: Some(InsecureReferenceValues::default()),
                        ..Default::default()
                    }),
                    kernel_layer: Some(KernelLayerReferenceValues {
                        kernel: Some(KernelBinaryReferenceValue {
                            r#type: Some(kernel_binary_reference_value::Type::Skip(
                                SkipVerification::default(),
                            )),
                        }),
                        kernel_cmd_line_text: Some(TextReferenceValue {
                            r#type: Some(text_reference_value::Type::Skip(
                                SkipVerification::default(),
                            )),
                        }),
                        init_ram_fs: Some(skip.clone()),
                        memory_map: Some(skip.clone()),
                        acpi: Some(skip.clone()),
                    }),
                    system_layer: Some(SystemLayerReferenceValues {
                        system_image: Some(skip.clone()),
                    }),
                    container_layer: Some(ContainerLayerReferenceValues {
                        binary: Some(skip.clone()),
                        configuration: Some(skip.clone()),
                    }),
                },
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Driver communication
// ---------------------------------------------------------------------------

type DriverRequest = (
    ReceiveMessageRequest,
    oneshot::Sender<ReceiveMessageResponse>,
);

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64
}

fn run_driver_loop(
    actor: storage_actor::StorageActor,
    mut rx: mpsc::Receiver<DriverRequest>,
) {
    let mut service = ApplicationService::new(actor);
    while let Some((request, response_tx)) = rx.blocking_recv() {
        if let Ok(response) = service.receive_message(request) {
            let _ = response_tx.send(response);
        }
    }
}

// ---------------------------------------------------------------------------
// Session Service
// ---------------------------------------------------------------------------

struct LauncherSessionService {
    driver_sender: mpsc::Sender<DriverRequest>,
    next_correlation_id: AtomicU64,
    next_session_id: AtomicU64,
}

impl LauncherSessionService {
    fn new(driver_sender: mpsc::Sender<DriverRequest>) -> Self {
        Self {
            driver_sender,
            next_correlation_id: AtomicU64::new(1),
            next_session_id: AtomicU64::new(1),
        }
    }
}

#[tonic::async_trait]
impl OakSessionV1Service for LauncherSessionService {
    type StreamStream =
        Pin<Box<dyn Stream<Item = Result<GrpcSessionResponse, tonic::Status>> + Send>>;

    async fn stream(
        &self,
        request: tonic::Request<tonic::Streaming<GrpcSessionRequest>>,
    ) -> Result<tonic::Response<Self::StreamStream>, tonic::Status> {
        let mut in_stream = request.into_inner();
        let (response_tx, response_rx) = mpsc::channel(128);
        let driver_sender = self.driver_sender.clone();
        let session_id = self.next_session_id.fetch_add(1, Ordering::Relaxed).to_le_bytes().to_vec();
        let correlation_source = Arc::new(AtomicU64::new(
            self.next_correlation_id.fetch_add(1000, Ordering::Relaxed),
        ));

        tokio::spawn(async move {
            while let Some(msg) = in_stream.next().await {
                let grpc_request = match msg {
                    Ok(r) => r,
                    Err(_) => break,
                };

                // Use re-encoding to convert from gRPC proto to PPL proto.
                let ppl_request = oak_proto_rust::oak::session::v1::SessionRequest::decode(
                    grpc_request.encode_to_vec().as_slice(),
                ).ok();

                let wrapped = oak_proto_rust::oak::session::v1::SessionRequestWithSessionId {
                    session_id: session_id.clone(),
                    request: ppl_request,
                };

                let correlation_id = correlation_source.fetch_add(1, Ordering::Relaxed);
                let in_msg = InMessage {
                    msg: Some(in_message::Msg::DeliverAppMessage(DeliverAppMessage {
                        correlation_id,
                        message_header: wrapped.encode_to_vec().into(),
                        message_payload: prost::bytes::Bytes::new(),
                    })),
                };

                let (tx, rx) = oneshot::channel();
                if driver_sender.send((
                    ReceiveMessageRequest { instant: now_ms(), message: Some(in_msg) },
                    tx,
                )).await.is_err() { break; }

                if let Ok(driver_response) = rx.await {
                    for out_msg in driver_response.messages {
                        if let Some(out_message::Msg::DeliverAppMessage(app_msg)) = out_msg.msg {
                            if app_msg.correlation_id == correlation_id {
                                if let Ok(status_response) = SessionResponseWithStatus::decode(app_msg.message_header) {
                                    if let Some(kms_session_response) = status_response.response {
                                        if let Ok(grpc_response) = GrpcSessionResponse::decode(
                                            kms_session_response.encode_to_vec().as_slice()
                                        ) {
                                            let _ = response_tx.send(Ok(grpc_response)).await;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });

        Ok(tonic::Response::new(Box::pin(ReceiverStream::new(response_rx))))
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let creds = InsecureCredentials::new();
    
    struct WallClock;
    impl Clock for WallClock {
        fn get_time(&self) -> Instant {
            Instant::from_unix_millis(SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as i64)
        }
    }

    let actor = storage_actor::StorageActor::new(
        creds.attester(),
        creds.endorser(),
        creds.session_binder(),
        creds.reference_values(),
        Arc::new(WallClock) as Arc<dyn Clock>,
    );

    let (driver_tx, driver_rx) = mpsc::channel(32);
    tokio::task::spawn_blocking(move || run_driver_loop(actor, driver_rx));

    let (init_tx, init_rx) = oneshot::channel();
    driver_tx.send((
        ReceiveMessageRequest {
            instant: now_ms(),
            message: Some(InMessage {
                msg: Some(in_message::Msg::StartReplica(StartReplicaRequest {
                    replica_id_hint: 1,
                    is_leader: true,
                    is_ephemeral: false,
                    raft_config: Some(RaftConfig {
                        tick_period: TICK_INTERVAL_MS,
                        election_tick: 10,
                        heartbeat_tick: 3,
                        max_size_per_msg: 0,
                        snapshot_config: Some(raft_config::SnapshotConfig {
                            snapshot_count: 1000,
                            ..Default::default()
                        }),
                        ..Default::default()
                    }),
                    app_config: prost::bytes::Bytes::new(),
                    endorsements: None,
                })),
            }),
        },
        init_tx,
    )).await?;
    let _ = init_rx.await?;

    let tick_sender = driver_tx.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_millis(TICK_INTERVAL_MS));
        loop {
            interval.tick().await;
            let (tx, _) = oneshot::channel();
            if tick_sender.send((ReceiveMessageRequest { instant: now_ms(), message: None }, tx)).await.is_err() { break; }
        }
    });

    let addr = "[::]:8008".parse()?;
    Server::builder().add_service(OakSessionV1ServiceServer::new(LauncherSessionService::new(driver_tx))).serve(addr).await?;
    Ok(())
}