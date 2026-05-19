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
use std::sync::Arc;
 
use anyhow::Context;

use oak_attestation_types::{attester::Attester, endorser::Endorser};
use oak_proto_rust::oak::attestation::v1::{Evidence, ReferenceValues, TeePlatform};
use oak_proto_rust::oak::session::v1::PlaintextMessage;
use oak_sdk_common::{StaticAttester, StaticEndorser};
use oak_sdk_containers::{InstanceSessionBinder, OrchestratorClient};
use oak_session::{session_binding::SessionBinder, ProtocolEngine, ServerSession, Session};
use oak_time::Clock;
use prost::Message;
use session_config::create_session_config;
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
use tracing::{debug, info, warn};

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
        eprintln!("StorageProxy: Received new gRPC stream request");
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
                eprintln!("StorageProxy: Session created successfully");
                s
            }
            Err(e) => {
                eprintln!("StorageProxy: FAILED to create session: {:?}", e);
                return Err(tonic::Status::internal(format!("failed to create session: {:?}", e)));
            }
        };

        let mut in_stream = request.into_inner();
        let (tx, rx) = mpsc::channel(128);
        let storage = self.storage.clone();
        let clock = self.clock.clone();

        tokio::spawn(async move {
            while let Some(msg) = in_stream.next().await {
                let msg = match msg {
                    Ok(m) => m,
                    Err(e) => {
                        debug!("Stream error: {:?}", e);
                        break;
                    }
                };

                let session_req = match oak_proto_rust::oak::session::v1::SessionRequest::decode(
                    msg.encode_to_vec().as_slice(),
                ) {
                    Ok(r) => r,
                    Err(e) => {
                        warn!("Failed to decode SessionRequest: {:?}", e);
                        break;
                    }
                };

                if let Err(e) = session.put_incoming_message(session_req) {
                    warn!("Failed to put incoming message: {:?}", e);
                    break;
                }
                
                if session.is_open() {
                    while let Ok(Some(msg)) = session.read() {
                        let request = match StorageRequest::decode(msg.plaintext.as_slice()) {
                            Ok(r) => r,
                            Err(e) => {
                                warn!("Failed to decode StorageRequest: {:?}", e);
                                break;
                            }
                        };

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
                            Err(e) => StorageResponse {
                                correlation_id: request.correlation_id,
                                kind: Some(storage_response::Kind::Error(
                                    storage_proto::status_proto::google::rpc::Status {
                                        code: tonic::Code::Internal as i32,
                                        message: format!("{:?}", e),
                                        ..Default::default()
                                    },
                                )),
                            },
                        };

                        if let Err(e) = session.write(PlaintextMessage {
                            plaintext: response.encode_to_vec(),
                        }) {
                            warn!("Failed to write to session: {:?}", e);
                            break;
                        }
                    }
                }

                while let Ok(Some(response_msg)) = session.get_outgoing_message() {
                    match SessionResponse::decode(response_msg.encode_to_vec().as_slice()) {
                        Ok(response) => {
                            if let Err(e) = tx.send(Ok(response)).await {
                                warn!("Failed to send SessionResponse: {:?}", e);
                                break;
                            }
                        }
                        Err(e) => {
                            warn!("Failed to decode SessionResponse: {:?}", e);
                            break;
                        }
                    }
                }
            }
        });

        Ok(tonic::Response::new(Box::pin(ReceiverStream::new(rx))))
    }
}

fn get_reference_values(evidence: &Evidence) -> anyhow::Result<ReferenceValues> {
    match evidence.root_layer.as_ref().map(|rl| rl.platform.try_into()) {
        Some(Ok(TeePlatform::AmdSevSnp)) => {
            // Production: load actual reference values
            // ReferenceValues::decode(include_bytes!(env!("REFERENCE_VALUES")).as_slice())
            //    .context("failed to decode ReferenceValues")
            ReferenceValues::decode(
                include_bytes!(env!("INSECURE_REFERENCE_VALUES")).as_slice(),
            )
            .context("failed to decode ReferenceValues")
        }
        Some(Ok(TeePlatform::None)) => {
            ReferenceValues::decode(
                include_bytes!(env!("INSECURE_REFERENCE_VALUES")).as_slice(),
            )
            .context("failed to decode insecure ReferenceValues")
        }
        platform => anyhow::bail!("platform {:?} is not supported", platform),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    info!("Storage Proxy starting...");

    let args: Vec<String> = std::env::args().collect();
    let port: u16 = if args.len() > 1 && args[1] == "--port" && args.len() > 2 {
        args[2].parse().context("failed to parse port")?
    } else {
        8008
    };

    let addr_str = format!("0.0.0.0:{}", port);
    let addr = addr_str.parse()?;
    
    // 1. Establish channel to the Orchestrator
    let channel = oak_sdk_containers::default_orchestrator_channel()
        .await
        .context("failed to create orchestrator channel")?;
    let mut orchestrator_client = OrchestratorClient::create(&channel);

    // 2. Fetch Evidence and Endorsements
    let endorsed_evidence = orchestrator_client
        .get_endorsed_evidence()
        .await
        .context("failed to get endorsed evidence")?;
    let evidence = endorsed_evidence.evidence.as_ref().context("EndorsedEvidence.evidence not set")?;
    let endorsements = endorsed_evidence.endorsements.as_ref().context("EndorsedEvidence.endorsements not set")?;

    // 3. Initialize Attestation Components (Matching KMS pattern)
    let attester = Arc::new(StaticAttester::new(evidence.clone()));
    let endorser = Arc::new(StaticEndorser::new(endorsements.clone()));
    let session_binder = Arc::new(InstanceSessionBinder::create(&channel));
    let reference_values = get_reference_values(evidence).context("failed to get reference values")?;
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

    info!("Starting Storage Proxy gRPC server on {}", addr);

    // 4. Notify Orchestrator that app is ready
    orchestrator_client.notify_app_ready().await.context("failed to notify that app is ready")?;

    Server::builder()
        .max_frame_size(1024 * 1024) // 1MB
        .add_service(OakSessionV1ServiceServer::new(proxy).max_encoding_message_size(10 * 1024 * 1024))
        .serve(addr)
        .await?;

    Ok(())
}
