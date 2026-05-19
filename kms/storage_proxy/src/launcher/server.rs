//
// Copyright 2023 The Project Oak Authors
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

use std::{
    pin::Pin,
    sync::{Arc, Mutex},
};

use anyhow::anyhow;
use bytes::BytesMut;
use futures::{FutureExt, Stream};
use oak_grpc::oak::containers::{
    launcher_server::{Launcher, LauncherServer},
    v1::hostlib_key_provisioning_server::{HostlibKeyProvisioning, HostlibKeyProvisioningServer},
};
use oak_proto_rust::oak::{
    attestation::v1::{Endorsements, Evidence},
    containers::{
        v1::{GetGroupKeysResponse, GetKeyProvisioningRoleResponse, KeyProvisioningRole},
        GetApplicationConfigResponse, GetImageResponse, SendAttestationEvidenceRequest,
    },
};
use tokio::{
    io::{AsyncReadExt, BufReader},
    net::TcpListener,
    sync::{oneshot, watch},
};
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{transport::Server, Request, Response, Status};

// Most gRPC implementations limit message sizes to 4MiB. Let's stay
// comfortably below that by limiting responses to 3MiB.
const MAX_RESPONSE_SIZE: usize = 3 * 1024 * 1024;

type GetImageResponseStream = Pin<Box<dyn Stream<Item = Result<GetImageResponse, Status>> + Send>>;

#[derive(Default)]
struct LauncherServerImplementation {
    system_image: std::path::PathBuf,
    container_bundle: std::path::PathBuf,
    application_config: Vec<u8>,
    evidence_sender: Mutex<Option<oneshot::Sender<Evidence>>>,
    app_ready_notifier: Mutex<Option<oneshot::Sender<()>>>,
    endorsements: Endorsements,
}

// Manual implementation of async stream using futures::stream::unfold
fn async_stream_generator(reader: BufReader<tokio::fs::File>) -> GetImageResponseStream {
    let stream = futures::stream::unfold(reader, |mut reader| async move {
        let mut buffer = BytesMut::with_capacity(MAX_RESPONSE_SIZE);
        match reader.read_buf(&mut buffer).await {
            Ok(0) => None,
            Ok(_) => {
                let response = GetImageResponse {
                    image_chunk: buffer.freeze()
                };
                Some((Result::<GetImageResponse, Status>::Ok(response), reader))
            }
            Err(e) => Some((Err(Status::internal(e.to_string())), reader)),
        }
    });
    Box::pin(stream)
}

#[tonic::async_trait]
impl Launcher for LauncherServerImplementation {
    type GetOakSystemImageStream = GetImageResponseStream;
    type GetContainerBundleStream = GetImageResponseStream;

    async fn get_oak_system_image(
        &self,
        _request: Request<()>,
    ) -> Result<Response<Self::GetOakSystemImageStream>, tonic::Status> {
        let system_image_file = tokio::fs::File::open(&self.system_image)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
        let reader = BufReader::new(system_image_file);
        let response_stream = async_stream_generator(reader);
        Ok(Response::new(response_stream))
    }

    async fn get_container_bundle(
        &self,
        _request: Request<()>,
    ) -> Result<Response<Self::GetContainerBundleStream>, tonic::Status> {
        let container_bundle_file = tokio::fs::File::open(&self.container_bundle)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
        let reader = BufReader::new(container_bundle_file);
        let response_stream = async_stream_generator(reader);
        Ok(Response::new(response_stream))
    }

    async fn get_application_config(
        &self,
        _request: Request<()>,
    ) -> Result<Response<GetApplicationConfigResponse>, tonic::Status> {
        Ok(tonic::Response::new(GetApplicationConfigResponse {
            config: self.application_config.clone(),
        }))
    }

    async fn send_attestation_evidence(
        &self,
        request: Request<SendAttestationEvidenceRequest>,
    ) -> Result<Response<()>, tonic::Status> {
        let request = request.into_inner();
        let evidence = request.dice_evidence.ok_or_else(|| {
            tonic::Status::internal("send_attestation_evidence_request doesn't have evidence")
        })?;

        self.evidence_sender
            .lock()
            .map_err(|err| {
                tonic::Status::internal(format!(
                    "couldn't get exclusive access to attestation evidence sender: {err}"
                ))
            })?
            .take()
            .ok_or_else(|| {
                tonic::Status::invalid_argument("app has already sent an attestation evidence")
            })?
            .send(evidence)
            .map_err(|_err| {
                tonic::Status::internal("couldn't send attestation evidence".to_string())
            })?;
        Ok(tonic::Response::new(()))
    }

    async fn get_endorsements(
        &self,
        _request: Request<()>,
    ) -> Result<Response<Endorsements>, tonic::Status> {
        Ok(tonic::Response::new(self.endorsements.clone()))
    }

    async fn notify_app_ready(&self, _request: Request<()>) -> Result<Response<()>, tonic::Status> {
        self.app_ready_notifier
            .lock()
            .map_err(|err| {
                tonic::Status::internal(format!(
                    "couldn't get exclusive access to notification channel: {err}"
                ))
            })?
            .take()
            .ok_or_else(|| {
                tonic::Status::invalid_argument("app has already sent a ready notification")
            })?
            .send(())
            .map_err(|_err| tonic::Status::internal("couldn't send notification".to_string()))?;
        Ok(tonic::Response::new(()))
    }
}

#[tonic::async_trait]
impl HostlibKeyProvisioning for LauncherServerImplementation {
    async fn get_key_provisioning_role(
        &self,
        _request: Request<()>,
    ) -> Result<Response<GetKeyProvisioningRoleResponse>, tonic::Status> {
        Ok(tonic::Response::new(GetKeyProvisioningRoleResponse {
            role: KeyProvisioningRole::Leader.into(),
        }))
    }

    async fn get_group_keys(
        &self,
        _request: Request<()>,
    ) -> Result<Response<GetGroupKeysResponse>, tonic::Status> {
        Err(tonic::Status::unimplemented("Key Provisioning is not implemented"))
    }
}

pub async fn new(
    listener: TcpListener,
    system_image: std::path::PathBuf,
    container_bundle: std::path::PathBuf,
    application_config: Vec<u8>,
    evidence_sender: oneshot::Sender<Evidence>,
    app_ready_notifier: oneshot::Sender<()>,
    shutdown: watch::Receiver<()>,
    endorsements: Endorsements,
) -> Result<(), anyhow::Error> {
    let server_impl = Arc::new(LauncherServerImplementation {
        system_image,
        container_bundle,
        application_config,
        evidence_sender: Mutex::new(Some(evidence_sender)),
        app_ready_notifier: Mutex::new(Some(app_ready_notifier)),
        endorsements,
    });

    let mut tcp_shutdown = shutdown.clone();
    Server::builder()
        .accept_http1(true) // Accept HTTP/1.1 to avoid FRAME_SIZE_ERROR from non-h2 clients
        .add_service(LauncherServer::from_arc(server_impl.clone()))
        .add_service(HostlibKeyProvisioningServer::from_arc(server_impl.clone()))
        .serve_with_incoming_shutdown(
            TcpListenerStream::new(listener),
            tcp_shutdown.changed().map(|_| ()),
        )
        .await
        .map_err(|error| anyhow!("server error: {:?}", error))
}
