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

//! Proxy service that forwards ComputationDelegation::Execute requests
//! from the root TEE to worker TEEs via ProgramWorker::Execute.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tonic::{Request, Response, Status};

use computation_delegation_proto::fcp::confidentialcompute::outgoing::{
    computation_delegation_server::ComputationDelegation,
    ComputationRequest as DelegationRequest, ComputationResponse as DelegationResponse,
};

use program_worker_proto::fcp::confidentialcompute::{
    program_worker_client::ProgramWorkerClient,
    ComputationRequest as WorkerRequest,
};

/// Proxy that implements ComputationDelegation by forwarding to ProgramWorker on workers.
/// Each worker gets its own mutex so calls to different workers don't block each other.
pub struct ComputationDelegationProxy {
    /// Map from worker_bns address to a per-worker client (each independently locked).
    clients: HashMap<String, Arc<Mutex<ProgramWorkerClient<tonic::transport::Channel>>>>,
}

impl ComputationDelegationProxy {
    /// Creates a new proxy with pre-connected clients for the given worker addresses.
    pub async fn new(worker_addresses: Vec<String>) -> Result<Self, Box<dyn std::error::Error>> {
        let mut clients = HashMap::new();
        // Match the 2GB limit used by the worker's gRPC server (kChannelMaxMessageSize).
        const MAX_MESSAGE_SIZE: usize = 2 * 1000 * 1000 * 1000;
        for addr in &worker_addresses {
            let channel = tonic::transport::Endpoint::from_shared(addr.clone())?
                .connect()
                .await?;
            let client = ProgramWorkerClient::new(channel)
                .max_decoding_message_size(MAX_MESSAGE_SIZE)
                .max_encoding_message_size(MAX_MESSAGE_SIZE);
            clients.insert(addr.clone(), Arc::new(Mutex::new(client)));
        }
        Ok(Self {
            clients,
        })
    }
}

#[tonic::async_trait]
impl ComputationDelegation for ComputationDelegationProxy {
    async fn execute(
        &self,
        request: Request<DelegationRequest>,
    ) -> Result<Response<DelegationResponse>, Status> {
        let req = request.into_inner();
        let worker_bns = &req.worker_bns;

        let client_mutex = self.clients.get(worker_bns).ok_or_else(|| {
            Status::not_found(format!("Unknown worker: {}", worker_bns))
        })?;

        // Only lock this specific worker's client, not all workers.
        let mut client = client_mutex.lock().await;

        let worker_request = WorkerRequest {
            computation: req.computation,
        };

        let worker_response = client
            .execute(worker_request)
            .await
            .map_err(|e| {
                Status::internal(format!("Worker error: {}", e))
            })?
            .into_inner();

        Ok(Response::new(DelegationResponse {
            result: worker_response.result,
        }))
    }
}
