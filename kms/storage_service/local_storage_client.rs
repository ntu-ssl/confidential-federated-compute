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

use std::sync::{Arc, Mutex};

use anyhow::Result;
use oak_time::Clock;
use storage::Storage;
use storage_client::StorageClient;
use storage_proto::{
    confidential_federated_compute::kms::{
        ReadRequest, ReadResponse, UpdateRequest, UpdateResponse,
    },
    timestamp_proto::google::protobuf::Timestamp,
};

/// A local storage client that uses an in-memory Storage instance.
/// This allows KMS to run as a single instance without needing network-based storage.
pub struct LocalStorageClient {
    storage: Arc<Mutex<Storage>>,
    clock: Arc<dyn Clock>,
}

impl LocalStorageClient {
    /// Creates a new LocalStorageClient with the given storage and clock.
    pub fn new(storage: Arc<Mutex<Storage>>, clock: Arc<dyn Clock>) -> Self {
        Self { storage, clock }
    }
}

impl StorageClient for LocalStorageClient {
    async fn read(&self, request: ReadRequest) -> Result<ReadResponse> {
        let storage = self.storage.lock().unwrap();
        storage.read(&request)
    }

    async fn update(&self, request: UpdateRequest) -> Result<UpdateResponse> {
        // Get current timestamp from the clock
        let instant = self.clock.get_time();
        let timestamp = instant.into_timestamp();
        let now = Timestamp { seconds: timestamp.seconds, nanos: timestamp.nanos };

        let mut storage = self.storage.lock().unwrap();
        storage.update(&now, request)
    }
}
