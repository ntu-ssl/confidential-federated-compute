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

//! Pure Rust implementation of the DataReadWrite gRPC service.
//!
//! This service allows storing and retrieving encrypted data with configurable
//! access policy hashes and KMS keys, enabling real KMS integration testing.

use anyhow::{anyhow, Result};
use coset::{CborSerializable, CoseKey};
use prost::Message;
use rand::RngCore;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tonic::{Request, Response, Status};
use tracing::{debug, error, info, warn};

use key_derivation;

// Proto imports
use blob_header_proto::fcp::confidentialcompute::BlobHeader;
use confidential_transform_proto::fcp::confidentialcompute::{
    blob_metadata::hpke_plus_aead_metadata::{
        KmsAssociatedData, SymmetricKeyAssociatedDataComponents,
    },
    blob_metadata::{CompressionType, EncryptionMetadata, HpkePlusAeadMetadata},
    BlobMetadata,
};
use data_read_write_proto::fcp::confidentialcompute::outgoing::{
    data_read_write_server::{DataReadWrite, DataReadWriteServer},
    ReadRequest, ReadResponse, WriteRequest, WriteResponse,
};


/// Configuration for the DataReadWriteService
#[derive(Clone)]
pub struct DataReadWriteConfig {
    /// The public key (COSE format) to use for encrypting stored data.
    /// For real KMS: obtained via `derive_keys()` from KMS.
    /// For testing: generated locally.
    pub input_public_key_cose: Vec<u8>,
    /// The raw private key for input encryption.
    /// For real KMS: None (KMS holds the private key).
    /// For testing: generated locally for verification.
    pub input_private_key_raw: Option<Vec<u8>>,
    /// The key ID associated with the input public key
    pub key_id: Vec<u8>,
    /// The access policy SHA-256 hash to use for stored blobs
    pub access_policy_sha256: Vec<u8>,
}

/// Pure Rust DataReadWrite service that supports real KMS integration.
///
/// This service implements the same interface as FakeDataReadWriteService in C++:
/// - StoreEncryptedMessageForKms: Encrypts and stores data for Read RPC
/// - StorePlaintextMessage: Stores unencrypted data for Read RPC
/// - Read RPC: Returns stored data
/// - Write RPC: Decrypts and stores released data from TEE
pub struct RealDataReadWriteService {
    /// Configuration for encryption
    config: DataReadWriteConfig,
    /// Storage for data to be returned by Read RPC: uri -> ReadResponse
    storage: Arc<Mutex<HashMap<String, ReadResponse>>>,
    /// List of URIs from received ReadRequests (for testing)
    read_request_uris: Arc<Mutex<Vec<String>>>,
    /// Storage for WriteRequests from Write RPC: key -> WriteRequest
    /// The data remains encrypted; decryption happens via KMS.ReleaseResults in main.rs
    write_requests: Arc<Mutex<HashMap<String, WriteRequest>>>,
    /// Key pair for result decryption (public_cose, private_raw)
    /// TEE encrypts results with result_public_key, service decrypts with result_private_key
    result_key_pair: (Vec<u8>, Vec<u8>),
}

impl RealDataReadWriteService {
    /// Creates a new service with the given configuration.
    ///
    /// # Arguments
    /// * `config` - Configuration containing public key, private key, and access policy hash
    pub fn new(config: DataReadWriteConfig) -> Result<Self> {
        // Generate key pair for result decryption
        let result_key_pair = key_derivation::generate_key_pair(b"result")?;

        Ok(Self {
            config,
            storage: Arc::new(Mutex::new(HashMap::new())),
            read_request_uris: Arc::new(Mutex::new(Vec::new())),
            write_requests: Arc::new(Mutex::new(HashMap::new())),
            result_key_pair,
        })
    }

    /// Creates a new service with a KMS-derived public key.
    ///
    /// This is the constructor for real KMS integration. The public key is obtained
    /// from KMS via `derive_keys()`, and KMS holds the corresponding private key
    /// for rewrapping symmetric keys to TEEs.
    ///
    /// # Arguments
    /// * `kms_public_key_cose` - The public key from KMS (COSE format)
    /// * `access_policy_sha256` - The SHA-256 hash of the DataAccessPolicy
    pub fn new_with_kms_key(
        kms_public_key_cose: Vec<u8>,
        access_policy_sha256: Vec<u8>,
    ) -> Result<Self> {
        // Extract key_id from the COSE key
        let cose_key = CoseKey::from_slice(&kms_public_key_cose)
            .map_err(|e| anyhow!("Failed to parse COSE key: {:?}", e))?;
        let key_id = cose_key.key_id.clone();

        let config = DataReadWriteConfig {
            input_public_key_cose: kms_public_key_cose,
            input_private_key_raw: None, // KMS holds the private key
            key_id,
            access_policy_sha256,
        };

        Self::new(config)
    }

    /// Creates a new service with test/simulation keys.
    ///
    /// This generates local key pairs for both input encryption and result decryption,
    /// suitable for testing without a real KMS.
    ///
    /// # Arguments
    /// * `access_policy_sha256` - The SHA-256 hash of the DataAccessPolicy
    pub fn new_for_testing(access_policy_sha256: Vec<u8>) -> Result<Self> {
        // Generate input key pair (both public and private)
        let (input_public_key_cose, input_private_key_raw) =
            key_derivation::generate_key_pair(b"input")?;

        // Extract key_id from the COSE key
        let cose_key = CoseKey::from_slice(&input_public_key_cose)
            .map_err(|e| anyhow!("Failed to parse COSE key: {:?}", e))?;
        let key_id = cose_key.key_id.clone();

        let config = DataReadWriteConfig {
            input_public_key_cose,
            input_private_key_raw: Some(input_private_key_raw),
            key_id,
            access_policy_sha256,
        };

        Self::new(config)
    }

    /// Encrypts the provided message and stores it within a ReadResponse.
    ///
    /// This is the Rust equivalent of C++ `StoreEncryptedMessageForKms`.
    /// The ReadResponse will be returned in response to a later ReadRequest for the given URI.
    ///
    /// # Arguments
    /// * `uri` - The URI to store the data under
    /// * `message` - The plaintext message to encrypt and store
    /// * `blob_id` - Optional blob ID (generated randomly if not provided)
    pub fn store_encrypted_message_for_kms(
        &self,
        uri: &str,
        message: &[u8],
        blob_id: Option<Vec<u8>>,
    ) -> Result<()> {
        // Check if URI already exists
        {
            let storage = self.storage.lock().map_err(|_| anyhow!("Failed to lock storage"))?;
            if storage.contains_key(uri) {
                return Err(anyhow!("Uri already set."));
            }
        }

        // Generate blob_id if not provided
        let blob_id = blob_id.unwrap_or_else(|| {
            let mut id = vec![0u8; 16];
            rand::thread_rng().fill_bytes(&mut id);
            id
        });

        // Create BlobHeader with access policy hash
        let blob_header = BlobHeader {
            blob_id: blob_id.clone(),
            key_id: self.config.key_id.clone(),
            access_policy_sha256: self.config.access_policy_sha256.clone(),
            access_policy_node_id: 0, // Input node
            payload_metadata: None,
            public_key_id: 0,
        };

        // Serialize BlobHeader as associated data
        let associated_data = blob_header.encode_to_vec();

        // Encrypt using two-layer FCP encryption
        let (encapsulated_key, encrypted_symmetric_key, ciphertext, _) =
            key_derivation::encrypt_with_two_layer(
                &self.config.input_public_key_cose,
                message,
                &associated_data,
                &[],
            )?;

        // Create KMS associated data (record_header = serialized BlobHeader)
        let kms_associated_data = KmsAssociatedData { record_header: associated_data.clone() };

        // Create HPKE metadata (matching C++ structure)
        let hpke_metadata = HpkePlusAeadMetadata {
            blob_id: Vec::new(), // C++ doesn't set blob_id in HpkePlusAeadMetadata
            ciphertext_associated_data: associated_data.clone(),
            encrypted_symmetric_key,
            encapsulated_public_key: encapsulated_key,
            counter: 0,
            symmetric_key_associated_data_components: Some(
                SymmetricKeyAssociatedDataComponents::KmsSymmetricKeyAssociatedData(
                    kms_associated_data,
                ),
            ),
        };

        // Create BlobMetadata
        let blob_metadata = BlobMetadata {
            total_size_bytes: ciphertext.len() as i64,
            compression_type: CompressionType::None as i32,
            encryption_metadata: Some(EncryptionMetadata::HpkePlusAeadData(hpke_metadata)),
        };

        // Create ReadResponse
        let read_response = ReadResponse {
            first_response_metadata: Some(blob_metadata),
            finish_read: true,
            data: ciphertext,
        };

        // Store in map
        let mut storage = self.storage.lock().map_err(|_| anyhow!("Failed to lock storage"))?;
        storage.insert(uri.to_string(), read_response);
        Ok(())
    }

    /// Stores the provided message within a ReadResponse as plaintext (unencrypted).
    ///
    /// This is the Rust equivalent of C++ `StorePlaintextMessage`.
    /// The ReadResponse will be returned in response to a later ReadRequest for the given URI.
    ///
    /// # Arguments
    /// * `uri` - The URI to store the data under
    /// * `message` - The plaintext message to store
    pub fn store_plaintext_message(&self, uri: &str, message: &[u8]) -> Result<()> {
        // Check if URI already exists
        {
            let storage = self.storage.lock().map_err(|_| anyhow!("Failed to lock storage"))?;
            if storage.contains_key(uri) {
                return Err(anyhow!("Uri already set."));
            }
        }

        // Create BlobMetadata with unencrypted marker (no blob_id, matching C++ behavior)
        let blob_metadata = BlobMetadata {
            total_size_bytes: message.len() as i64,
            compression_type: CompressionType::None as i32,
            encryption_metadata: Some(EncryptionMetadata::Unencrypted(
                confidential_transform_proto::fcp::confidentialcompute::blob_metadata::Unencrypted {
                    blob_id: Vec::new(), // C++ doesn't set blob_id for plaintext
                },
            )),
        };

        let read_response = ReadResponse {
            first_response_metadata: Some(blob_metadata),
            finish_read: true,
            data: message.to_vec(),
        };

        let mut storage = self.storage.lock().map_err(|_| anyhow!("Failed to lock storage"))?;
        storage.insert(uri.to_string(), read_response);
        Ok(())
    }

    /// Gets the input public/private key pair.
    /// Returns (public_key_cose, Option<private_key_raw>).
    /// Note: private_key is None when using real KMS (KMS holds the key).
    pub fn get_input_public_private_key_pair(&self) -> (&[u8], Option<&[u8]>) {
        (&self.config.input_public_key_cose, self.config.input_private_key_raw.as_deref())
    }

    /// Gets the result public/private key pair.
    /// Returns (public_key_cose, private_key_raw).
    pub fn get_result_public_private_key_pair(&self) -> (&[u8], &[u8]) {
        (&self.result_key_pair.0, &self.result_key_pair.1)
    }

    /// Gets the public key for input encryption (COSE format).
    pub fn get_input_public_key(&self) -> &[u8] {
        &self.config.input_public_key_cose
    }

    /// Gets the result public key (COSE format) for TEE to encrypt results.
    pub fn get_result_public_key(&self) -> &[u8] {
        &self.result_key_pair.0
    }

    /// Gets the result private key (raw format) for decrypting results.
    pub fn get_result_private_key(&self) -> &[u8] {
        &self.result_key_pair.1
    }

    /// Gets the access policy hash being used.
    pub fn get_access_policy_hash(&self) -> &[u8] {
        &self.config.access_policy_sha256
    }

    /// Returns a list of URIs from received ReadRequest args.
    pub fn get_read_request_uris(&self) -> Vec<String> {
        self.read_request_uris.lock().map(|guard| guard.clone()).unwrap_or_default()
    }

    /// Returns a map of key to WriteRequest received by the Write endpoint.
    /// The data is encrypted; use KMS.ReleaseResults to get decryption keys.
    pub fn get_all_write_requests(&self) -> HashMap<String, WriteRequest> {
        self.write_requests.lock().map(|guard| guard.clone()).unwrap_or_default()
    }

    /// Gets specific WriteRequest by key.
    /// Contains encrypted data, metadata, and release_token needed for KMS.ReleaseResults.
    pub fn get_write_request_for_key(&self, key: &str) -> Option<WriteRequest> {
        let requests = self.write_requests.lock().ok()?;
        requests.get(key).cloned()
    }

    /// Starts the gRPC server on the specified address.
    ///
    /// Returns the actual address the server is listening on.
    pub async fn start_server(self: Arc<Self>, addr: SocketAddr) -> Result<SocketAddr> {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        let local_addr = listener.local_addr()?;

        let service = DataReadWriteServer::from_arc(self);

        tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(service)
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
                .await
                .expect("gRPC server error");
        });

        Ok(local_addr)
    }

    /// Starts the gRPC server on IPv4 localhost with a random available port.
    ///
    /// This is the preferred method when the service needs to be accessible
    /// from QEMU VMs via port forwarding.
    ///
    /// Returns the actual port the server is listening on.
    pub async fn start_server_ipv4(self: Arc<Self>) -> Result<u16> {
        use std::net::Ipv4Addr;
        let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, 0));
        let local_addr = self.start_server(addr).await?;
        Ok(local_addr.port())
    }

    /// Starts the gRPC server on IPv4 localhost with a specific port.
    ///
    /// Use this when you need a predictable port for QEMU guestfwd configuration.
    ///
    /// # Arguments
    /// * `port` - The port to bind to
    pub async fn start_server_on_port(self: Arc<Self>, port: u16) -> Result<()> {
        use std::net::Ipv4Addr;
        let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
        self.start_server(addr).await?;
        Ok(())
    }
}

#[tonic::async_trait]
impl DataReadWrite for RealDataReadWriteService {
    type ReadStream = tokio_stream::wrappers::ReceiverStream<Result<ReadResponse, Status>>;

    async fn read(
        &self,
        request: Request<ReadRequest>,
    ) -> Result<Response<Self::ReadStream>, Status> {
        let uri = request.into_inner().uri;

        // Clone response while holding the lock, then drop the lock before await
        let response = {
            let storage =
                self.storage.lock().map_err(|_| Status::internal("Failed to lock storage"))?;
            storage
                .get(&uri)
                .cloned()
                .ok_or_else(|| Status::not_found(format!("Requested uri {} not found.", uri)))?
        }; // MutexGuard dropped here

        // Track the URI from the request (for testing)
        {
            let mut uris = self
                .read_request_uris
                .lock()
                .map_err(|_| Status::internal("Failed to lock read_request_uris"))?;
            uris.push(uri);
        }

        // Create a stream that sends the single response
        let (tx, rx) = mpsc::channel(1);
        tx.send(Ok(response)).await.map_err(|_| Status::internal("Failed to send response"))?;

        Ok(Response::new(tokio_stream::wrappers::ReceiverStream::new(rx)))
    }

    async fn write(
        &self,
        request: Request<tonic::Streaming<WriteRequest>>,
    ) -> Result<Response<WriteResponse>, Status> {
        info!("Write RPC called");
        let mut stream = request.into_inner();

        // Collect all write requests
        let mut requests = Vec::new();
        while let Some(req) = stream.message().await? {
            debug!("Received WriteRequest chunk");
            requests.push(req);
        }

        info!("Received {} WriteRequest(s)", requests.len());

        if requests.is_empty() {
            error!("No write requests received");
            return Err(Status::invalid_argument("No write requests received"));
        }

        // For now, we only support non-chunked writes (matching C++ behavior)
        if requests.len() > 1 {
            error!("Chunked WriteRequests not supported, got {} requests", requests.len());
            return Err(Status::invalid_argument(
                "Chunked WriteRequests are not supported by the RealDataReadWriteService",
            ));
        }

        let write_request = requests[0].clone();
        let key = write_request.key.clone();

        info!(
            "WriteRequest: key='{}', data_size={}, commit={}, release_token_size={}",
            key,
            write_request.data.len(),
            write_request.commit,
            write_request.release_token.len()
        );

        // Get metadata from the request and verify HPKE encryption
        let metadata = match &write_request.first_request_metadata {
            Some(m) => {
                debug!(
                    "Metadata: total_size={}, compression_type={}",
                    m.total_size_bytes, m.compression_type
                );
                m
            }
            None => {
                error!("WriteRequest missing first_request_metadata");
                return Err(Status::invalid_argument(
                    "WriteRequest missing first_request_metadata",
                ));
            }
        };

        // Verify we have HPKE metadata
        match &metadata.encryption_metadata {
            Some(EncryptionMetadata::HpkePlusAeadData(hpke)) => {
                debug!(
                    "HPKE metadata: encapped_key_size={}, encrypted_sym_key_size={}, ciphertext_aad_size={}",
                    hpke.encapsulated_public_key.len(),
                    hpke.encrypted_symmetric_key.len(),
                    hpke.ciphertext_associated_data.len()
                );
            }
            Some(EncryptionMetadata::Unencrypted(_)) => {
                error!("WriteRequest has Unencrypted metadata, expected HpkePlusAeadData");
                return Err(Status::invalid_argument(
                    "WriteRequest has Unencrypted metadata, expected HpkePlusAeadData",
                ));
            }
            None => {
                error!("WriteRequest missing encryption_metadata");
                return Err(Status::invalid_argument(
                    "WriteRequest missing HpkePlusAeadData encryption metadata",
                ));
            }
        };

        // Store the WriteRequest directly
        // Decryption will happen later via KMS.ReleaseResults in main.rs
        {
            let mut requests = self
                .write_requests
                .lock()
                .map_err(|_| Status::internal("Failed to lock write_requests"))?;
            requests.insert(key.clone(), write_request.clone());
            info!(
                "Stored WriteRequest for key '{}' (will decrypt via KMS.ReleaseResults)",
                key
            );
        }

        info!("Write RPC completed successfully");
        Ok(Response::new(WriteResponse { reply: None, file_uri: String::new() }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_service_for_testing() -> Result<()> {
        let access_policy_hash = vec![1u8; 32]; // Fake 32-byte hash
        let service = RealDataReadWriteService::new_for_testing(access_policy_hash.clone())?;

        assert_eq!(service.get_access_policy_hash(), &access_policy_hash);
        assert!(!service.get_input_public_key().is_empty());
        assert!(!service.get_result_public_key().is_empty());
        assert!(!service.get_result_private_key().is_empty());

        // Verify key pairs
        let (input_pub, input_priv) = service.get_input_public_private_key_pair();
        assert!(!input_pub.is_empty());
        // For testing, private key is available
        assert_eq!(input_priv.unwrap().len(), 32); // X25519 private key is 32 bytes

        let (result_pub, result_priv) = service.get_result_public_private_key_pair();
        assert!(!result_pub.is_empty());
        assert_eq!(result_priv.len(), 32);

        Ok(())
    }

    #[test]
    fn test_store_and_retrieve_plaintext() -> Result<()> {
        let access_policy_hash = vec![1u8; 32];
        let service = RealDataReadWriteService::new_for_testing(access_policy_hash)?;

        let uri = "test/data/uri";
        let message = b"test plaintext data";

        service.store_plaintext_message(uri, message)?;

        // Verify storage
        let storage = service.storage.lock().unwrap();
        assert!(storage.contains_key(uri));

        let response = storage.get(uri).unwrap();
        assert_eq!(response.data, message);
        assert!(response.finish_read);

        // Verify unencrypted metadata (no blob_id)
        let metadata = response.first_response_metadata.as_ref().unwrap();
        match &metadata.encryption_metadata {
            Some(EncryptionMetadata::Unencrypted(unenc)) => {
                assert!(unenc.blob_id.is_empty()); // C++ doesn't set blob_id
            }
            _ => panic!("Expected Unencrypted metadata"),
        }

        Ok(())
    }

    #[test]
    fn test_store_encrypted_message_for_kms() -> Result<()> {
        let access_policy_hash = vec![1u8; 32];
        let service = RealDataReadWriteService::new_for_testing(access_policy_hash.clone())?;

        let uri = "test/encrypted/uri";
        let plaintext = b"secret data to encrypt";

        service.store_encrypted_message_for_kms(uri, plaintext, None)?;

        // Verify storage contains encrypted data
        let storage = service.storage.lock().unwrap();
        assert!(storage.contains_key(uri));

        let response = storage.get(uri).unwrap();
        // Encrypted data should be different from plaintext
        assert_ne!(response.data, plaintext);
        assert!(response.finish_read);

        // Verify metadata
        let metadata = response.first_response_metadata.as_ref().unwrap();
        match &metadata.encryption_metadata {
            Some(EncryptionMetadata::HpkePlusAeadData(hpke)) => {
                assert!(!hpke.encapsulated_public_key.is_empty());
                assert!(!hpke.encrypted_symmetric_key.is_empty());
                assert!(!hpke.ciphertext_associated_data.is_empty());
                // Verify KMS associated data
                match &hpke.symmetric_key_associated_data_components {
                    Some(SymmetricKeyAssociatedDataComponents::KmsSymmetricKeyAssociatedData(
                        kms_data,
                    )) => {
                        assert!(!kms_data.record_header.is_empty());
                    }
                    _ => panic!("Expected KmsSymmetricKeyAssociatedData"),
                }
            }
            _ => panic!("Expected HpkePlusAeadData encryption metadata"),
        }

        Ok(())
    }

    #[test]
    fn test_duplicate_uri_rejected() -> Result<()> {
        let access_policy_hash = vec![1u8; 32];
        let service = RealDataReadWriteService::new_for_testing(access_policy_hash)?;

        let uri = "test/duplicate/uri";

        service.store_plaintext_message(uri, b"first")?;
        let result = service.store_plaintext_message(uri, b"second");

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already set"));

        Ok(())
    }

    #[test]
    fn test_store_with_custom_blob_id() -> Result<()> {
        let access_policy_hash = vec![1u8; 32];
        let service = RealDataReadWriteService::new_for_testing(access_policy_hash)?;

        let uri = "test/custom_blob_id/uri";
        let plaintext = b"test data";
        let custom_blob_id = vec![0xAB; 16];

        service.store_encrypted_message_for_kms(uri, plaintext, Some(custom_blob_id.clone()))?;

        // Verify the blob_id in BlobHeader (via ciphertext_associated_data)
        let storage = service.storage.lock().unwrap();
        let response = storage.get(uri).unwrap();
        let metadata = response.first_response_metadata.as_ref().unwrap();

        if let Some(EncryptionMetadata::HpkePlusAeadData(hpke)) = &metadata.encryption_metadata {
            // Parse the BlobHeader from ciphertext_associated_data
            let blob_header = BlobHeader::decode(hpke.ciphertext_associated_data.as_slice())
                .expect("Failed to decode BlobHeader");
            assert_eq!(blob_header.blob_id, custom_blob_id);
        }

        Ok(())
    }
}
