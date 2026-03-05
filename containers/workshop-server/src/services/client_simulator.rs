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

use anyhow::{anyhow, Result};
use coset::{
    cbor::value::Value,
    cwt::{ClaimName, ClaimsSet},
    iana, Algorithm, CborSerializable, CoseKey, CoseSign1, KeyOperation, KeyType, Label,
};
use prost::Message;
use rand::Rng;

use super::key_derivation;
use blob_data_proto::fcp::confidentialcompute::BlobData;
use blob_header_proto::fcp::confidentialcompute::BlobHeader;
use confidential_transform_proto::fcp::confidentialcompute::{
    blob_metadata::hpke_plus_aead_metadata::{
        KmsAssociatedData, SymmetricKeyAssociatedDataComponents,
    },
    blob_metadata::{CompressionType, HpkePlusAeadMetadata},
    BlobMetadata,
};
use kms_proto::fcp::confidentialcompute::DeriveKeysRequest;

// Claim name for the public key in the signed COSE message (from KMS)
const PUBLIC_KEY_CLAIM: i64 = -65537;

/// Represents uploaded data from a client
#[derive(Clone, Debug)]
pub struct ClientUpload {
    pub upload_id: String,
    pub plaintext_data: Vec<u8>,
    pub blob_header: BlobHeader,
    pub blob_data: BlobData,
}

/// Client simulator for data uploads
pub struct ClientSimulator;

impl ClientSimulator {
    /// Simulates a client encrypting its data with a KMS-provided public key.
    ///
    /// # Arguments
    /// * `upload_id` - Unique identifier for this upload
    /// * `plaintext_data` - The data to encrypt
    /// * `public_key_cose` - The COSE key from KMS DeriveKeysResponse
    /// * `access_policy_sha256` - SHA-256 hash of the DataAccessPolicy
    ///
    /// # Returns
    /// A ClientUpload containing the encrypted data in BlobData format
    pub fn encrypt_data(
        upload_id: &str,
        plaintext_data: Vec<u8>,
        public_key_cose: &[u8],
        access_policy_sha256: &[u8],
    ) -> Result<ClientUpload> {
        // Parse the COSE key to extract the full CoseKey (not just raw key material)
        let cose_key = extract_cose_key(public_key_cose)?;
        let key_id = cose_key.key_id.clone();
        let cose_key_bytes =
            cose_key.to_vec().map_err(|e| anyhow!("Failed to serialize COSE key: {}", e))?;

        // Create BlobHeader with metadata FIRST, because its serialization
        // serves as the associated data for both AES-GCM and HPKE encryption
        // (per FCP spec: client_payload.h lines 40-41)
        let blob_id = Self::generate_blob_id();

        let blob_header = BlobHeader {
            blob_id: blob_id.clone(),
            key_id,
            access_policy_sha256: access_policy_sha256.to_vec(),
            access_policy_node_id: 0, // This is an input blob (node 0)
            payload_metadata: None,   // Will be set in BlobMetadata
            public_key_id: 0,
        };

        // The serialized BlobHeader is the associated data for BOTH:
        // 1. AES-GCM encryption of the plaintext
        // 2. HPKE encryption of the symmetric key
        // This is per FCP spec (client_payload.h lines 40-41)
        let ciphertext_associated_data = blob_header.encode_to_vec();

        // Encrypt the data using two-layer FCP encryption
        // Pass the full COSE key (not just raw material) to cfc_crypto
        let (encapsulated_key, encrypted_symmetric_key, encrypted_data, _associated_data) =
            key_derivation::encrypt_with_two_layer(
                &cose_key_bytes,
                &plaintext_data,
                &ciphertext_associated_data,
                &[],
            )?;

        // Create BlobMetadata with HPKE encryption details
        // For KMS integration, we need to set the KmsAssociatedData
        let kms_associated_data = KmsAssociatedData { record_header: blob_header.encode_to_vec() };

        let hpke_metadata = HpkePlusAeadMetadata {
            blob_id: blob_id.clone(),
            ciphertext_associated_data: ciphertext_associated_data.clone(),
            encrypted_symmetric_key: encrypted_symmetric_key,
            encapsulated_public_key: encapsulated_key,
            // counter: 0,
            symmetric_key_associated_data_components: Some(
                SymmetricKeyAssociatedDataComponents::KmsSymmetricKeyAssociatedData(
                    kms_associated_data,
                ),
            ),
        };

        let blob_metadata = BlobMetadata {
            total_size_bytes: plaintext_data.len() as i64,
            compression_type: CompressionType::None as i32,
            encryption_metadata: Some(
                confidential_transform_proto::fcp::confidentialcompute::blob_metadata::EncryptionMetadata::HpkePlusAeadData(hpke_metadata)
            ),
        };

        // Create BlobData with encrypted data
        let blob_data = BlobData { metadata: Some(blob_metadata), data: encrypted_data };

        Ok(ClientUpload {
            upload_id: upload_id.to_string(),
            plaintext_data,
            blob_header,
            blob_data,
        })
    }

    /// Generates a random blob ID
    fn generate_blob_id() -> Vec<u8> {
        let mut rng = rand::thread_rng();
        let mut blob_id = vec![0u8; 16];
        for byte in &mut blob_id {
            *byte = rng.gen();
        }
        blob_id
    }

    /// Simulates a client retrieving encryption keys from the KMS.
    ///
    /// # Arguments
    /// * `keyset_id` - The keyset ID to derive keys from
    /// * `access_policy_hash` - SHA-256 hash of the DataAccessPolicy
    ///
    /// # Returns
    /// A DeriveKeysRequest to send to the KMS
    pub fn create_derive_keys_request(
        keyset_id: u64,
        access_policy_hash: &[u8],
    ) -> DeriveKeysRequest {
        DeriveKeysRequest {
            keyset_id,
            authorized_logical_pipeline_policies_hashes: vec![access_policy_hash.to_vec()],
        }
    }

    /// Extracts the public key from a COSE key response from KMS.
    pub fn extract_public_key_from_response(public_key_cose: &[u8]) -> Result<Vec<u8>> {
        let (public_key, _) = extract_cose_key_data(public_key_cose)?;
        Ok(public_key)
    }
}

/// Extracts the full CoseKey from a CoseSign1-wrapped public key response from KMS.
/// Returns the complete CoseKey structure that can be used for encryption.
fn extract_cose_key(public_key_bytes: &[u8]) -> Result<CoseKey> {
    // Parse the signed COSE message (CoseSign1) containing the public key
    let cose_sign1 = CoseSign1::from_slice(public_key_bytes)
        .map_err(|e| anyhow!("Failed to parse CoseSign1: {:?}", e))?;

    // Extract the payload which contains the claims
    let payload =
        cose_sign1.payload.as_ref().ok_or_else(|| anyhow!("CoseSign1 payload is missing"))?;

    // Parse the payload as a ClaimsSet
    let claims = ClaimsSet::from_slice(payload)
        .map_err(|e| anyhow!("Failed to parse ClaimsSet from payload: {:?}", e))?;

    // Find the PUBLIC_KEY_CLAIM in the rest claims
    let cose_key_value = claims
        .rest
        .iter()
        .find(|(name, _)| name == &ClaimName::PrivateUse(PUBLIC_KEY_CLAIM))
        .map(|(_, value)| value)
        .ok_or_else(|| anyhow!("PUBLIC_KEY_CLAIM not found in ClaimsSet"))?;

    // Extract bytes from the CBOR value
    let cose_key_bytes = cose_key_value
        .as_bytes()
        .ok_or_else(|| anyhow!("Failed to extract bytes from PUBLIC_KEY_CLAIM value"))?;

    // Parse the bytes as a CoseKey and return it
    CoseKey::from_slice(cose_key_bytes).map_err(|e| anyhow!("Failed to parse CoseKey: {:?}", e))
}

/// Extracts multiple fields from a COSE key structure.
/// Returns (public_key_material, key_id)
/// Following the pattern from containers/kms/key_derivation_test.rs extract_raw_key()
fn extract_cose_key_data(public_key_bytes: &[u8]) -> Result<(Vec<u8>, Vec<u8>)> {
    // Parse the signed COSE message (CoseSign1) containing the public key
    let cose_sign1 = CoseSign1::from_slice(public_key_bytes)
        .map_err(|e| anyhow!("Failed to parse CoseSign1: {:?}", e))?;

    // Extract the payload which contains the claims
    let payload =
        cose_sign1.payload.as_ref().ok_or_else(|| anyhow!("CoseSign1 payload is missing"))?;

    // Parse the payload as a ClaimsSet
    let claims = ClaimsSet::from_slice(payload)
        .map_err(|e| anyhow!("Failed to parse ClaimsSet from payload: {:?}", e))?;

    // Find the PUBLIC_KEY_CLAIM in the rest claims
    let cose_key_value = claims
        .rest
        .iter()
        .find(|(name, _)| name == &ClaimName::PrivateUse(PUBLIC_KEY_CLAIM))
        .map(|(_, value)| value)
        .ok_or_else(|| anyhow!("PUBLIC_KEY_CLAIM not found in ClaimsSet"))?;

    // Extract bytes from the CBOR value
    let cose_key_bytes = cose_key_value
        .as_bytes()
        .ok_or_else(|| anyhow!("Failed to extract bytes from PUBLIC_KEY_CLAIM value"))?;

    // Parse the bytes as a CoseKey
    let cose_key = CoseKey::from_slice(cose_key_bytes)
        .map_err(|e| anyhow!("Failed to parse CoseKey: {:?}", e))?;

    // Extract public key (label -1 / OKP X coordinate) from params
    let public_key = cose_key
        .params
        .iter()
        .find(|(label, _)| label == &Label::Int(iana::OkpKeyParameter::X as i64))
        .ok_or_else(|| anyhow!("X coordinate not found in OKP key"))?
        .1
        .clone()
        .as_bytes()
        .ok_or_else(|| anyhow!("Failed to extract bytes from X coordinate"))?
        .to_vec();

    // Extract key_id from the CoseKey
    let key_id = cose_key.key_id.clone();

    Ok((public_key, key_id))
}

/// Simulates storing uploaded data in a storage system.
pub struct StorageSimulator {
    data: std::collections::HashMap<String, ClientUpload>,
}

impl StorageSimulator {
    pub fn new() -> Self {
        StorageSimulator { data: std::collections::HashMap::new() }
    }

    pub fn store_upload(&mut self, upload: ClientUpload) {
        self.data.insert(upload.upload_id.clone(), upload);
    }

    pub fn get_upload(&self, upload_id: &str) -> Option<ClientUpload> {
        self.data.get(upload_id).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_simulator() {
        let mut storage = StorageSimulator::new();
        let blob_data = BlobData { metadata: None, data: vec![4, 5, 6] };
        let blob_header = BlobHeader {
            blob_id: vec![1, 2, 3],
            key_id: vec![],
            access_policy_sha256: vec![],
            access_policy_node_id: 0,
            payload_metadata: None,
            public_key_id: 0,
        };
        let upload = ClientUpload {
            upload_id: "test_upload".to_string(),
            plaintext_data: vec![1, 2, 3],
            blob_header,
            blob_data,
        };

        storage.store_upload(upload.clone());
        let retrieved = storage.get_upload("test_upload").unwrap();

        assert_eq!(upload.upload_id, retrieved.upload_id);
        assert_eq!(upload.blob_data.data, retrieved.blob_data.data);
    }
}
