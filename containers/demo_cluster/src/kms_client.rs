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

use anyhow::{anyhow, Context, Result};
use prost::{DecodeError, Message};
use std::time::Duration;
use tonic::transport::Channel;
use tracing::{debug, error, info};

/// An extension trait to convert between messages built using different proto libraries
/// Converts by serializing and deserializing
pub trait ProstProtoConversionExt<T: Message + Default>: Message + Sized {
    fn convert(&self) -> Result<T, DecodeError> {
        T::decode(self.encode_to_vec().as_slice())
    }
}

// Implement conversion for Evidence between oak_proto_rust and kms_proto versions
impl ProstProtoConversionExt<kms_proto::evidence_proto::oak::attestation::v1::Evidence>
    for oak_proto_rust::oak::attestation::v1::Evidence
{
}

// Implement conversion for Endorsements between oak_proto_rust and kms_proto versions
impl ProstProtoConversionExt<kms_proto::endorsement_proto::oak::attestation::v1::Endorsements>
    for oak_proto_rust::oak::attestation::v1::Endorsements
{
}

// Implement conversion for ReferenceValues between oak_proto_rust and reference_value_proto versions
use reference_value_proto::oak::attestation::v1::ReferenceValues as RefValueProtoReferenceValues;
impl ProstProtoConversionExt<RefValueProtoReferenceValues>
    for oak_proto_rust::oak::attestation::v1::ReferenceValues
{
}

// Implement conversion for Evidence between kms_proto and evidence_proto versions
impl ProstProtoConversionExt<oak_proto_rust::oak::attestation::v1::Evidence>
    for kms_proto::evidence_proto::oak::attestation::v1::Evidence
{
}

use duration_proto::google::protobuf::Duration as ProtoDuration;
use kms_proto::fcp::confidentialcompute::{
    key_management_service_client::KeyManagementServiceClient,
    release_results_request::ReleasableResult, AuthorizeConfidentialTransformRequest,
    DeriveKeysRequest, GetKeysetRequest, RegisterPipelineInvocationRequest, ReleaseResultsRequest,
    RotateKeysetRequest,
};
use kms_proto::{
    endorsement_proto::oak::attestation::v1::Endorsements,
    evidence_proto::oak::attestation::v1::Evidence,
};

// NOTE: Test utilities (FakeTestData, get_test_evidence, get_test_endorsements)
// have been removed from this simplified version.
// In a full implementation, these would be imported from oak_session utilities.

/// KMS client for communicating with the running KMS service
pub struct KmsClient {
    client: KeyManagementServiceClient<Channel>,
}

impl KmsClient {
    /// Creates a new KMS client connected to the specified address
    pub async fn new(kms_address: &str) -> Result<Self> {
        info!("Connecting to KMS at {}", kms_address);
        let channel = Channel::from_shared(kms_address.to_string())
            .context("Invalid KMS address")?
            .connect_timeout(Duration::from_secs(10))
            .connect()
            .await
            .context("Failed to connect to KMS")?;

        Ok(KmsClient { client: KeyManagementServiceClient::new(channel) })
    }

    /// Rotates the keyset to add a new encryption key
    pub async fn rotate_keyset(&mut self, keyset_id: u64, ttl_seconds: i64) -> Result<()> {
        debug!("Rotating keyset {}", keyset_id);

        let request = RotateKeysetRequest {
            keyset_id,
            ttl: Some(ProtoDuration { seconds: ttl_seconds, nanos: 0 }),
        };

        self.client.rotate_keyset(request).await.map_err(|status| {
            let error_msg = format!(
                "Failed to rotate keyset {} - Code: {:?}, Message: {}",
                keyset_id,
                status.code(),
                status.message()
            );
            error!("{}", error_msg);
            anyhow!(error_msg)
        })?;

        info!("Keyset {} rotated successfully", keyset_id);
        Ok(())
    }

    /// Gets information about a keyset
    pub async fn get_keyset(&mut self, keyset_id: u64) -> Result<Vec<u8>> {
        debug!("Getting keyset {}", keyset_id);

        let request = GetKeysetRequest { keyset_id };
        let response =
            self.client.get_keyset(request).await.context("Failed to get keyset")?.into_inner();

        info!("Retrieved keyset {} with {} keys", keyset_id, response.keys.len());
        Ok(response.encode_to_vec())
    }

    /// Derives encryption keys for the given access policies
    pub async fn derive_keys(
        &mut self,
        keyset_id: u64,
        policy_hashes: Vec<Vec<u8>>,
    ) -> Result<Vec<Vec<u8>>> {
        debug!("Deriving keys for keyset {} with {} policies", keyset_id, policy_hashes.len());

        let request = DeriveKeysRequest {
            keyset_id,
            authorized_logical_pipeline_policies_hashes: policy_hashes,
        };

        let response = self
            .client
            .derive_keys(request)
            .await
            .map_err(|status| {
                let error_msg = format!(
                    "Failed to derive keys - Code: {:?}, Message: {}",
                    status.code(),
                    status.message()
                );
                error!("{}", error_msg);
                anyhow!(error_msg)
            })?
            .into_inner();

        info!("Derived {} public keys for keyset {}", response.public_keys.len(), keyset_id);

        Ok(response.public_keys)
    }

    /// Registers a pipeline invocation with the KMS
    pub async fn register_pipeline_invocation(
        &mut self,
        logical_pipeline_name: String,
        pipeline_variant_policy: Vec<u8>,
        keyset_ids: Vec<u64>,
        authorized_policies: Vec<Vec<u8>>,
        intermediates_ttl_seconds: i64,
    ) -> Result<(Vec<u8>, Vec<Vec<u8>>)> {
        debug!("Registering pipeline invocation: {}", logical_pipeline_name);

        let request = RegisterPipelineInvocationRequest {
            logical_pipeline_name,
            pipeline_variant_policy,
            intermediates_ttl: Some(ProtoDuration { seconds: intermediates_ttl_seconds, nanos: 0 }),
            keyset_ids,
            authorized_logical_pipeline_policies: authorized_policies,
            include_keys_in_response: true,
        };

        let response = self
            .client
            .register_pipeline_invocation(request)
            .await
            .map_err(|status| {
                let error_msg = format!(
                    "Failed to register pipeline invocation - Code: {:?}, Message: {}",
                    status.code(),
                    status.message()
                );
                error!("{}", error_msg);
                anyhow!(error_msg)
            })?
            .into_inner();

        let invocation_id = response.invocation_id.clone();
        let keys_info: Vec<Vec<u8>> =
            response.keys.iter().map(|k| format!("Key ID: {:?}", k.key_id).into_bytes()).collect();

        info!(
            "Pipeline invocation registered with ID: {:?}",
            String::from_utf8_lossy(&invocation_id)
        );

        Ok((invocation_id, keys_info))
    }

    /// Authorizes a pipeline transform and retrieves encrypted decryption keys
    pub async fn authorize_transform(
        &mut self,
        invocation_id: Vec<u8>,
        pipeline_variant_policy: Vec<u8>,
        evidence: Option<Evidence>,
        endorsements: Option<Endorsements>,
    ) -> Result<(kms_proto::crypto_proto::oak::crypto::v1::EncryptedRequest, Vec<u8>)> {
        debug!(
            "Authorizing transform for invocation: {:?}",
            String::from_utf8_lossy(&invocation_id)
        );

        // Use provided evidence/endorsements - MUST be provided for real operation
        let final_evidence =
            evidence.ok_or_else(|| anyhow!("Evidence must be provided for authorize_transform"))?;

        let final_endorsements = endorsements
            .ok_or_else(|| anyhow!("Endorsements must be provided for authorize_transform"))?;

        let request = AuthorizeConfidentialTransformRequest {
            invocation_id,
            pipeline_variant_policy,
            evidence: Some(final_evidence),
            endorsements: Some(final_endorsements),
            tag: "tag".into(),
        };

        let response = self
            .client
            .authorize_confidential_transform(request)
            .await
            .map_err(|status| {
                let error_msg = format!(
                    "Failed to authorize transform - Code: {:?}, Message: {}",
                    status.code(),
                    status.message()
                );
                error!("{}", error_msg);
                anyhow!(error_msg)
            })?
            .into_inner();

        let protected_response = response
            .protected_response
            .clone()
            .ok_or_else(|| anyhow!("No protected response from KMS"))?;

        info!("Transform authorized, received encrypted protected response");

        Ok((protected_response, response.signing_key_endorsement))
    }

    /// Releases pipeline results and gets decryption keys
    pub async fn release_results(
        &mut self,
        release_tokens: Vec<(Vec<u8>, Vec<u8>)>, // (release_token, signing_key_endorsement)
    ) -> Result<Vec<Vec<u8>>> {
        debug!("Releasing {} pipeline results", release_tokens.len());

        let releasable_results = release_tokens
            .into_iter()
            .map(|(release_token, signing_key_endorsement)| ReleasableResult {
                release_token,
                signing_key_endorsement,
            })
            .collect();

        let request = ReleaseResultsRequest { releasable_results };

        let response = self
            .client
            .release_results(request)
            .await
            .context("Failed to release results")?
            .into_inner();

        info!("Released {} results", response.decryption_keys.len());

        Ok(response.decryption_keys)
    }
}
