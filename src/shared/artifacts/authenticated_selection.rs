use anyhow::{anyhow, Result};
use serde::de::DeserializeOwned;

use crate::shared::artifacts::merkle::MerkleProof;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub enum AuthenticatedSourceKind {
    RasterArtifact,
    ExternalSource,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub enum AuthenticatedSelector {
    ArtifactLeaf { leaf_idx: usize },
    ExternalRequest { request_key: Vec<u8> },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct VerifiedSelectedPayload {
    source_kind: AuthenticatedSourceKind,
    source_name: String,
    commitment: String,
    selector: AuthenticatedSelector,
    selected_payload: Vec<u8>,
    merkle_leaf_idx: usize,
    merkle_leaf_payload: Vec<u8>,
    proof: MerkleProof,
}

impl VerifiedSelectedPayload {
    pub fn artifact_leaf(
        source_name: impl Into<String>,
        commitment: impl Into<String>,
        leaf_idx: usize,
        payload: Vec<u8>,
        proof: MerkleProof,
    ) -> Self {
        Self {
            source_kind: AuthenticatedSourceKind::RasterArtifact,
            source_name: source_name.into(),
            commitment: commitment.into(),
            selector: AuthenticatedSelector::ArtifactLeaf { leaf_idx },
            selected_payload: payload.clone(),
            merkle_leaf_idx: leaf_idx,
            merkle_leaf_payload: payload,
            proof,
        }
    }

    pub fn external_response(
        source_name: impl Into<String>,
        commitment: impl Into<String>,
        request_key: Vec<u8>,
        response_payload: Vec<u8>,
        merkle_leaf_idx: usize,
        merkle_leaf_payload: Vec<u8>,
        proof: MerkleProof,
    ) -> Self {
        Self {
            source_kind: AuthenticatedSourceKind::ExternalSource,
            source_name: source_name.into(),
            commitment: commitment.into(),
            selector: AuthenticatedSelector::ExternalRequest { request_key },
            selected_payload: response_payload,
            merkle_leaf_idx,
            merkle_leaf_payload,
            proof,
        }
    }

    pub fn source_kind(&self) -> &AuthenticatedSourceKind {
        &self.source_kind
    }

    pub fn source_name(&self) -> &str {
        &self.source_name
    }

    pub fn commitment(&self) -> &str {
        &self.commitment
    }

    pub fn selector(&self) -> &AuthenticatedSelector {
        &self.selector
    }

    pub fn bytes(&self) -> &[u8] {
        &self.selected_payload
    }

    pub fn merkle_leaf_idx(&self) -> usize {
        self.merkle_leaf_idx
    }

    pub fn merkle_leaf_payload(&self) -> &[u8] {
        &self.merkle_leaf_payload
    }

    pub fn proof(&self) -> &MerkleProof {
        &self.proof
    }

    pub fn deserialize<T: DeserializeOwned>(&self) -> Result<T> {
        postcard::from_bytes(self.bytes()).map_err(|error| {
            anyhow!(
                "failed to deserialize authenticated selected payload from postcard bytes: {}",
                error
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::artifacts::merkle::MerkleProof;

    fn empty_proof() -> MerkleProof {
        MerkleProof::new(1, Vec::new()).expect("proof")
    }

    #[test]
    fn artifact_payload_carries_leaf_selector_and_proof() {
        let payload = VerifiedSelectedPayload::artifact_leaf(
            "tokens",
            "root",
            2,
            vec![1, 2, 3],
            empty_proof(),
        );

        assert_eq!(
            payload.source_kind(),
            &AuthenticatedSourceKind::RasterArtifact
        );
        assert_eq!(payload.source_name(), "tokens");
        assert_eq!(payload.commitment(), "root");
        assert_eq!(
            payload.selector(),
            &AuthenticatedSelector::ArtifactLeaf { leaf_idx: 2 }
        );
        assert_eq!(payload.bytes(), &[1, 2, 3]);
        assert_eq!(payload.merkle_leaf_payload(), &[1, 2, 3]);
        assert_eq!(payload.proof().leaf_count(), 1);
    }

    #[test]
    fn external_payload_carries_request_selector_and_response_bytes() {
        let payload = VerifiedSelectedPayload::external_response(
            "model",
            "root",
            vec![9],
            vec![1, 2],
            4,
            vec![9, 1, 2],
            empty_proof(),
        );

        assert_eq!(
            payload.source_kind(),
            &AuthenticatedSourceKind::ExternalSource
        );
        assert_eq!(
            payload.selector(),
            &AuthenticatedSelector::ExternalRequest {
                request_key: vec![9]
            }
        );
        assert_eq!(payload.bytes(), &[1, 2]);
        assert_eq!(payload.merkle_leaf_idx(), 4);
        assert_eq!(payload.merkle_leaf_payload(), &[9, 1, 2]);
    }

    #[test]
    fn malformed_postcard_payload_fails_during_typed_decode() {
        let payload =
            VerifiedSelectedPayload::artifact_leaf("tokens", "root", 0, vec![0xff], empty_proof());

        assert!(payload.deserialize::<u32>().is_err());
    }
}
