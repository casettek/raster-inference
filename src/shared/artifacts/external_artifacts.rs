use std::{cell::RefCell, collections::HashMap};

use anyhow::{anyhow, bail, Context, Result};

use crate::shared::artifacts::artifact_io::AuthRead;
use crate::shared::artifacts::merkle::{
    merkle_proof, merkle_root, verify_merkle_proof, MerkleProof,
};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub struct ExternalSourceId {
    source_name: String,
}

impl ExternalSourceId {
    pub fn new(source_name: impl Into<String>) -> Result<Self> {
        let source_name = source_name.into();
        if source_name.is_empty() {
            bail!("external source id requires a non-empty source name");
        }
        Ok(Self { source_name })
    }

    pub fn source_name(&self) -> &str {
        &self.source_name
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub struct ExternalSourceMetadata {
    kind: String,
    domain: String,
    leaf_count: usize,
}

impl ExternalSourceMetadata {
    pub fn new(
        kind: impl Into<String>,
        domain: impl Into<String>,
        leaf_count: usize,
    ) -> Result<Self> {
        let kind = kind.into();
        if kind.is_empty() {
            bail!("external source metadata requires a non-empty kind");
        }
        let domain = domain.into();
        if domain.is_empty() {
            bail!("external source metadata requires a non-empty domain");
        }
        Ok(Self {
            kind,
            domain,
            leaf_count,
        })
    }

    pub fn kind(&self) -> &str {
        &self.kind
    }

    pub fn domain(&self) -> &str {
        &self.domain
    }

    pub fn leaf_count(&self) -> usize {
        self.leaf_count
    }

    fn domain_bytes(&self) -> &[u8] {
        self.domain.as_bytes()
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub struct ExternalSourceRef {
    id: ExternalSourceId,
    metadata: ExternalSourceMetadata,
    root: String,
}

impl ExternalSourceRef {
    pub fn id(&self) -> &ExternalSourceId {
        &self.id
    }

    pub fn metadata(&self) -> &ExternalSourceMetadata {
        &self.metadata
    }

    pub fn kind(&self) -> &str {
        self.metadata.kind()
    }

    pub fn root(&self) -> &str {
        &self.root
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct ExternalSourceRead {
    leaf_idx: usize,
    payload: Vec<u8>,
    proof: MerkleProof,
}

impl ExternalSourceRead {
    pub fn leaf_idx(&self) -> usize {
        self.leaf_idx
    }

    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    pub fn proof(&self) -> &MerkleProof {
        &self.proof
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalSourceEntry {
    request_key: Vec<u8>,
    response_payload: Vec<u8>,
}

impl ExternalSourceEntry {
    pub fn new(request_key: Vec<u8>, response_payload: Vec<u8>) -> Result<Self> {
        if request_key.is_empty() {
            bail!("external source entry requires a non-empty request key");
        }
        Ok(Self {
            request_key,
            response_payload,
        })
    }
}

#[derive(Debug, Clone)]
struct StoredExternalSource {
    metadata: ExternalSourceMetadata,
    leaves: Vec<Vec<u8>>,
}

#[derive(Debug, Default, Clone)]
pub struct ExternalSourceStore {
    sources: HashMap<ExternalSourceId, StoredExternalSource>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub struct CommittedExternalSource {
    source_ref: ExternalSourceRef,
}

impl CommittedExternalSource {
    pub fn new(source_ref: ExternalSourceRef) -> Self {
        Self { source_ref }
    }

    pub fn from_root(root: &str) -> Result<Self> {
        Ok(Self {
            source_ref: external_source_ref_for_root(root)?,
        })
    }

    pub fn source_ref(&self) -> &ExternalSourceRef {
        &self.source_ref
    }

    pub fn root(&self) -> &str {
        self.source_ref.root()
    }
}

pub trait CommittedExternalRequest {
    type Output;

    fn request_key(&self) -> Result<Vec<u8>>;
    fn decode_response(&self, response_payload: &[u8]) -> Result<Self::Output>;

    fn decode_missing_response(&self) -> Result<Self::Output> {
        bail!("external source response is not committed")
    }
}

impl<Request> AuthRead<Request> for CommittedExternalSource
where
    Request: CommittedExternalRequest,
{
    type Output = Request::Output;

    fn auth_read(&self, request: Request) -> Result<Self::Output> {
        let request_key = request.request_key()?;
        let Some(read) =
            read_optional_external_source_by_request_key(&self.source_ref, &request_key)?
        else {
            return request.decode_missing_response();
        };
        verify_external_source_read(&self.source_ref, &read)?;
        let (_, response_payload) = decode_source_leaf(read.payload())?;
        request.decode_response(&response_payload)
    }
}

thread_local! {
    static EXTERNAL_SOURCE_STORE: RefCell<ExternalSourceStore> =
        RefCell::new(ExternalSourceStore::new());
}

impl ExternalSourceStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_source(
        &mut self,
        id: ExternalSourceId,
        kind: impl Into<String>,
        domain: impl Into<String>,
        entries: Vec<ExternalSourceEntry>,
    ) -> Result<ExternalSourceRef> {
        let leaves = source_leaves(entries)?;
        self.register_source_leaves(id, kind, domain, leaves)
    }

    pub fn register_source_leaves(
        &mut self,
        id: ExternalSourceId,
        kind: impl Into<String>,
        domain: impl Into<String>,
        leaves: Vec<Vec<u8>>,
    ) -> Result<ExternalSourceRef> {
        let metadata = ExternalSourceMetadata::new(kind, domain, leaves.len())?;
        let source_ref = external_source_ref(id.clone(), &metadata, &leaves);

        match self.sources.get(&id) {
            Some(existing) => {
                let existing_ref = external_source_ref(id, &existing.metadata, &existing.leaves);
                if existing_ref != source_ref {
                    bail!(
                        "external source id {} is already registered with a different commitment",
                        existing_ref.id().source_name()
                    );
                }
                Ok(existing_ref)
            }
            None => {
                self.sources
                    .insert(id, StoredExternalSource { metadata, leaves });
                Ok(source_ref)
            }
        }
    }

    pub fn read_by_request_key(
        &self,
        source_ref: &ExternalSourceRef,
        request_key: &[u8],
    ) -> Result<ExternalSourceRead> {
        self.read_optional_by_request_key(source_ref, request_key)?
            .ok_or_else(|| {
                anyhow!(
                    "external source {} has no committed response for request key",
                    source_ref.id().source_name()
                )
            })
    }

    pub fn read_optional_by_request_key(
        &self,
        source_ref: &ExternalSourceRef,
        request_key: &[u8],
    ) -> Result<Option<ExternalSourceRead>> {
        let source = self.source(source_ref)?;
        for (leaf_idx, payload) in source.leaves.iter().enumerate() {
            let (leaf_request_key, _) = decode_source_leaf(payload)?;
            if leaf_request_key == request_key {
                let proof = merkle_proof(source.metadata.domain_bytes(), &source.leaves, leaf_idx)?;
                return Ok(Some(ExternalSourceRead {
                    leaf_idx,
                    payload: payload.clone(),
                    proof,
                }));
            }
        }

        Ok(None)
    }

    pub fn source_ref_for_root(&self, root: &str) -> Result<ExternalSourceRef> {
        let mut matches = self.sources.iter().filter_map(|(id, source)| {
            let actual_root = source_root(&source.metadata, &source.leaves);
            (actual_root == root).then_some((id, source))
        });
        let Some((id, source)) = matches.next() else {
            bail!("external source root {root} is not registered");
        };
        if matches.next().is_some() {
            bail!("external source root {root} matches multiple registered sources");
        }
        Ok(ExternalSourceRef {
            id: id.clone(),
            metadata: source.metadata.clone(),
            root: root.to_string(),
        })
    }

    fn source(&self, source_ref: &ExternalSourceRef) -> Result<&StoredExternalSource> {
        let source = self.sources.get(source_ref.id()).ok_or_else(|| {
            anyhow!(
                "external source {} is not registered",
                source_ref.id().source_name()
            )
        })?;
        let expected =
            external_source_ref(source_ref.id().clone(), &source.metadata, &source.leaves);
        if expected != *source_ref {
            bail!(
                "external source {} metadata mismatch",
                source_ref.id().source_name()
            );
        }
        Ok(source)
    }
}

pub fn reset_external_source_store() {
    EXTERNAL_SOURCE_STORE.with(|store_ref| *store_ref.borrow_mut() = ExternalSourceStore::new());
}

pub fn register_external_source(
    id: ExternalSourceId,
    kind: impl Into<String>,
    domain: impl Into<String>,
    entries: Vec<ExternalSourceEntry>,
) -> Result<ExternalSourceRef> {
    EXTERNAL_SOURCE_STORE.with(|store_ref| {
        let mut store = store_ref.borrow_mut();
        store.register_source(id, kind, domain, entries)
    })
}

pub fn register_external_source_leaves(
    id: ExternalSourceId,
    kind: impl Into<String>,
    domain: impl Into<String>,
    leaves: Vec<Vec<u8>>,
) -> Result<ExternalSourceRef> {
    EXTERNAL_SOURCE_STORE.with(|store_ref| {
        let mut store = store_ref.borrow_mut();
        store.register_source_leaves(id, kind, domain, leaves)
    })
}

pub fn read_external_source_by_request_key(
    source_ref: &ExternalSourceRef,
    request_key: &[u8],
) -> Result<ExternalSourceRead> {
    EXTERNAL_SOURCE_STORE.with(|store_ref| {
        let store = store_ref.borrow();
        store.read_by_request_key(source_ref, request_key)
    })
}

pub fn read_optional_external_source_by_request_key(
    source_ref: &ExternalSourceRef,
    request_key: &[u8],
) -> Result<Option<ExternalSourceRead>> {
    EXTERNAL_SOURCE_STORE.with(|store_ref| {
        let store = store_ref.borrow();
        store.read_optional_by_request_key(source_ref, request_key)
    })
}

pub fn external_source_ref_for_root(root: &str) -> Result<ExternalSourceRef> {
    EXTERNAL_SOURCE_STORE.with(|store_ref| {
        let store = store_ref.borrow();
        store.source_ref_for_root(root)
    })
}

pub fn verify_external_source_read(
    source_ref: &ExternalSourceRef,
    read: &ExternalSourceRead,
) -> Result<()> {
    if read.leaf_idx >= source_ref.metadata.leaf_count() {
        bail!(
            "external source read leaf {} is out of range for {} leaves",
            read.leaf_idx,
            source_ref.metadata.leaf_count()
        );
    }
    verify_merkle_proof(
        source_ref.metadata.domain_bytes(),
        source_ref.root(),
        read.leaf_idx,
        &read.payload,
        &read.proof,
    )
}

pub fn external_request_key(request_kind: &str, payload: &[u8]) -> Result<Vec<u8>> {
    if request_kind.is_empty() {
        bail!("external source request kind must be non-empty");
    }
    let mut key = Vec::with_capacity(16 + request_kind.len() + payload.len());
    key.extend_from_slice(&(request_kind.len() as u64).to_le_bytes());
    key.extend_from_slice(request_kind.as_bytes());
    key.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    key.extend_from_slice(payload);
    Ok(key)
}

fn source_leaves(mut entries: Vec<ExternalSourceEntry>) -> Result<Vec<Vec<u8>>> {
    entries.sort_by(|left, right| left.request_key.cmp(&right.request_key));
    for pair in entries.windows(2) {
        if pair[0].request_key == pair[1].request_key {
            bail!("external source has duplicate request key");
        }
    }
    entries
        .iter()
        .map(|entry| source_leaf(&entry.request_key, &entry.response_payload))
        .collect()
}

fn source_leaf(request_key: &[u8], response_payload: &[u8]) -> Result<Vec<u8>> {
    let mut payload = Vec::with_capacity(16 + request_key.len() + response_payload.len());
    payload.extend_from_slice(&(request_key.len() as u64).to_le_bytes());
    payload.extend_from_slice(request_key);
    payload.extend_from_slice(&(response_payload.len() as u64).to_le_bytes());
    payload.extend_from_slice(response_payload);
    Ok(payload)
}

fn decode_source_leaf(payload: &[u8]) -> Result<(Vec<u8>, Vec<u8>)> {
    let (request_len, rest) = read_len_prefixed(payload).context("invalid request key payload")?;
    let (response_len, remainder) = read_len_prefixed(rest).context("invalid response payload")?;
    if !remainder.is_empty() {
        bail!("external source leaf has trailing bytes");
    }
    Ok((request_len, response_len))
}

fn read_len_prefixed(payload: &[u8]) -> Result<(Vec<u8>, &[u8])> {
    if payload.len() < 8 {
        bail!("length-prefixed payload is too short");
    }
    let len = u64::from_le_bytes(
        payload[0..8]
            .try_into()
            .expect("slice length checked above"),
    ) as usize;
    let bytes = payload
        .get(8..8 + len)
        .ok_or_else(|| anyhow!("length-prefixed payload length mismatch"))?;
    Ok((bytes.to_vec(), &payload[8 + len..]))
}

fn external_source_ref(
    id: ExternalSourceId,
    metadata: &ExternalSourceMetadata,
    leaves: &[Vec<u8>],
) -> ExternalSourceRef {
    ExternalSourceRef {
        id,
        metadata: metadata.clone(),
        root: source_root(metadata, leaves),
    }
}

fn source_root(metadata: &ExternalSourceMetadata, leaves: &[Vec<u8>]) -> String {
    merkle_root(metadata.domain_bytes(), leaves)
}

#[cfg(test)]
mod tests {
    use super::*;

    const KIND: &str = "test_source";
    const DOMAIN: &str = "raster-external-source-test-merkle-v1";

    #[test]
    fn committed_source_reads_verify_against_root() {
        reset_external_source_store();
        let request_key = external_request_key("test.request", b"a").expect("request key");
        let source_ref = register_external_source(
            ExternalSourceId::new("fixture").expect("source id"),
            KIND,
            DOMAIN,
            vec![
                ExternalSourceEntry::new(request_key.clone(), b"response".to_vec()).expect("entry"),
            ],
        )
        .expect("source should register");

        let read =
            read_external_source_by_request_key(&source_ref, &request_key).expect("source read");
        verify_external_source_read(&source_ref, &read).expect("read should verify");
        let (_, response) = decode_source_leaf(read.payload()).expect("leaf should decode");

        assert_eq!(response, b"response");
    }

    #[test]
    fn committed_source_rejects_tampered_root() {
        reset_external_source_store();
        let request_key = external_request_key("test.request", b"a").expect("request key");
        let mut source_ref = register_external_source(
            ExternalSourceId::new("fixture").expect("source id"),
            KIND,
            DOMAIN,
            vec![
                ExternalSourceEntry::new(request_key.clone(), b"response".to_vec()).expect("entry"),
            ],
        )
        .expect("source should register");
        let read =
            read_external_source_by_request_key(&source_ref, &request_key).expect("source read");
        source_ref.root = "not-the-root".to_string();

        assert!(verify_external_source_read(&source_ref, &read).is_err());
    }
}
