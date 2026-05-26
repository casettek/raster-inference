use std::{cell::RefCell, collections::HashMap};

use anyhow::{anyhow, bail, Context, Result};
use serde::{de::DeserializeOwned, Serialize};

use crate::shared::artifacts::artifact_io::AuthRead;
use crate::shared::artifacts::authenticated_selection::{
    AuthenticatedSelector, VerifiedSelectedPayload,
};
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
pub struct VerifiedExternalSourceRead {
    commitment: String,
    request_key: Vec<u8>,
    response_payload: Vec<u8>,
}

impl VerifiedExternalSourceRead {
    pub fn from_selected_external(selected: VerifiedSelectedPayload) -> Result<Self> {
        let AuthenticatedSelector::ExternalRequest { request_key } = selected.selector() else {
            bail!("verified selected payload is not an external source response");
        };
        Ok(Self {
            commitment: selected.commitment().to_string(),
            request_key: request_key.clone(),
            response_payload: selected.bytes().to_vec(),
        })
    }

    pub fn commitment(&self) -> &str {
        &self.commitment
    }

    pub fn request_key(&self) -> &[u8] {
        &self.request_key
    }

    pub fn bytes(&self) -> &[u8] {
        &self.response_payload
    }

    pub fn deserialize<T: DeserializeOwned>(&self) -> Result<T> {
        postcard::from_bytes(self.bytes()).map_err(|error| {
            anyhow!(
                "failed to deserialize committed external source response from postcard bytes: {}",
                error
            )
        })
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

    pub fn read_verified(&self, request_key: &[u8]) -> Result<Option<VerifiedExternalSourceRead>> {
        let Some(selected) = self.read_authenticated(request_key)? else {
            return Ok(None);
        };
        Ok(Some(VerifiedExternalSourceRead::from_selected_external(
            selected,
        )?))
    }

    pub fn read_authenticated(
        &self,
        request_key: &[u8],
    ) -> Result<Option<VerifiedSelectedPayload>> {
        let Some(read) =
            read_optional_external_source_by_request_key(&self.source_ref, request_key)?
        else {
            return Ok(None);
        };
        verify_external_source_read(&self.source_ref, &read)?;
        let leaf = decode_source_leaf(read.payload())?;
        Ok(Some(VerifiedSelectedPayload::external_response(
            self.source_ref.id().source_name(),
            self.root(),
            leaf.request_key,
            leaf.response_payload,
            read.leaf_idx,
            read.payload,
            read.proof,
        )))
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
        let Some(read) = self.read_authenticated(&request_key)? else {
            return request.decode_missing_response();
        };
        request.decode_response(read.bytes())
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
            let leaf = decode_source_leaf(payload)?;
            if leaf.request_key == request_key {
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

pub fn read_authenticated_external_source(
    source_ref: &ExternalSourceRef,
    request_key: &[u8],
) -> Result<Option<VerifiedSelectedPayload>> {
    CommittedExternalSource::new(source_ref.clone()).read_authenticated(request_key)
}

pub fn read_authenticated_external_source_by_root(
    source_root: &str,
    request_key: &[u8],
) -> Result<Option<VerifiedSelectedPayload>> {
    CommittedExternalSource::from_root(source_root)?.read_authenticated(request_key)
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

pub fn postcard_request_key<T: Serialize + ?Sized>(
    request_kind: &str,
    payload: &T,
) -> Result<Vec<u8>> {
    let payload = postcard::to_allocvec(payload)?;
    external_request_key(request_kind, &payload)
}

pub fn postcard_response_payload<T: Serialize + ?Sized>(response: &T) -> Result<Vec<u8>> {
    Ok(postcard::to_allocvec(response)?)
}

pub fn postcard_external_source_entry<T: Serialize + ?Sized>(
    request_key: Vec<u8>,
    response: &T,
) -> Result<ExternalSourceEntry> {
    ExternalSourceEntry::new(request_key, postcard_response_payload(response)?)
}

pub fn decode_postcard_response<T: DeserializeOwned>(response_payload: &[u8]) -> Result<T> {
    postcard::from_bytes(response_payload)
        .map_err(|error| anyhow!("failed to deserialize committed external response: {error}"))
}

pub fn postcard_i32_vec_external_source_entry(
    request_key: Vec<u8>,
    values: impl IntoIterator<Item = i32>,
) -> Result<ExternalSourceEntry> {
    let values = values.into_iter().collect::<Vec<_>>();
    postcard_external_source_entry(request_key, &values)
}

pub fn decode_i32_vec_response(response_payload: &[u8]) -> Result<Vec<i32>> {
    decode_postcard_response(response_payload)
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
    Ok(postcard::to_allocvec(&ExternalSourceLeaf {
        request_key,
        response_payload,
    })?)
}

#[derive(Serialize, serde::Deserialize)]
struct ExternalSourceLeaf<'a> {
    #[serde(borrow)]
    request_key: &'a [u8],
    #[serde(borrow)]
    response_payload: &'a [u8],
}

struct DecodedExternalSourceLeaf {
    request_key: Vec<u8>,
    response_payload: Vec<u8>,
}

fn decode_source_leaf(payload: &[u8]) -> Result<DecodedExternalSourceLeaf> {
    let leaf: ExternalSourceLeaf<'_> =
        postcard::from_bytes(payload).context("invalid committed external source leaf")?;
    Ok(DecodedExternalSourceLeaf {
        request_key: leaf.request_key.to_vec(),
        response_payload: leaf.response_payload.to_vec(),
    })
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
        let response = decode_source_leaf(read.payload())
            .expect("leaf should decode")
            .response_payload;

        assert_eq!(response, b"response");
    }

    #[test]
    fn committed_source_verified_read_exposes_response_bytes() {
        reset_external_source_store();
        let request_key = external_request_key("test.request", b"a").expect("request key");
        let source_ref = register_external_source(
            ExternalSourceId::new("verified").expect("source id"),
            KIND,
            DOMAIN,
            vec![
                ExternalSourceEntry::new(request_key.clone(), b"response".to_vec()).expect("entry"),
            ],
        )
        .expect("source should register");
        let source = CommittedExternalSource::new(source_ref.clone());

        let read = source
            .read_verified(&request_key)
            .expect("verified read should succeed")
            .expect("response should exist");

        assert_eq!(read.commitment(), source_ref.root());
        assert_eq!(read.request_key(), request_key);
        assert_eq!(read.bytes(), b"response");
    }

    #[test]
    fn committed_source_authenticated_read_exposes_selector_payload_and_proof() {
        reset_external_source_store();
        let request_key = external_request_key("test.request", b"a").expect("request key");
        let source_ref = register_external_source(
            ExternalSourceId::new("authenticated").expect("source id"),
            KIND,
            DOMAIN,
            vec![
                ExternalSourceEntry::new(request_key.clone(), b"response".to_vec()).expect("entry"),
            ],
        )
        .expect("source should register");

        let selected = read_authenticated_external_source_by_root(source_ref.root(), &request_key)
            .expect("authenticated read should succeed")
            .expect("response should exist");

        assert_eq!(
            selected.source_kind(),
            &crate::shared::artifacts::authenticated_selection::AuthenticatedSourceKind::ExternalSource
        );
        assert_eq!(
            selected.selector(),
            &crate::shared::artifacts::authenticated_selection::AuthenticatedSelector::ExternalRequest {
                request_key: request_key.clone()
            }
        );
        assert_eq!(selected.source_name(), "authenticated");
        assert_eq!(selected.commitment(), source_ref.root());
        assert_eq!(selected.bytes(), b"response");
        assert_eq!(selected.proof().leaf_count(), 1);
        assert_ne!(selected.merkle_leaf_payload(), selected.bytes());
    }

    #[test]
    fn committed_source_authenticated_read_returns_none_for_missing_request() {
        reset_external_source_store();
        let request_key = external_request_key("test.request", b"a").expect("request key");
        let missing_key = external_request_key("test.request", b"missing").expect("request key");
        let source_ref = register_external_source(
            ExternalSourceId::new("missing-response").expect("source id"),
            KIND,
            DOMAIN,
            vec![
                ExternalSourceEntry::new(request_key.clone(), b"response".to_vec()).expect("entry"),
            ],
        )
        .expect("source should register");
        let source = CommittedExternalSource::new(source_ref);

        assert!(source
            .read_authenticated(&missing_key)
            .expect("missing request should not error")
            .is_none());
    }

    #[test]
    fn committed_source_authenticated_read_rejects_unknown_root() {
        reset_external_source_store();
        let request_key = external_request_key("test.request", b"a").expect("request key");

        let error = read_authenticated_external_source_by_root("missing-root", &request_key)
            .expect_err("unknown root should fail");

        assert!(error.to_string().contains("not registered"));
    }

    #[test]
    fn committed_source_authenticated_read_rejects_tampered_ref_root() {
        reset_external_source_store();
        let request_key = external_request_key("test.request", b"a").expect("request key");
        let mut source_ref = register_external_source(
            ExternalSourceId::new("tampered-ref").expect("source id"),
            KIND,
            DOMAIN,
            vec![
                ExternalSourceEntry::new(request_key.clone(), b"response".to_vec()).expect("entry"),
            ],
        )
        .expect("source should register");
        source_ref.root = "wrong-root".to_string();
        let source = CommittedExternalSource::new(source_ref);

        let error = source
            .read_authenticated(&request_key)
            .expect_err("tampered source ref should fail");

        assert!(error.to_string().contains("metadata mismatch"));
    }

    #[test]
    fn committed_source_authenticated_read_rejects_malformed_leaf_payload() {
        reset_external_source_store();
        let request_key = external_request_key("test.request", b"a").expect("request key");
        let source_ref = register_external_source_leaves(
            ExternalSourceId::new("malformed-leaf").expect("source id"),
            KIND,
            DOMAIN,
            vec![b"not-postcard".to_vec()],
        )
        .expect("source should register");
        let source = CommittedExternalSource::new(source_ref);

        let error = source
            .read_authenticated(&request_key)
            .expect_err("malformed leaf should fail");

        assert!(error
            .to_string()
            .contains("invalid committed external source leaf"));
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
