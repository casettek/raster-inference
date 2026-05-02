use std::collections::HashMap;

use anyhow::{anyhow, bail, Result};
use sha2::{Digest, Sha256};

use crate::raster_authoring::AuthRead;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub struct RasterTokenizerSequenceId {
    source_name: String,
}

impl RasterTokenizerSequenceId {
    pub fn new(source_name: impl Into<String>) -> Result<Self> {
        let source_name = source_name.into();
        if source_name.is_empty() {
            bail!("raster tokenizer sequence id requires a non-empty source name");
        }
        Ok(Self { source_name })
    }

    pub fn source_name(&self) -> &str {
        &self.source_name
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub struct RasterBpePieceSequenceRef {
    id: RasterTokenizerSequenceId,
    piece_count: usize,
    det_commitment: String,
}

impl RasterBpePieceSequenceRef {
    pub fn new(
        id: RasterTokenizerSequenceId,
        piece_count: usize,
        det_commitment: impl Into<String>,
    ) -> Result<Self> {
        let det_commitment = det_commitment.into();
        if det_commitment.is_empty() {
            bail!("raster BPE piece sequence ref requires a non-empty deterministic commitment");
        }
        Ok(Self {
            id,
            piece_count,
            det_commitment,
        })
    }

    pub fn id(&self) -> &RasterTokenizerSequenceId {
        &self.id
    }

    pub fn piece_count(&self) -> usize {
        self.piece_count
    }

    pub fn det_commitment(&self) -> &str {
        &self.det_commitment
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub struct RasterTokenIdSequenceRef {
    id: RasterTokenizerSequenceId,
    token_count: usize,
    det_commitment: String,
}

impl RasterTokenIdSequenceRef {
    pub fn new(
        id: RasterTokenizerSequenceId,
        token_count: usize,
        det_commitment: impl Into<String>,
    ) -> Result<Self> {
        let det_commitment = det_commitment.into();
        if det_commitment.is_empty() {
            bail!("raster token-id sequence ref requires a non-empty deterministic commitment");
        }
        Ok(Self {
            id,
            token_count,
            det_commitment,
        })
    }

    pub fn id(&self) -> &RasterTokenizerSequenceId {
        &self.id
    }

    pub fn token_count(&self) -> usize {
        self.token_count
    }

    pub fn det_commitment(&self) -> &str {
        &self.det_commitment
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterBpePieceRequest {
    pub pieces_ref: RasterBpePieceSequenceRef,
    pub piece_idx: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterBpePairRequest {
    pub pieces_ref: RasterBpePieceSequenceRef,
    pub pair_idx: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterBpePair {
    pub left: String,
    pub right: String,
}

pub trait RasterTokenizerPieceSource {
    fn read_bpe_piece(&self, request: RasterBpePieceRequest) -> Result<String>;
    fn read_bpe_pair(&self, request: RasterBpePairRequest) -> Result<RasterBpePair>;
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterBpePieceSequenceBuilderRef {
    id: RasterTokenizerSequenceId,
    expected_piece_count: usize,
    pieces_written: usize,
    running_commitment: String,
}

impl RasterBpePieceSequenceBuilderRef {
    pub fn id(&self) -> &RasterTokenizerSequenceId {
        &self.id
    }

    pub fn expected_piece_count(&self) -> usize {
        self.expected_piece_count
    }

    pub fn pieces_written(&self) -> usize {
        self.pieces_written
    }

    pub fn running_commitment(&self) -> &str {
        &self.running_commitment
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterTokenIdSequenceBuilderRef {
    id: RasterTokenizerSequenceId,
    expected_token_count: usize,
    token_ids_written: usize,
    running_commitment: String,
}

impl RasterTokenIdSequenceBuilderRef {
    pub fn id(&self) -> &RasterTokenizerSequenceId {
        &self.id
    }

    pub fn expected_token_count(&self) -> usize {
        self.expected_token_count
    }

    pub fn token_ids_written(&self) -> usize {
        self.token_ids_written
    }

    pub fn running_commitment(&self) -> &str {
        &self.running_commitment
    }
}

#[derive(Debug, Clone)]
struct PieceBuilderState {
    expected_piece_count: usize,
    pieces: Vec<String>,
    finalized: bool,
}

#[derive(Debug, Clone)]
struct TokenIdBuilderState {
    expected_token_count: usize,
    token_ids: Vec<u32>,
    finalized: bool,
}

#[derive(Debug, Default, Clone)]
pub struct AuthenticatedRasterTokenizerStore {
    piece_sequences: HashMap<RasterTokenizerSequenceId, Vec<String>>,
    token_id_sequences: HashMap<RasterTokenizerSequenceId, Vec<u32>>,
    piece_builders: HashMap<RasterTokenizerSequenceId, PieceBuilderState>,
    token_id_builders: HashMap<RasterTokenizerSequenceId, TokenIdBuilderState>,
}

impl AuthenticatedRasterTokenizerStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert_bpe_piece_sequence(
        &mut self,
        id: RasterTokenizerSequenceId,
        pieces: Vec<String>,
    ) -> Result<RasterBpePieceSequenceRef> {
        let commitment = build_bpe_piece_sequence_commitment(&pieces);
        let reference = RasterBpePieceSequenceRef::new(id.clone(), pieces.len(), commitment)?;
        self.ensure_id_available(&id)?;
        self.piece_sequences.insert(id, pieces);
        Ok(reference)
    }

    pub fn start_bpe_piece_sequence_builder(
        &mut self,
        id: RasterTokenizerSequenceId,
        expected_piece_count: usize,
    ) -> Result<RasterBpePieceSequenceBuilderRef> {
        self.ensure_id_available(&id)?;
        let builder = PieceBuilderState {
            expected_piece_count,
            pieces: Vec::with_capacity(expected_piece_count),
            finalized: false,
        };
        let builder_ref = RasterBpePieceSequenceBuilderRef {
            id: id.clone(),
            expected_piece_count,
            pieces_written: 0,
            running_commitment: piece_builder_running_commitment(&builder),
        };
        self.piece_builders.insert(id, builder);
        Ok(builder_ref)
    }

    pub fn append_bpe_piece(
        &mut self,
        builder_ref: &mut RasterBpePieceSequenceBuilderRef,
        piece_idx: usize,
        piece: String,
    ) -> Result<()> {
        let builder = self.piece_builder_mut(builder_ref)?;
        if piece_idx != builder.pieces.len() {
            bail!("BPE piece builder duplicate or skipped piece {piece_idx}");
        }
        if piece_idx >= builder.expected_piece_count {
            bail!(
                "BPE piece builder piece {piece_idx} is out of range for {} pieces",
                builder.expected_piece_count
            );
        }
        builder.pieces.push(piece);
        builder_ref.pieces_written = builder.pieces.len();
        builder_ref.running_commitment = piece_builder_running_commitment(builder);
        Ok(())
    }

    pub fn finalize_bpe_piece_sequence_builder(
        &mut self,
        builder_ref: RasterBpePieceSequenceBuilderRef,
    ) -> Result<RasterBpePieceSequenceRef> {
        let builder = self.piece_builders.get(builder_ref.id()).ok_or_else(|| {
            anyhow!(
                "raster BPE piece builder {} is not registered",
                builder_ref.id().source_name()
            )
        })?;
        validate_piece_builder_ref(&builder_ref, builder)?;
        if builder.pieces.len() != builder.expected_piece_count {
            bail!(
                "BPE piece builder finalized with {} pieces, expected {}",
                builder.pieces.len(),
                builder.expected_piece_count
            );
        }
        let builder = self
            .piece_builders
            .remove(builder_ref.id())
            .expect("builder existence checked before removal");
        self.insert_bpe_piece_sequence(builder_ref.id, builder.pieces)
    }

    pub fn materialize_bpe_pieces(
        &self,
        pieces_ref: &RasterBpePieceSequenceRef,
    ) -> Result<Vec<String>> {
        let pieces = self.piece_sequence(pieces_ref)?.clone();
        ensure_commitment(
            pieces_ref.det_commitment(),
            &build_bpe_piece_sequence_commitment(&pieces),
            "BPE piece sequence",
        )?;
        Ok(pieces)
    }

    pub fn start_token_id_sequence_builder(
        &mut self,
        id: RasterTokenizerSequenceId,
        expected_token_count: usize,
    ) -> Result<RasterTokenIdSequenceBuilderRef> {
        self.ensure_id_available(&id)?;
        let builder = TokenIdBuilderState {
            expected_token_count,
            token_ids: Vec::with_capacity(expected_token_count),
            finalized: false,
        };
        let builder_ref = RasterTokenIdSequenceBuilderRef {
            id: id.clone(),
            expected_token_count,
            token_ids_written: 0,
            running_commitment: token_id_builder_running_commitment(&builder),
        };
        self.token_id_builders.insert(id, builder);
        Ok(builder_ref)
    }

    pub fn append_token_id(
        &mut self,
        builder_ref: &mut RasterTokenIdSequenceBuilderRef,
        token_idx: usize,
        token_id: u32,
    ) -> Result<()> {
        let builder = self.token_id_builder_mut(builder_ref)?;
        if token_idx != builder.token_ids.len() {
            bail!("token-id builder duplicate or skipped token {token_idx}");
        }
        if token_idx >= builder.expected_token_count {
            bail!(
                "token-id builder token {token_idx} is out of range for {} tokens",
                builder.expected_token_count
            );
        }
        builder.token_ids.push(token_id);
        builder_ref.token_ids_written = builder.token_ids.len();
        builder_ref.running_commitment = token_id_builder_running_commitment(builder);
        Ok(())
    }

    pub fn finalize_token_id_sequence_builder(
        &mut self,
        builder_ref: RasterTokenIdSequenceBuilderRef,
    ) -> Result<RasterTokenIdSequenceRef> {
        let builder = self
            .token_id_builders
            .get(builder_ref.id())
            .ok_or_else(|| {
                anyhow!(
                    "raster token-id builder {} is not registered",
                    builder_ref.id().source_name()
                )
            })?;
        validate_token_id_builder_ref(&builder_ref, builder)?;
        if builder.token_ids.len() != builder.expected_token_count {
            bail!(
                "token-id builder finalized with {} tokens, expected {}",
                builder.token_ids.len(),
                builder.expected_token_count
            );
        }
        let builder = self
            .token_id_builders
            .remove(builder_ref.id())
            .expect("builder existence checked before removal");
        let commitment = build_token_id_sequence_commitment(&builder.token_ids);
        let reference = RasterTokenIdSequenceRef::new(
            builder_ref.id.clone(),
            builder.token_ids.len(),
            commitment,
        )?;
        self.token_id_sequences
            .insert(builder_ref.id, builder.token_ids);
        Ok(reference)
    }

    pub fn materialize_token_ids(
        &self,
        token_ids_ref: &RasterTokenIdSequenceRef,
    ) -> Result<Vec<u32>> {
        let token_ids = self
            .token_id_sequences
            .get(token_ids_ref.id())
            .ok_or_else(|| {
                anyhow!(
                    "raster token-id sequence ref {} is not registered",
                    token_ids_ref.id().source_name()
                )
            })?;
        if token_ids.len() != token_ids_ref.token_count() {
            bail!("stored token-id sequence length mismatch");
        }
        ensure_commitment(
            token_ids_ref.det_commitment(),
            &build_token_id_sequence_commitment(token_ids),
            "token-id sequence",
        )?;
        Ok(token_ids.clone())
    }

    fn ensure_id_available(&self, id: &RasterTokenizerSequenceId) -> Result<()> {
        if self.piece_sequences.contains_key(id)
            || self.token_id_sequences.contains_key(id)
            || self.piece_builders.contains_key(id)
            || self.token_id_builders.contains_key(id)
        {
            bail!(
                "raster tokenizer sequence id {} is already registered",
                id.source_name()
            );
        }
        Ok(())
    }

    fn piece_sequence(&self, pieces_ref: &RasterBpePieceSequenceRef) -> Result<&Vec<String>> {
        let pieces = self.piece_sequences.get(pieces_ref.id()).ok_or_else(|| {
            anyhow!(
                "raster BPE piece sequence ref {} is not registered",
                pieces_ref.id().source_name()
            )
        })?;
        if pieces.len() != pieces_ref.piece_count() {
            bail!("stored BPE piece sequence length mismatch");
        }
        ensure_commitment(
            pieces_ref.det_commitment(),
            &build_bpe_piece_sequence_commitment(pieces),
            "BPE piece sequence",
        )?;
        Ok(pieces)
    }

    fn piece_builder_mut(
        &mut self,
        builder_ref: &RasterBpePieceSequenceBuilderRef,
    ) -> Result<&mut PieceBuilderState> {
        let builder = self
            .piece_builders
            .get_mut(builder_ref.id())
            .ok_or_else(|| {
                anyhow!(
                    "raster BPE piece builder {} is not registered",
                    builder_ref.id().source_name()
                )
            })?;
        if builder.finalized {
            bail!(
                "raster BPE piece builder {} is already finalized",
                builder_ref.id().source_name()
            );
        }
        validate_piece_builder_ref(builder_ref, builder)?;
        Ok(builder)
    }

    fn token_id_builder_mut(
        &mut self,
        builder_ref: &RasterTokenIdSequenceBuilderRef,
    ) -> Result<&mut TokenIdBuilderState> {
        let builder = self
            .token_id_builders
            .get_mut(builder_ref.id())
            .ok_or_else(|| {
                anyhow!(
                    "raster token-id builder {} is not registered",
                    builder_ref.id().source_name()
                )
            })?;
        if builder.finalized {
            bail!(
                "raster token-id builder {} is already finalized",
                builder_ref.id().source_name()
            );
        }
        validate_token_id_builder_ref(builder_ref, builder)?;
        Ok(builder)
    }
}

impl RasterTokenizerPieceSource for AuthenticatedRasterTokenizerStore {
    fn read_bpe_piece(&self, request: RasterBpePieceRequest) -> Result<String> {
        let pieces = self.piece_sequence(&request.pieces_ref)?;
        let piece = pieces.get(request.piece_idx).ok_or_else(|| {
            anyhow!(
                "BPE piece {} is out of range for {} pieces",
                request.piece_idx,
                pieces.len()
            )
        })?;
        Ok(piece.clone())
    }

    fn read_bpe_pair(&self, request: RasterBpePairRequest) -> Result<RasterBpePair> {
        let pieces = self.piece_sequence(&request.pieces_ref)?;
        if request.pair_idx + 1 >= pieces.len() {
            bail!(
                "BPE pair {} is out of range for {} pieces",
                request.pair_idx,
                pieces.len()
            );
        }
        Ok(RasterBpePair {
            left: pieces[request.pair_idx].clone(),
            right: pieces[request.pair_idx + 1].clone(),
        })
    }
}

impl AuthRead<RasterBpePieceRequest> for AuthenticatedRasterTokenizerStore {
    type Output = String;

    fn auth_read(&self, request: RasterBpePieceRequest) -> Result<Self::Output> {
        self.read_bpe_piece(request)
    }
}

impl AuthRead<RasterBpePairRequest> for AuthenticatedRasterTokenizerStore {
    type Output = RasterBpePair;

    fn auth_read(&self, request: RasterBpePairRequest) -> Result<Self::Output> {
        self.read_bpe_pair(request)
    }
}

pub fn build_bpe_piece_sequence_commitment(pieces: &[String]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"raster-tokenizer-bpe-pieces-v1");
    update_pieces(&mut hasher, pieces);
    hex_digest(hasher.finalize())
}

pub fn build_token_id_sequence_commitment(token_ids: &[u32]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"raster-tokenizer-token-ids-v1");
    update_token_ids(&mut hasher, token_ids);
    hex_digest(hasher.finalize())
}

fn piece_builder_running_commitment(builder: &PieceBuilderState) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"raster-tokenizer-bpe-piece-builder-v1");
    hasher.update((builder.expected_piece_count as u64).to_le_bytes());
    update_pieces(&mut hasher, &builder.pieces);
    hex_digest(hasher.finalize())
}

fn token_id_builder_running_commitment(builder: &TokenIdBuilderState) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"raster-tokenizer-token-id-builder-v1");
    hasher.update((builder.expected_token_count as u64).to_le_bytes());
    update_token_ids(&mut hasher, &builder.token_ids);
    hex_digest(hasher.finalize())
}

fn validate_piece_builder_ref(
    builder_ref: &RasterBpePieceSequenceBuilderRef,
    builder: &PieceBuilderState,
) -> Result<()> {
    if builder_ref.expected_piece_count != builder.expected_piece_count
        || builder_ref.pieces_written != builder.pieces.len()
        || builder_ref.running_commitment != piece_builder_running_commitment(builder)
    {
        bail!(
            "raster BPE piece builder {} metadata mismatch",
            builder_ref.id().source_name()
        );
    }
    Ok(())
}

fn validate_token_id_builder_ref(
    builder_ref: &RasterTokenIdSequenceBuilderRef,
    builder: &TokenIdBuilderState,
) -> Result<()> {
    if builder_ref.expected_token_count != builder.expected_token_count
        || builder_ref.token_ids_written != builder.token_ids.len()
        || builder_ref.running_commitment != token_id_builder_running_commitment(builder)
    {
        bail!(
            "raster token-id builder {} metadata mismatch",
            builder_ref.id().source_name()
        );
    }
    Ok(())
}

fn update_pieces(hasher: &mut Sha256, pieces: &[String]) {
    hasher.update((pieces.len() as u64).to_le_bytes());
    for piece in pieces {
        let bytes = piece.as_bytes();
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    }
}

fn update_token_ids(hasher: &mut Sha256, token_ids: &[u32]) {
    hasher.update((token_ids.len() as u64).to_le_bytes());
    for token_id in token_ids {
        hasher.update(token_id.to_le_bytes());
    }
}

fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    bytes
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn ensure_commitment(expected: &str, actual: &str, label: &str) -> Result<()> {
    if expected != actual {
        bail!("{label} commitment mismatch: {actual} vs {expected}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raster_authoring::auth_read;

    fn sequence_id(name: &str) -> RasterTokenizerSequenceId {
        RasterTokenizerSequenceId::new(name).expect("sequence id")
    }

    #[test]
    fn piece_sequences_read_pieces_pairs_and_materialize_in_order() {
        let mut store = AuthenticatedRasterTokenizerStore::new();
        let pieces_ref = store
            .insert_bpe_piece_sequence(
                sequence_id("pieces"),
                vec!["a".to_string(), "b".to_string(), "c".to_string()],
            )
            .expect("insert pieces");

        assert_eq!(
            auth_read(
                &store,
                RasterBpePieceRequest {
                    pieces_ref: pieces_ref.clone(),
                    piece_idx: 1,
                }
            )
            .expect("piece"),
            "b"
        );
        assert_eq!(
            auth_read(
                &store,
                RasterBpePairRequest {
                    pieces_ref: pieces_ref.clone(),
                    pair_idx: 1,
                }
            )
            .expect("pair"),
            RasterBpePair {
                left: "b".to_string(),
                right: "c".to_string(),
            }
        );
        assert_eq!(
            store.materialize_bpe_pieces(&pieces_ref).expect("pieces"),
            vec!["a", "b", "c"]
        );
    }

    #[test]
    fn one_piece_sequence_has_no_valid_adjacent_pair() {
        let mut store = AuthenticatedRasterTokenizerStore::new();
        let pieces_ref = store
            .insert_bpe_piece_sequence(sequence_id("pieces"), vec!["a".to_string()])
            .expect("insert pieces");

        assert!(auth_read(
            &store,
            RasterBpePairRequest {
                pieces_ref,
                pair_idx: 0,
            }
        )
        .is_err());
    }

    #[test]
    fn piece_reads_fail_closed_for_range_identity_and_commitment_errors() {
        let mut store = AuthenticatedRasterTokenizerStore::new();
        let pieces_ref = store
            .insert_bpe_piece_sequence(sequence_id("pieces"), vec!["a".to_string()])
            .expect("insert pieces");

        assert!(auth_read(
            &store,
            RasterBpePieceRequest {
                pieces_ref: pieces_ref.clone(),
                piece_idx: 1,
            }
        )
        .is_err());

        let missing_ref = RasterBpePieceSequenceRef::new(
            sequence_id("missing"),
            pieces_ref.piece_count(),
            pieces_ref.det_commitment().to_string(),
        )
        .expect("missing ref");
        assert!(auth_read(
            &store,
            RasterBpePieceRequest {
                pieces_ref: missing_ref,
                piece_idx: 0,
            }
        )
        .is_err());

        let wrong_commitment_ref = RasterBpePieceSequenceRef::new(
            sequence_id("pieces"),
            pieces_ref.piece_count(),
            "not-the-piece-commitment",
        )
        .expect("wrong commitment ref");
        assert!(auth_read(
            &store,
            RasterBpePieceRequest {
                pieces_ref: wrong_commitment_ref,
                piece_idx: 0,
            }
        )
        .is_err());
    }

    #[test]
    fn piece_builders_finalize_and_fail_closed_for_invalid_writes() {
        let mut store = AuthenticatedRasterTokenizerStore::new();
        let mut builder = store
            .start_bpe_piece_sequence_builder(sequence_id("pieces"), 2)
            .expect("builder");

        assert!(store
            .append_bpe_piece(&mut builder, 1, "b".to_string())
            .is_err());
        assert!(store
            .finalize_bpe_piece_sequence_builder(builder.clone())
            .is_err());
        store
            .append_bpe_piece(&mut builder, 0, "a".to_string())
            .expect("append");
        assert!(store
            .append_bpe_piece(&mut builder, 0, "duplicate".to_string())
            .is_err());
        store
            .append_bpe_piece(&mut builder, 1, "b".to_string())
            .expect("append");

        let pieces_ref = store
            .finalize_bpe_piece_sequence_builder(builder)
            .expect("finalize");
        assert_eq!(
            store.materialize_bpe_pieces(&pieces_ref).expect("pieces"),
            vec!["a", "b"]
        );
    }

    #[test]
    fn token_id_builders_finalize_and_materialize() {
        let mut store = AuthenticatedRasterTokenizerStore::new();
        let mut builder = store
            .start_token_id_sequence_builder(sequence_id("ids"), 2)
            .expect("builder");
        store
            .append_token_id(&mut builder, 0, 1)
            .expect("append token");
        store
            .append_token_id(&mut builder, 1, 3)
            .expect("append token");

        let token_ids_ref = store
            .finalize_token_id_sequence_builder(builder)
            .expect("finalize");

        assert_eq!(
            store
                .materialize_token_ids(&token_ids_ref)
                .expect("token ids"),
            vec![1, 3]
        );
    }

    #[test]
    fn serialized_refs_and_requests_do_not_embed_backing_store() {
        let mut store = AuthenticatedRasterTokenizerStore::new();
        let pieces_ref = store
            .insert_bpe_piece_sequence(sequence_id("pieces"), vec!["secret-piece".to_string()])
            .expect("insert pieces");
        let request = RasterBpePieceRequest {
            pieces_ref,
            piece_idx: 0,
        };

        let serialized = serde_json::to_string(&request).expect("serialize request");

        assert!(!serialized.contains("secret-piece"));
        assert!(!serialized.contains("piece_sequences"));
        assert!(!serialized.contains("piece_builders"));
        assert!(!serialized.contains("token_id_sequences"));
        assert!(!serialized.contains("token_id_builders"));
    }
}
