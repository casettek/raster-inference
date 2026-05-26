use std::cell::RefCell;

use anyhow::{anyhow, bail, Result};

use crate::shared::artifacts::artifact_io::{ArtifactIo, AuthRead};
use crate::shared::artifacts::external_artifacts::{
    decode_i32_vec_response, decode_postcard_response, postcard_external_source_entry,
    postcard_i32_vec_external_source_entry, postcard_request_key, register_external_source,
    CommittedExternalRequest, CommittedExternalSource, ExternalSourceEntry, ExternalSourceId,
    ExternalSourceRef,
};
use crate::shared::artifacts::raster_artifact_store::{
    RasterActivationSequenceArtifactRef, RasterArtifactId, RasterArtifactMetadata,
    RasterArtifactStoreRoots,
};
use crate::shared::model::transformer::{
    Gemma4ModelProvenance, Gemma4PleGlobalWeights, Gemma4PleMatrixSource, Gemma4TransformerModel,
};
use crate::shared::numerics::det_num::{Acc, Act, Wgt};
use crate::shared::raster_kernels::transformer::det_num_matrix_row_wgts;

#[derive(Debug, Clone)]
pub struct AuthenticatedGemmaPleSource {
    identifier: String,
    layers: Vec<GemmaPleLayerMetadata>,
    projection_norm_weights: Option<Vec<Wgt>>,
    scalars: Option<GemmaPleScalars>,
    backing: GemmaPleBacking,
    committed_source: RefCell<Option<CommittedExternalSource>>,
}

#[derive(Debug, Clone)]
enum GemmaPleBacking {
    None,
    Owned {
        token_embeddings: Vec<Vec<Vec<Act>>>,
        model_projections: Vec<Vec<Vec<Wgt>>>,
    },
    Model(Gemma4PleGlobalWeights),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaPleLayerConfig {
    pub has_ple: bool,
    pub hidden_width: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaPleMetadata {
    pub source_id: String,
    pub has_ple_global: bool,
    pub layer_count: usize,
    pub token_embedding_layer_count: usize,
    pub model_projection_layer_count: usize,
    pub projection_norm_width: Option<usize>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterPrefillPleInputRefs {
    artifact_store_roots: RasterArtifactStoreRoots,
    source_id: String,
    layer_count: usize,
    token_count: usize,
    per_layer_inputs: Vec<Option<RasterActivationSequenceArtifactRef>>,
}

const PREFILL_PLE_INPUT_MANIFEST_KIND: &str = "prefill_ple_input_manifest";
const PREFILL_PLE_INPUT_MANIFEST_DOMAIN: &str = "raster-prefill-ple-input-manifest-v1";
const PREFILL_PLE_INPUT_MANIFEST_SOURCE_NAME: &str = "prefill.prepare_aux.ple_input_manifest";

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterPrefillPleInputManifest {
    source_id: String,
    layer_count: usize,
    token_count: usize,
    per_layer_input_roots: Vec<Option<String>>,
}

impl RasterPrefillPleInputManifest {
    pub fn new(
        source_id: impl Into<String>,
        layer_count: usize,
        token_count: usize,
        per_layer_input_roots: Vec<Option<String>>,
    ) -> Result<Self> {
        let source_id = validate_identifier(source_id.into())?;
        if layer_count == 0 {
            bail!("raster PLE input manifest requires at least one layer");
        }
        if token_count == 0 {
            bail!("raster PLE input manifest requires at least one token");
        }
        if per_layer_input_roots.len() != layer_count {
            bail!(
                "raster PLE input manifest received {} layers, expected {layer_count}",
                per_layer_input_roots.len()
            );
        }
        Ok(Self {
            source_id,
            layer_count,
            token_count,
            per_layer_input_roots,
        })
    }

    pub fn source_id(&self) -> &str {
        &self.source_id
    }

    pub fn layer_count(&self) -> usize {
        self.layer_count
    }

    pub fn token_count(&self) -> usize {
        self.token_count
    }

    pub fn per_layer_input_roots(&self) -> &[Option<String>] {
        &self.per_layer_input_roots
    }

    pub fn into_prefill_ple_input_refs(
        self,
        artifact_store_roots: RasterArtifactStoreRoots,
    ) -> Result<RasterPrefillPleInputRefs> {
        let per_layer_inputs = self
            .per_layer_input_roots
            .into_iter()
            .map(|root| {
                root.map(|root| {
                    ensure_artifact_root_present(&artifact_store_roots, &root)?;
                    RasterActivationSequenceArtifactRef::new(ArtifactIo::artifact_ref_for_root_any(
                        &root,
                    )?)
                })
                .transpose()
            })
            .collect::<Result<Vec<_>>>()?;

        RasterPrefillPleInputRefs::new_with_roots(
            artifact_store_roots,
            self.source_id,
            self.layer_count,
            self.token_count,
            per_layer_inputs,
        )
    }
}

pub fn store_prefill_ple_input_manifest_with_roots(
    roots: &RasterArtifactStoreRoots,
    source_id: impl Into<String>,
    layer_count: usize,
    token_count: usize,
    per_layer_inputs: &[Option<RasterActivationSequenceArtifactRef>],
) -> Result<(RasterArtifactStoreRoots, String)> {
    let manifest = RasterPrefillPleInputManifest::new(
        source_id,
        layer_count,
        token_count,
        per_layer_inputs
            .iter()
            .map(|input| input.as_ref().map(|input| input.root().to_string()))
            .collect(),
    )?;
    let payload = postcard::to_allocvec(&manifest)?;
    let (roots, artifact_ref) = ArtifactIo::insert_artifact_with_roots(
        roots,
        RasterArtifactId::new(PREFILL_PLE_INPUT_MANIFEST_SOURCE_NAME)?,
        RasterArtifactMetadata::new(
            PREFILL_PLE_INPUT_MANIFEST_KIND,
            PREFILL_PLE_INPUT_MANIFEST_DOMAIN,
            1,
            Vec::new(),
        )?,
        vec![payload],
    )?;
    Ok((roots, artifact_ref.root().to_string()))
}

pub fn read_prefill_ple_input_manifest_from_roots(
    roots: &RasterArtifactStoreRoots,
    manifest_root: &str,
) -> Result<RasterPrefillPleInputManifest> {
    ensure_artifact_root_present(roots, manifest_root)?;
    let artifact_ref = ArtifactIo::artifact_ref_for_root_any(manifest_root)?;
    if artifact_ref.metadata().kind() != PREFILL_PLE_INPUT_MANIFEST_KIND {
        bail!(
            "raster PLE input manifest kind mismatch: {}",
            artifact_ref.metadata().kind()
        );
    }
    if artifact_ref.metadata().domain() != PREFILL_PLE_INPUT_MANIFEST_DOMAIN {
        bail!(
            "raster PLE input manifest domain mismatch: {}",
            artifact_ref.metadata().domain()
        );
    }
    if artifact_ref.metadata().leaf_count() != 1 {
        bail!("raster PLE input manifest must have exactly one leaf");
    }
    ArtifactIo::read_authenticated_leaf_from_roots(roots, &artifact_ref, 0)?.deserialize()
}

fn ensure_artifact_root_present(roots: &RasterArtifactStoreRoots, root: &str) -> Result<()> {
    if roots.artifacts.iter().any(|entry| entry.root() == root) {
        return Ok(());
    }
    bail!("raster artifact root {root} is not present in the store roots snapshot")
}

impl RasterPrefillPleInputRefs {
    pub fn new(
        source_id: impl Into<String>,
        layer_count: usize,
        token_count: usize,
        per_layer_inputs: Vec<Option<RasterActivationSequenceArtifactRef>>,
    ) -> Result<Self> {
        Self::new_with_roots(
            RasterArtifactStoreRoots::default(),
            source_id,
            layer_count,
            token_count,
            per_layer_inputs,
        )
    }

    pub fn new_with_roots(
        artifact_store_roots: RasterArtifactStoreRoots,
        source_id: impl Into<String>,
        layer_count: usize,
        token_count: usize,
        per_layer_inputs: Vec<Option<RasterActivationSequenceArtifactRef>>,
    ) -> Result<Self> {
        let source_id = validate_identifier(source_id.into())?;
        if layer_count == 0 {
            bail!("raster PLE input refs require at least one layer");
        }
        if token_count == 0 {
            bail!("raster PLE input refs require at least one token");
        }
        if per_layer_inputs.len() != layer_count {
            bail!(
                "raster PLE input refs received {} layers, expected {layer_count}",
                per_layer_inputs.len()
            );
        }
        Ok(Self {
            artifact_store_roots,
            source_id,
            layer_count,
            token_count,
            per_layer_inputs,
        })
    }

    pub fn source_id(&self) -> &str {
        &self.source_id
    }

    pub fn artifact_store_roots(&self) -> &RasterArtifactStoreRoots {
        &self.artifact_store_roots
    }

    pub fn layer_count(&self) -> usize {
        self.layer_count
    }

    pub fn token_count(&self) -> usize {
        self.token_count
    }

    pub fn per_layer_inputs(&self) -> &[Option<RasterActivationSequenceArtifactRef>] {
        &self.per_layer_inputs
    }

    pub fn clone_layer_ref(
        &self,
        layer_idx: usize,
    ) -> Result<Option<RasterActivationSequenceArtifactRef>> {
        self.per_layer_inputs
            .get(layer_idx)
            .cloned()
            .ok_or_else(|| {
                anyhow!(
                    "raster PLE input ref layer {layer_idx} is out of range for {} layers",
                    self.layer_count
                )
            })
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GemmaPleLayerMetadata {
    pub layer_idx: usize,
    pub has_ple: bool,
    pub hidden_width: usize,
    pub token_embedding_vocab_size: Option<usize>,
    pub ple_width: Option<usize>,
    pub model_projection_rows: Option<usize>,
    pub model_projection_cols: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaPleScalars {
    pub embedding_scale: Act,
    pub projection_scalar: Act,
    pub input_scale: Act,
    pub rms_norm_eps: Acc,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
struct GemmaPleScalarsPayload {
    embedding_scale_bits: i32,
    projection_scalar_bits: i32,
    input_scale_bits: i32,
    rms_norm_eps_bits: i64,
}

const GEMMA_PLE_SOURCE_KIND: &str = "gemma_ple";
const GEMMA_PLE_SOURCE_DOMAIN: &str = "raster-external-source-gemma-ple-merkle-v1";
const PLE_METADATA_REQUEST: &str = "gemma_ple.metadata";
const PLE_LAYER_METADATA_REQUEST: &str = "gemma_ple.layer_metadata";
const PLE_TOKEN_EMBEDDING_ROW_REQUEST: &str = "gemma_ple.token_embedding_row";
const PLE_MODEL_PROJECTION_ROW_REQUEST: &str = "gemma_ple.model_projection_row";
const PLE_PROJECTION_NORM_WEIGHTS_REQUEST: &str = "gemma_ple.projection_norm_weights";
const PLE_SCALARS_REQUEST: &str = "gemma_ple.scalars";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaPleMetadataRequest;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaPleLayerMetadataRequest {
    pub layer_idx: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaPleTokenEmbeddingRowRequest {
    pub layer_idx: usize,
    pub token_id: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaPleModelProjectionRowRequest {
    pub layer_idx: usize,
    pub row_idx: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaPleProjectionNormWeightsRequest;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmaPleScalarsRequest;

impl AuthenticatedGemmaPleSource {
    pub fn from_model(
        identifier: impl Into<String>,
        model: &Gemma4TransformerModel,
    ) -> Result<Self> {
        let layers = model
            .layers
            .iter()
            .map(|layer| GemmaPleLayerConfig {
                has_ple: layer.ple.is_some(),
                hidden_width: layer.hidden_size,
            })
            .collect();
        Self::from_ple_global(
            identifier,
            model.provenance,
            layers,
            model.ple_global.clone(),
            model.rms_norm_eps_det,
        )
    }

    pub fn from_ple_global(
        identifier: impl Into<String>,
        provenance: Gemma4ModelProvenance,
        layer_configs: Vec<GemmaPleLayerConfig>,
        ple_global: Option<Gemma4PleGlobalWeights>,
        rms_norm_eps_det: Option<Acc>,
    ) -> Result<Self> {
        let identifier = validate_identifier(identifier.into())?;
        let has_ple_layer = layer_configs.iter().any(|layer| layer.has_ple);
        let Some(ple_global) = ple_global else {
            if has_ple_layer {
                bail!("Gemma PLE source has PLE layers but no global PLE weights");
            }
            let layers = layer_configs
                .into_iter()
                .enumerate()
                .map(|(layer_idx, config)| GemmaPleLayerMetadata {
                    layer_idx,
                    has_ple: config.has_ple,
                    hidden_width: config.hidden_width,
                    token_embedding_vocab_size: None,
                    ple_width: None,
                    model_projection_rows: None,
                    model_projection_cols: None,
                })
                .collect();
            return Ok(Self {
                identifier,
                layers,
                projection_norm_weights: None,
                scalars: None,
                backing: GemmaPleBacking::None,
                committed_source: RefCell::new(None),
            });
        };

        if provenance != Gemma4ModelProvenance::DetNumWgt {
            bail!(
                "deterministic raster PLE source requires a model loaded from a .detwgt artifact"
            );
        }

        ensure_model_backing_is_canonical(&ple_global)?;
        let projection_norm_weights = canonical_projection_norm_weights(&ple_global)?;
        let scalars = canonical_scalars(&ple_global, rms_norm_eps_det)?;
        let layers = build_layer_metadata(
            layer_configs,
            &ple_global.token_embeddings,
            &ple_global.model_projections,
            projection_norm_weights.len(),
        )?;

        Ok(Self {
            identifier,
            layers,
            projection_norm_weights: Some(projection_norm_weights),
            scalars: Some(scalars),
            backing: GemmaPleBacking::Model(ple_global),
            committed_source: RefCell::new(None),
        })
    }

    pub fn from_canonical_parts(
        identifier: impl Into<String>,
        layer_configs: Vec<GemmaPleLayerConfig>,
        token_embeddings: Vec<Vec<Vec<Act>>>,
        model_projections: Vec<Vec<Vec<Wgt>>>,
        projection_norm_weights: Vec<Wgt>,
        scalars: GemmaPleScalars,
    ) -> Result<Self> {
        let identifier = validate_identifier(identifier.into())?;
        let layers = build_owned_layer_metadata(
            layer_configs,
            &token_embeddings,
            &model_projections,
            projection_norm_weights.len(),
        )?;

        Ok(Self {
            identifier,
            layers,
            projection_norm_weights: Some(projection_norm_weights),
            scalars: Some(scalars),
            backing: GemmaPleBacking::Owned {
                token_embeddings,
                model_projections,
            },
            committed_source: RefCell::new(None),
        })
    }

    pub fn no_ple(
        identifier: impl Into<String>,
        layer_configs: Vec<GemmaPleLayerConfig>,
    ) -> Result<Self> {
        Self::from_ple_global(
            identifier,
            Gemma4ModelProvenance::DetNumWgt,
            layer_configs,
            None,
            None,
        )
    }

    pub fn identifier(&self) -> &str {
        &self.identifier
    }

    pub fn committed_source_ref(&self) -> Result<ExternalSourceRef> {
        Ok(self.committed_source()?.source_ref().clone())
    }

    pub fn committed_source(&self) -> Result<CommittedExternalSource> {
        if let Some(source) = self.committed_source.borrow().clone() {
            return Ok(source);
        }

        let source_ref = register_external_source(
            ExternalSourceId::new(format!("ple:{}", self.identifier))?,
            GEMMA_PLE_SOURCE_KIND,
            GEMMA_PLE_SOURCE_DOMAIN,
            self.committed_source_entries()?,
        )?;
        let source = CommittedExternalSource::new(source_ref);
        *self.committed_source.borrow_mut() = Some(source.clone());
        Ok(source)
    }

    fn metadata(&self) -> GemmaPleMetadata {
        let has_ple_global = !matches!(self.backing, GemmaPleBacking::None);
        GemmaPleMetadata {
            source_id: self.identifier.clone(),
            has_ple_global,
            layer_count: self.layers.len(),
            token_embedding_layer_count: if has_ple_global { self.layers.len() } else { 0 },
            model_projection_layer_count: if has_ple_global { self.layers.len() } else { 0 },
            projection_norm_width: self.projection_norm_weights.as_ref().map(Vec::len),
        }
    }

    fn require_ple_global(&self) -> Result<()> {
        if matches!(self.backing, GemmaPleBacking::None) {
            bail!("Gemma PLE source has no global PLE weights");
        }
        Ok(())
    }

    fn committed_source_entries(&self) -> Result<Vec<ExternalSourceEntry>> {
        let mut entries = Vec::new();
        entries.push(postcard_external_source_entry(
            GemmaPleMetadataRequest.request_key()?,
            &self.metadata(),
        )?);
        for layer in &self.layers {
            entries.push(postcard_external_source_entry(
                GemmaPleLayerMetadataRequest {
                    layer_idx: layer.layer_idx,
                }
                .request_key()?,
                layer,
            )?);
        }

        if !matches!(self.backing, GemmaPleBacking::None) {
            let norm_bits = self
                .native_projection_norm_weights()?
                .into_iter()
                .map(|value| value.to_bits());
            entries.push(postcard_i32_vec_external_source_entry(
                GemmaPleProjectionNormWeightsRequest.request_key()?,
                norm_bits,
            )?);
            entries.push(postcard_external_source_entry(
                GemmaPleScalarsRequest.request_key()?,
                &self.native_scalars()?.payload(),
            )?);
            for layer in &self.layers {
                let Some(token_count) = layer.token_embedding_vocab_size else {
                    continue;
                };
                for token_id in 0..token_count {
                    let request = GemmaPleTokenEmbeddingRowRequest {
                        layer_idx: layer.layer_idx,
                        token_id: u32::try_from(token_id).map_err(|_| {
                            anyhow!(
                                "PLE token id {token_id} exceeds u32 at layer {}",
                                layer.layer_idx
                            )
                        })?,
                    };
                    let row_bits = self
                        .native_token_embedding_row(request)?
                        .into_iter()
                        .map(|value| value.to_bits());
                    entries.push(postcard_i32_vec_external_source_entry(
                        request.request_key()?,
                        row_bits,
                    )?);
                }
                if let Some(row_count) = layer.model_projection_rows {
                    for row_idx in 0..row_count {
                        let request = GemmaPleModelProjectionRowRequest {
                            layer_idx: layer.layer_idx,
                            row_idx,
                        };
                        let row_bits = self
                            .native_model_projection_row(request)?
                            .into_iter()
                            .map(|value| value.to_bits());
                        entries.push(postcard_i32_vec_external_source_entry(
                            request.request_key()?,
                            row_bits,
                        )?);
                    }
                }
            }
        }

        Ok(entries)
    }

    fn native_token_embedding_row(
        &self,
        request: GemmaPleTokenEmbeddingRowRequest,
    ) -> Result<Vec<Act>> {
        self.require_ple_global()?;
        let row_idx = usize::try_from(request.token_id).expect("u32 should fit into usize");
        match &self.backing {
            GemmaPleBacking::None => unreachable!("require_ple_global checked this case"),
            GemmaPleBacking::Owned {
                token_embeddings, ..
            } => token_embeddings
                .get(request.layer_idx)
                .ok_or_else(|| {
                    anyhow!(
                        "transformer PLE token embedding slice count mismatch at layer {}",
                        request.layer_idx
                    )
                })?
                .get(row_idx)
                .cloned()
                .ok_or_else(|| {
                    anyhow!(
                        "Gemma PLE token id {} is out of range for layer {}",
                        request.token_id,
                        request.layer_idx
                    )
                }),
            GemmaPleBacking::Model(ple_global) => {
                let row = crate::io::load_ple_token_embedding_row_internal(
                    ple_global,
                    request.layer_idx,
                    request.token_id,
                )?;
                row.det_values().map(<[Act]>::to_vec).ok_or_else(|| {
                    anyhow!("deterministic raster PLE token row requires canonical Act values")
                })
            }
        }
    }

    fn native_model_projection_row(
        &self,
        request: GemmaPleModelProjectionRowRequest,
    ) -> Result<Vec<Wgt>> {
        self.require_ple_global()?;
        match &self.backing {
            GemmaPleBacking::None => unreachable!("require_ple_global checked this case"),
            GemmaPleBacking::Owned {
                model_projections, ..
            } => model_projections
                .get(request.layer_idx)
                .ok_or_else(|| {
                    anyhow!(
                        "transformer PLE model projection slice count mismatch at layer {}",
                        request.layer_idx
                    )
                })?
                .get(request.row_idx)
                .cloned()
                .ok_or_else(|| {
                    anyhow!(
                        "Gemma PLE model projection row {} is out of range for layer {}",
                        request.row_idx,
                        request.layer_idx
                    )
                }),
            GemmaPleBacking::Model(ple_global) => {
                let matrix = crate::io::materialize_det_num_ple_model_projection(
                    ple_global,
                    request.layer_idx,
                )?
                .ok_or_else(|| {
                    anyhow!(
                        "deterministic raster PLE model projection layer {} requires canonical matrix",
                        request.layer_idx
                    )
                })?;
                det_num_matrix_row_wgts(&matrix, request.row_idx, "PLE model projection")
            }
        }
    }

    fn native_projection_norm_weights(&self) -> Result<Vec<Wgt>> {
        self.require_ple_global()?;
        self.projection_norm_weights.clone().ok_or_else(|| {
            anyhow!("deterministic raster PLE source requires canonical norm weights")
        })
    }

    fn native_scalars(&self) -> Result<GemmaPleScalars> {
        self.require_ple_global()?;
        self.scalars
            .ok_or_else(|| anyhow!("deterministic raster PLE source requires canonical scalars"))
    }
}

impl AuthRead<GemmaPleMetadataRequest> for AuthenticatedGemmaPleSource {
    type Output = GemmaPleMetadata;

    fn auth_read(&self, request: GemmaPleMetadataRequest) -> Result<Self::Output> {
        self.committed_source()?.auth_read(request)
    }
}

impl AuthRead<GemmaPleLayerMetadataRequest> for AuthenticatedGemmaPleSource {
    type Output = GemmaPleLayerMetadata;

    fn auth_read(&self, request: GemmaPleLayerMetadataRequest) -> Result<Self::Output> {
        if request.layer_idx >= self.layers.len() {
            bail!(
                "Gemma PLE layer index {} is out of range for {} layers",
                request.layer_idx,
                self.layers.len()
            );
        }
        self.committed_source()?.auth_read(request)
    }
}

impl AuthRead<GemmaPleTokenEmbeddingRowRequest> for AuthenticatedGemmaPleSource {
    type Output = Vec<Act>;

    fn auth_read(&self, request: GemmaPleTokenEmbeddingRowRequest) -> Result<Self::Output> {
        self.require_ple_global()?;
        self.committed_source()?.auth_read(request)
    }
}

impl AuthRead<GemmaPleModelProjectionRowRequest> for AuthenticatedGemmaPleSource {
    type Output = Vec<Wgt>;

    fn auth_read(&self, request: GemmaPleModelProjectionRowRequest) -> Result<Self::Output> {
        self.require_ple_global()?;
        self.committed_source()?.auth_read(request)
    }
}

impl AuthRead<GemmaPleProjectionNormWeightsRequest> for AuthenticatedGemmaPleSource {
    type Output = Vec<Wgt>;

    fn auth_read(&self, request: GemmaPleProjectionNormWeightsRequest) -> Result<Self::Output> {
        self.require_ple_global()?;
        self.committed_source()?.auth_read(request)
    }
}

impl AuthRead<GemmaPleScalarsRequest> for AuthenticatedGemmaPleSource {
    type Output = GemmaPleScalars;

    fn auth_read(&self, request: GemmaPleScalarsRequest) -> Result<Self::Output> {
        self.require_ple_global()?;
        self.committed_source()?.auth_read(request)
    }
}

impl CommittedExternalRequest for GemmaPleMetadataRequest {
    type Output = GemmaPleMetadata;

    fn request_key(&self) -> Result<Vec<u8>> {
        postcard_request_key(PLE_METADATA_REQUEST, &())
    }

    fn decode_response(&self, response_payload: &[u8]) -> Result<Self::Output> {
        decode_postcard_response(response_payload)
    }
}

impl CommittedExternalRequest for GemmaPleLayerMetadataRequest {
    type Output = GemmaPleLayerMetadata;

    fn request_key(&self) -> Result<Vec<u8>> {
        postcard_request_key(PLE_LAYER_METADATA_REQUEST, &self.layer_idx)
    }

    fn decode_response(&self, response_payload: &[u8]) -> Result<Self::Output> {
        decode_postcard_response(response_payload)
    }
}

impl CommittedExternalRequest for GemmaPleTokenEmbeddingRowRequest {
    type Output = Vec<Act>;

    fn request_key(&self) -> Result<Vec<u8>> {
        postcard_request_key(
            PLE_TOKEN_EMBEDDING_ROW_REQUEST,
            &(self.layer_idx, self.token_id),
        )
    }

    fn decode_response(&self, response_payload: &[u8]) -> Result<Self::Output> {
        Ok(decode_i32_vec_response(response_payload)?
            .into_iter()
            .map(Act::from_bits)
            .collect())
    }
}

impl CommittedExternalRequest for GemmaPleModelProjectionRowRequest {
    type Output = Vec<Wgt>;

    fn request_key(&self) -> Result<Vec<u8>> {
        postcard_request_key(
            PLE_MODEL_PROJECTION_ROW_REQUEST,
            &(self.layer_idx, self.row_idx),
        )
    }

    fn decode_response(&self, response_payload: &[u8]) -> Result<Self::Output> {
        Ok(decode_i32_vec_response(response_payload)?
            .into_iter()
            .map(Wgt::from_bits)
            .collect())
    }
}

impl CommittedExternalRequest for GemmaPleProjectionNormWeightsRequest {
    type Output = Vec<Wgt>;

    fn request_key(&self) -> Result<Vec<u8>> {
        postcard_request_key(PLE_PROJECTION_NORM_WEIGHTS_REQUEST, &())
    }

    fn decode_response(&self, response_payload: &[u8]) -> Result<Self::Output> {
        Ok(decode_i32_vec_response(response_payload)?
            .into_iter()
            .map(Wgt::from_bits)
            .collect())
    }
}

impl CommittedExternalRequest for GemmaPleScalarsRequest {
    type Output = GemmaPleScalars;

    fn request_key(&self) -> Result<Vec<u8>> {
        postcard_request_key(PLE_SCALARS_REQUEST, &())
    }

    fn decode_response(&self, response_payload: &[u8]) -> Result<Self::Output> {
        Ok(GemmaPleScalars::from_payload(decode_postcard_response(
            response_payload,
        )?))
    }
}

impl GemmaPleScalars {
    fn payload(self) -> GemmaPleScalarsPayload {
        GemmaPleScalarsPayload {
            embedding_scale_bits: self.embedding_scale.to_bits(),
            projection_scalar_bits: self.projection_scalar.to_bits(),
            input_scale_bits: self.input_scale.to_bits(),
            rms_norm_eps_bits: self.rms_norm_eps.to_bits(),
        }
    }

    fn from_payload(payload: GemmaPleScalarsPayload) -> Self {
        Self {
            embedding_scale: Act::from_bits(payload.embedding_scale_bits),
            projection_scalar: Act::from_bits(payload.projection_scalar_bits),
            input_scale: Act::from_bits(payload.input_scale_bits),
            rms_norm_eps: Acc::from_bits(payload.rms_norm_eps_bits),
        }
    }
}

fn validate_identifier(identifier: String) -> Result<String> {
    if identifier.is_empty() {
        bail!("Gemma PLE source identifier must not be empty");
    }
    Ok(identifier)
}

fn ensure_model_backing_is_canonical(ple_global: &Gemma4PleGlobalWeights) -> Result<()> {
    for (layer_idx, source) in ple_global.token_embeddings.iter().enumerate() {
        if !matches!(source, Gemma4PleMatrixSource::DetNumLazy(_)) {
            bail!(
                "deterministic raster PLE source requires .detwgt token embedding source at layer {layer_idx}"
            );
        }
    }
    for (layer_idx, source) in ple_global.model_projections.iter().enumerate() {
        if !matches!(source, Gemma4PleMatrixSource::DetNumLazy(_)) {
            bail!(
                "deterministic raster PLE source requires .detwgt model projection source at layer {layer_idx}"
            );
        }
    }
    Ok(())
}

fn canonical_projection_norm_weights(ple_global: &Gemma4PleGlobalWeights) -> Result<Vec<Wgt>> {
    ple_global
        .projection_norm_weight_det
        .clone()
        .ok_or_else(|| anyhow!("deterministic raster PLE source requires canonical norm weights"))
}

fn canonical_scalars(
    ple_global: &Gemma4PleGlobalWeights,
    rms_norm_eps_det: Option<Acc>,
) -> Result<GemmaPleScalars> {
    Ok(GemmaPleScalars {
        embedding_scale: ple_global.embedding_scale_det.ok_or_else(|| {
            anyhow!("deterministic raster PLE source requires canonical embedding scale")
        })?,
        projection_scalar: ple_global.projection_scalar_det.ok_or_else(|| {
            anyhow!("deterministic raster PLE source requires canonical projection scalar")
        })?,
        input_scale: ple_global.input_scale_det.ok_or_else(|| {
            anyhow!("deterministic raster PLE source requires canonical input scale")
        })?,
        rms_norm_eps: rms_norm_eps_det.ok_or_else(|| {
            anyhow!("deterministic raster PLE source requires canonical RMSNorm epsilon")
        })?,
    })
}

fn build_layer_metadata(
    layer_configs: Vec<GemmaPleLayerConfig>,
    token_embeddings: &[Gemma4PleMatrixSource],
    model_projections: &[Gemma4PleMatrixSource],
    projection_norm_width: usize,
) -> Result<Vec<GemmaPleLayerMetadata>> {
    if token_embeddings.len() != layer_configs.len() {
        bail!(
            "transformer PLE token embedding slice count mismatch: {} vs {}",
            token_embeddings.len(),
            layer_configs.len()
        );
    }
    if model_projections.len() != layer_configs.len() {
        bail!(
            "transformer PLE model projection slice count mismatch: {} vs {}",
            model_projections.len(),
            layer_configs.len()
        );
    }

    layer_configs
        .into_iter()
        .enumerate()
        .map(|(layer_idx, config)| {
            let token_shape = source_shape(&token_embeddings[layer_idx]);
            let projection_shape = source_shape(&model_projections[layer_idx]);
            validate_layer_shape(
                layer_idx,
                config,
                token_shape,
                projection_shape,
                projection_norm_width,
            )
        })
        .collect()
}

fn build_owned_layer_metadata(
    layer_configs: Vec<GemmaPleLayerConfig>,
    token_embeddings: &[Vec<Vec<Act>>],
    model_projections: &[Vec<Vec<Wgt>>],
    projection_norm_width: usize,
) -> Result<Vec<GemmaPleLayerMetadata>> {
    if token_embeddings.len() != layer_configs.len() {
        bail!(
            "transformer PLE token embedding slice count mismatch: {} vs {}",
            token_embeddings.len(),
            layer_configs.len()
        );
    }
    if model_projections.len() != layer_configs.len() {
        bail!(
            "transformer PLE model projection slice count mismatch: {} vs {}",
            model_projections.len(),
            layer_configs.len()
        );
    }

    layer_configs
        .into_iter()
        .enumerate()
        .map(|(layer_idx, config)| {
            let token_shape = owned_act_matrix_shape(
                token_embeddings
                    .get(layer_idx)
                    .expect("layer count was validated"),
                "PLE token embedding",
                layer_idx,
            )?;
            let projection_shape = owned_wgt_matrix_shape(
                model_projections
                    .get(layer_idx)
                    .expect("layer count was validated"),
                "PLE model projection",
                layer_idx,
            )?;
            validate_layer_shape(
                layer_idx,
                config,
                token_shape,
                projection_shape,
                projection_norm_width,
            )
        })
        .collect()
}

fn validate_layer_shape(
    layer_idx: usize,
    config: GemmaPleLayerConfig,
    token_shape: (usize, usize),
    projection_shape: (usize, usize),
    projection_norm_width: usize,
) -> Result<GemmaPleLayerMetadata> {
    let (token_embedding_vocab_size, ple_width) = token_shape;
    let (model_projection_rows, model_projection_cols) = projection_shape;

    if config.has_ple {
        if ple_width != model_projection_rows {
            bail!(
                "Gemma PLE width mismatch at layer {layer_idx}: token embedding width {ple_width} vs projection rows {model_projection_rows}"
            );
        }
        if model_projection_cols != config.hidden_width {
            bail!(
                "Gemma PLE projection hidden width mismatch at layer {layer_idx}: projection cols {model_projection_cols} vs hidden width {}",
                config.hidden_width
            );
        }
        if projection_norm_width != ple_width {
            bail!(
                "Gemma PLE projection norm width mismatch at layer {layer_idx}: norm width {projection_norm_width} vs PLE width {ple_width}"
            );
        }
    }

    Ok(GemmaPleLayerMetadata {
        layer_idx,
        has_ple: config.has_ple,
        hidden_width: config.hidden_width,
        token_embedding_vocab_size: Some(token_embedding_vocab_size),
        ple_width: Some(ple_width),
        model_projection_rows: Some(model_projection_rows),
        model_projection_cols: Some(model_projection_cols),
    })
}

fn source_shape(source: &Gemma4PleMatrixSource) -> (usize, usize) {
    match source {
        Gemma4PleMatrixSource::Materialized(matrix) => (matrix.rows, matrix.cols),
        Gemma4PleMatrixSource::Lazy(source) => (source.row_count, source.col_count),
        Gemma4PleMatrixSource::DetNumLazy(source) => (source.row_count, source.col_count),
    }
}

fn owned_act_matrix_shape(
    matrix: &[Vec<Act>],
    label: &str,
    layer_idx: usize,
) -> Result<(usize, usize)> {
    let width = rectangular_width(matrix, label, layer_idx)?;
    Ok((matrix.len(), width))
}

fn owned_wgt_matrix_shape(
    matrix: &[Vec<Wgt>],
    label: &str,
    layer_idx: usize,
) -> Result<(usize, usize)> {
    let width = rectangular_width(matrix, label, layer_idx)?;
    Ok((matrix.len(), width))
}

fn rectangular_width<T>(matrix: &[Vec<T>], label: &str, layer_idx: usize) -> Result<usize> {
    let Some(first_row) = matrix.first() else {
        bail!("Gemma {label} matrix at layer {layer_idx} must have at least one row");
    };
    let width = first_row.len();
    if width == 0 {
        bail!("Gemma {label} matrix at layer {layer_idx} must have at least one column");
    }
    if matrix.iter().any(|row| row.len() != width) {
        bail!("Gemma {label} matrix at layer {layer_idx} is not rectangular");
    }
    Ok(width)
}

#[cfg(test)]
mod tests {
    use super::{
        AuthenticatedGemmaPleSource, GemmaPleLayerConfig, GemmaPleLayerMetadataRequest,
        GemmaPleMetadataRequest, GemmaPleModelProjectionRowRequest,
        GemmaPleProjectionNormWeightsRequest, GemmaPleScalars, GemmaPleScalarsRequest,
        GemmaPleTokenEmbeddingRowRequest,
    };
    use crate::shared::model::transformer::{
        DetNumTensorSliceSource, Gemma4ModelProvenance, Gemma4PleGlobalWeights, MatrixF32,
    };
    use crate::shared::numerics::det_num::{Acc, Act, Wgt};
    use std::path::{Path, PathBuf};

    #[test]
    fn canonical_source_reads_metadata_and_layer_shape() {
        let source = canonical_source();

        let metadata =
            crate::auth_read!(&source, GemmaPleMetadataRequest).expect("metadata should read");
        assert_eq!(source.identifier(), "ple-fixture");
        assert_eq!(metadata.source_id, "ple-fixture");
        assert!(metadata.has_ple_global);
        assert_eq!(metadata.layer_count, 1);
        assert_eq!(metadata.token_embedding_layer_count, 1);
        assert_eq!(metadata.model_projection_layer_count, 1);
        assert_eq!(metadata.projection_norm_width, Some(2));

        let layer = crate::auth_read!(&source, GemmaPleLayerMetadataRequest { layer_idx: 0 })
            .expect("layer metadata should read");
        assert!(layer.has_ple);
        assert_eq!(layer.hidden_width, 3);
        assert_eq!(layer.token_embedding_vocab_size, Some(2));
        assert_eq!(layer.ple_width, Some(2));
        assert_eq!(layer.model_projection_rows, Some(2));
        assert_eq!(layer.model_projection_cols, Some(3));
    }

    #[test]
    fn canonical_source_reads_token_embedding_rows() {
        let source = canonical_source();

        let row = crate::auth_read!(
            &source,
            GemmaPleTokenEmbeddingRowRequest {
                layer_idx: 0,
                token_id: 1,
            },
        )
        .expect("token embedding row should read");

        assert_eq!(row, vec![Act::from_num(0.25), Act::from_num(-0.5)]);
    }

    #[test]
    fn canonical_source_reads_model_projection_rows() {
        let source = canonical_source();

        let row = crate::auth_read!(
            &source,
            GemmaPleModelProjectionRowRequest {
                layer_idx: 0,
                row_idx: 1,
            },
        )
        .expect("model projection row should read");

        assert_eq!(
            row,
            vec![
                Wgt::from_num(0.5),
                Wgt::from_num(0.25),
                Wgt::from_num(-0.25)
            ]
        );
    }

    #[test]
    fn model_backing_reads_projection_row_from_det_source() {
        let (path, token_source, projection_source) = write_det_sources(
            vec![
                vec![Act::from_num(1.0), Act::from_num(0.0)],
                vec![Act::from_num(0.0), Act::from_num(1.0)],
            ],
            vec![
                vec![Wgt::from_num(1.0), Wgt::from_num(0.0), Wgt::from_num(-1.0)],
                vec![
                    Wgt::from_num(0.5),
                    Wgt::from_num(0.25),
                    Wgt::from_num(-0.25),
                ],
            ],
        );
        let ple_global = Gemma4PleGlobalWeights::from_det_num_sources_with_canonical(
            vec![token_source],
            vec![projection_source],
            vec![1.0, 1.0],
            vec![Wgt::from_num(1.0), Wgt::from_num(1.0)],
            1.0,
            Act::from_num(1.0),
            1.0,
            Act::from_num(1.0),
            1.0,
            Act::from_num(1.0),
        );
        let source = AuthenticatedGemmaPleSource::from_ple_global(
            "det-source-ple",
            Gemma4ModelProvenance::DetNumWgt,
            vec![GemmaPleLayerConfig {
                has_ple: true,
                hidden_width: 3,
            }],
            Some(ple_global),
            Some(Acc::from_num(0.001)),
        )
        .expect("det source should build");

        let row = crate::auth_read!(
            &source,
            GemmaPleModelProjectionRowRequest {
                layer_idx: 0,
                row_idx: 1,
            }
        )
        .expect("projection row should read");

        assert_eq!(
            row,
            vec![
                Wgt::from_num(0.5),
                Wgt::from_num(0.25),
                Wgt::from_num(-0.25)
            ]
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn canonical_source_reads_norm_weights_and_scalars() {
        let source = canonical_source();

        let norm = crate::auth_read!(&source, GemmaPleProjectionNormWeightsRequest)
            .expect("norm weights should read");
        assert_eq!(norm, vec![Wgt::from_num(1.0), Wgt::from_num(0.5)]);

        let scalars =
            crate::auth_read!(&source, GemmaPleScalarsRequest).expect("scalars should read");
        assert_eq!(scalars.embedding_scale, Act::from_num(2.0));
        assert_eq!(scalars.projection_scalar, Act::from_num(0.5));
        assert_eq!(scalars.input_scale, Act::from_num(0.25));
        assert_eq!(scalars.rms_norm_eps, Acc::from_num(0.001));
    }

    #[test]
    fn source_with_no_ple_reports_no_ple_state() {
        let source = AuthenticatedGemmaPleSource::no_ple(
            "no-ple",
            vec![GemmaPleLayerConfig {
                has_ple: false,
                hidden_width: 3,
            }],
        )
        .expect("no-PLE source should build");

        let metadata =
            crate::auth_read!(&source, GemmaPleMetadataRequest).expect("metadata should read");
        assert!(!metadata.has_ple_global);
        assert_eq!(metadata.token_embedding_layer_count, 0);
        assert_eq!(metadata.model_projection_layer_count, 0);
        assert_eq!(metadata.projection_norm_width, None);

        let layer = crate::auth_read!(&source, GemmaPleLayerMetadataRequest { layer_idx: 0 })
            .expect("layer metadata should read");
        assert!(!layer.has_ple);
        assert_eq!(layer.ple_width, None);

        let error = crate::auth_read!(&source, GemmaPleProjectionNormWeightsRequest)
            .expect_err("PLE reads should fail without globals");
        assert!(error.to_string().contains("no global PLE weights"));
    }

    #[test]
    fn invalid_layer_index_fails_with_clear_layer_count() {
        let source = canonical_source();

        let error = crate::auth_read!(&source, GemmaPleLayerMetadataRequest { layer_idx: 1 })
            .expect_err("invalid layer should fail");

        assert!(error.to_string().contains("out of range for 1 layers"));
    }

    #[test]
    fn construction_rejects_ple_layers_without_global_weights() {
        let error = AuthenticatedGemmaPleSource::no_ple(
            "bad-no-ple",
            vec![GemmaPleLayerConfig {
                has_ple: true,
                hidden_width: 3,
            }],
        )
        .expect_err("PLE layer without globals should fail");

        assert!(error
            .to_string()
            .contains("PLE layers but no global PLE weights"));
    }

    #[test]
    fn construction_rejects_fp32_ple_backing() {
        let ple_global = Gemma4PleGlobalWeights::from_materialized(
            vec![MatrixF32 {
                rows: 2,
                cols: 2,
                values: vec![1.0, 0.0, 0.0, 1.0],
            }],
            vec![MatrixF32 {
                rows: 2,
                cols: 3,
                values: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            }],
            vec![1.0, 1.0],
            1.0,
            1.0,
            1.0,
        );

        let error = AuthenticatedGemmaPleSource::from_ple_global(
            "fp32-ple",
            Gemma4ModelProvenance::Fp32,
            vec![GemmaPleLayerConfig {
                has_ple: true,
                hidden_width: 3,
            }],
            Some(ple_global),
            Some(Acc::from_num(0.001)),
        )
        .expect_err("fp32 PLE backing should fail");

        assert!(error.to_string().contains(".detwgt artifact"));
    }

    #[test]
    fn construction_rejects_non_canonical_ple_weights() {
        let ple_global = Gemma4PleGlobalWeights::from_materialized(
            vec![MatrixF32 {
                rows: 2,
                cols: 2,
                values: vec![1.0, 0.0, 0.0, 1.0],
            }],
            vec![MatrixF32 {
                rows: 2,
                cols: 3,
                values: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            }],
            vec![1.0, 1.0],
            1.0,
            1.0,
            1.0,
        );

        let error = AuthenticatedGemmaPleSource::from_ple_global(
            "non-canonical-ple",
            Gemma4ModelProvenance::DetNumWgt,
            vec![GemmaPleLayerConfig {
                has_ple: true,
                hidden_width: 3,
            }],
            Some(ple_global),
            Some(Acc::from_num(0.001)),
        )
        .expect_err("non-canonical PLE backing should fail");

        assert!(error.to_string().contains(".detwgt token embedding source"));
    }

    #[test]
    fn construction_rejects_layer_count_mismatch() {
        let error = AuthenticatedGemmaPleSource::from_canonical_parts(
            "bad-shape",
            vec![
                GemmaPleLayerConfig {
                    has_ple: true,
                    hidden_width: 3,
                },
                GemmaPleLayerConfig {
                    has_ple: false,
                    hidden_width: 3,
                },
            ],
            vec![vec![vec![Act::from_num(1.0), Act::from_num(0.0)]]],
            vec![vec![
                vec![Wgt::from_num(1.0), Wgt::from_num(0.0), Wgt::from_num(0.0)],
                vec![Wgt::from_num(0.0), Wgt::from_num(1.0), Wgt::from_num(0.0)],
            ]],
            vec![Wgt::from_num(1.0), Wgt::from_num(1.0)],
            test_scalars(),
        )
        .expect_err("layer count mismatch should fail");

        assert!(error
            .to_string()
            .contains("token embedding slice count mismatch: 1 vs 2"));
    }

    fn canonical_source() -> AuthenticatedGemmaPleSource {
        AuthenticatedGemmaPleSource::from_canonical_parts(
            "ple-fixture",
            vec![GemmaPleLayerConfig {
                has_ple: true,
                hidden_width: 3,
            }],
            vec![vec![
                vec![Act::from_num(1.0), Act::from_num(0.5)],
                vec![Act::from_num(0.25), Act::from_num(-0.5)],
            ]],
            vec![vec![
                vec![Wgt::from_num(1.0), Wgt::from_num(0.0), Wgt::from_num(-1.0)],
                vec![
                    Wgt::from_num(0.5),
                    Wgt::from_num(0.25),
                    Wgt::from_num(-0.25),
                ],
            ]],
            vec![Wgt::from_num(1.0), Wgt::from_num(0.5)],
            test_scalars(),
        )
        .expect("canonical source should build")
    }

    fn test_scalars() -> GemmaPleScalars {
        GemmaPleScalars {
            embedding_scale: Act::from_num(2.0),
            projection_scalar: Act::from_num(0.5),
            input_scale: Act::from_num(0.25),
            rms_norm_eps: Acc::from_num(0.001),
        }
    }

    fn write_det_sources(
        token_embeddings: Vec<Vec<Act>>,
        model_projection: Vec<Vec<Wgt>>,
    ) -> (PathBuf, DetNumTensorSliceSource, DetNumTensorSliceSource) {
        let unique_suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "raster-prefill-ple-source-{}-{unique_suffix}.detwgt",
            std::process::id()
        ));
        let mut bytes = Vec::new();
        let token_offset = bytes.len();
        for row in &token_embeddings {
            for value in row {
                bytes.extend(value.to_bits().to_le_bytes());
            }
        }
        let projection_offset = bytes.len();
        for row in &model_projection {
            for value in row {
                bytes.extend(value.to_bits().to_le_bytes());
            }
        }
        std::fs::write(&path, bytes).expect("fixture weights should write");

        (
            path.clone(),
            det_source(
                &path,
                token_embeddings.len(),
                token_embeddings[0].len(),
                token_offset,
            ),
            det_source(
                &path,
                model_projection.len(),
                model_projection[0].len(),
                projection_offset,
            ),
        )
    }

    fn det_source(
        path: &Path,
        rows: usize,
        cols: usize,
        data_offset: usize,
    ) -> DetNumTensorSliceSource {
        DetNumTensorSliceSource {
            weights_path: path.to_path_buf(),
            total_rows: rows,
            total_cols: cols,
            data_offset,
            row_offset: 0,
            row_count: rows,
            col_offset: 0,
            col_count: cols,
        }
    }
}
