use anyhow::{bail, Result};

use crate::decode_transition::raster::DecodeLayerCacheSlot;
use crate::prefill_layer::raster::PrefillLayerCacheSlot;
use crate::shared::artifacts::raster_artifact_store::{
    RasterArtifactStoreRoots, RasterTokenIdSequenceRef,
};
use crate::shared::tensors::raster_tensor_artifacts::RasterActivationSequenceRef;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterPrefillOutputRefs {
    pub artifact_store_roots: RasterArtifactStoreRoots,
    pub prompt_token_count: usize,
    pub final_hidden_states_ref: RasterActivationSequenceRef,
    pub layer_caches: Vec<PrefillLayerCacheSlot>,
    pub logits_ref: RasterActivationSequenceRef,
    pub logit_count: usize,
}

impl RasterPrefillOutputRefs {
    pub fn new(
        artifact_store_roots: RasterArtifactStoreRoots,
        prompt_token_count: usize,
        final_hidden_states_ref: RasterActivationSequenceRef,
        layer_caches: Vec<PrefillLayerCacheSlot>,
        logits_ref: RasterActivationSequenceRef,
        logit_count: usize,
    ) -> Result<Self> {
        validate_sequence_row_count(
            &final_hidden_states_ref,
            prompt_token_count,
            "prefill output",
        )?;
        validate_logits_ref(&logits_ref, logit_count, "prefill output")?;
        Ok(Self {
            artifact_store_roots,
            prompt_token_count,
            final_hidden_states_ref,
            layer_caches,
            logits_ref,
            logit_count,
        })
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterDecodeLoopState {
    pub artifact_store_roots: RasterArtifactStoreRoots,
    pub full_token_ids_ref: Option<RasterTokenIdSequenceRef>,
    pub full_token_count: usize,
    pub generated_token_ids_ref: Option<RasterTokenIdSequenceRef>,
    pub generated_token_count: usize,
    pub current_logits_ref: RasterActivationSequenceRef,
    pub logit_count: usize,
    pub layer_caches: Vec<DecodeLayerCacheSlot>,
    pub position: usize,
    pub token_count: usize,
    pub activation_state_ref: Option<RasterActivationSequenceRef>,
}

impl RasterDecodeLoopState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        artifact_store_roots: RasterArtifactStoreRoots,
        full_token_ids_ref: Option<RasterTokenIdSequenceRef>,
        full_token_count: usize,
        generated_token_ids_ref: Option<RasterTokenIdSequenceRef>,
        generated_token_count: usize,
        current_logits_ref: RasterActivationSequenceRef,
        logit_count: usize,
        layer_caches: Vec<DecodeLayerCacheSlot>,
        position: usize,
        token_count: usize,
        activation_state_ref: Option<RasterActivationSequenceRef>,
    ) -> Result<Self> {
        validate_token_ref_count(
            full_token_ids_ref.as_ref(),
            full_token_count,
            "full token ids",
        )?;
        validate_token_ref_count(
            generated_token_ids_ref.as_ref(),
            generated_token_count,
            "generated token ids",
        )?;
        validate_logits_ref(&current_logits_ref, logit_count, "decode loop")?;
        if position != token_count {
            bail!("raster decode loop position/token_count mismatch: {position} vs {token_count}");
        }
        Ok(Self {
            artifact_store_roots,
            full_token_ids_ref,
            full_token_count,
            generated_token_ids_ref,
            generated_token_count,
            current_logits_ref,
            logit_count,
            layer_caches,
            position,
            token_count,
            activation_state_ref,
        })
    }

    pub fn with_roots(mut self, artifact_store_roots: RasterArtifactStoreRoots) -> Self {
        self.artifact_store_roots = artifact_store_roots;
        self
    }
}

impl From<PrefillLayerCacheSlot> for DecodeLayerCacheSlot {
    fn from(cache: PrefillLayerCacheSlot) -> Self {
        match cache {
            PrefillLayerCacheSlot::Empty { num_kv_heads } => {
                DecodeLayerCacheSlot::Empty { num_kv_heads }
            }
            PrefillLayerCacheSlot::Ref(cache_ref) => DecodeLayerCacheSlot::Ref(cache_ref),
        }
    }
}

fn validate_token_ref_count(
    token_ref: Option<&RasterTokenIdSequenceRef>,
    token_count: usize,
    label: &str,
) -> Result<()> {
    match (token_ref, token_count) {
        (Some(token_ref), token_count) if token_ref.token_count() == token_count => Ok(()),
        (Some(token_ref), token_count) => bail!(
            "raster {label} count mismatch: ref has {}, state has {token_count}",
            token_ref.token_count()
        ),
        (None, 0) => Ok(()),
        (None, _) => bail!("raster {label} ref is missing for {token_count} tokens"),
    }
}

fn validate_sequence_row_count(
    sequence_ref: &RasterActivationSequenceRef,
    expected_rows: usize,
    label: &str,
) -> Result<()> {
    let (row_count, _) = sequence_ref.tensor_ref().shape().sequence_metadata()?;
    if row_count != expected_rows {
        bail!("raster {label} row count mismatch: {row_count} vs {expected_rows}");
    }
    Ok(())
}

fn validate_logits_ref(
    logits_ref: &RasterActivationSequenceRef,
    expected_logits: usize,
    label: &str,
) -> Result<()> {
    let (row_count, width) = logits_ref.tensor_ref().shape().sequence_metadata()?;
    let actual_logits = match (row_count, width) {
        (rows, 1) => rows,
        (1, cols) => cols,
        _ => bail!("raster {label} logits shape {row_count}x{width} must be Nx1 or 1xN"),
    };
    if actual_logits != expected_logits {
        bail!("raster {label} logits count mismatch: {actual_logits} vs {expected_logits}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::artifacts::artifact_io::ArtifactIo;
    use crate::shared::artifacts::raster_artifact_store::{
        activation_row_leaf, token_id_leaf, RasterActivationSequenceArtifactRef, RasterArtifactId,
        RasterArtifactMetadata,
    };
    use crate::shared::numerics::det_num::Act;
    use crate::shared::raster_kernels::transformer::RasterActivationRow;
    use crate::shared::tensors::raster_tensor_artifacts::{
        activation_sequence_ref_from_artifact, RasterTensorId,
    };

    #[test]
    fn decode_loop_state_carries_roots_separately_from_refs() {
        ArtifactIo::reset_store();
        let roots = ArtifactIo::export_store_roots();
        let (roots, full_token_ids_ref) = insert_token_ids(roots, "test.full", &[1, 2]).unwrap();
        let (roots, logits_ref) =
            insert_logits(roots, "test.logits", &[Act::from_bits(3)]).unwrap();

        let state = RasterDecodeLoopState::new(
            roots.clone(),
            Some(full_token_ids_ref),
            2,
            None,
            0,
            logits_ref,
            1,
            Vec::new(),
            2,
            2,
            None,
        )
        .unwrap();

        assert_eq!(state.artifact_store_roots, roots);
        assert_eq!(state.full_token_count, 2);
        assert_eq!(state.generated_token_count, 0);
    }

    #[test]
    fn decode_loop_state_rejects_mismatched_token_counts() {
        ArtifactIo::reset_store();
        let roots = ArtifactIo::export_store_roots();
        let (roots, full_token_ids_ref) = insert_token_ids(roots, "test.full", &[1, 2]).unwrap();
        let (roots, logits_ref) =
            insert_logits(roots, "test.logits", &[Act::from_bits(3)]).unwrap();

        let error = RasterDecodeLoopState::new(
            roots,
            Some(full_token_ids_ref),
            3,
            None,
            0,
            logits_ref,
            1,
            Vec::new(),
            2,
            2,
            None,
        )
        .expect_err("mismatched token count should fail");

        assert!(error.to_string().contains("full token ids count mismatch"));
    }

    fn insert_token_ids(
        roots: RasterArtifactStoreRoots,
        source_name: &str,
        token_ids: &[u32],
    ) -> Result<(RasterArtifactStoreRoots, RasterTokenIdSequenceRef)> {
        let leaves = token_ids.iter().copied().map(token_id_leaf).collect();
        let (roots, token_ids_ref) = ArtifactIo::insert_artifact_with_roots(
            &roots,
            RasterArtifactId::new(source_name)?,
            RasterArtifactMetadata::token_ids(token_ids.len()),
            leaves,
        )?;
        Ok((roots, RasterTokenIdSequenceRef::new(token_ids_ref)?))
    }

    fn insert_logits(
        roots: RasterArtifactStoreRoots,
        source_name: &str,
        logits: &[Act],
    ) -> Result<(RasterArtifactStoreRoots, RasterActivationSequenceRef)> {
        let leaves = logits
            .iter()
            .map(|logit| activation_row_leaf(&RasterActivationRow::from_acts(vec![*logit])))
            .collect::<Vec<_>>();
        let (roots, logits_artifact_ref) = ArtifactIo::insert_artifact_with_roots(
            &roots,
            RasterArtifactId::new(source_name)?,
            RasterArtifactMetadata::activation_rows(logits.len(), 1)?,
            leaves,
        )?;
        let logits_ref = activation_sequence_ref_from_artifact(
            RasterTensorId::new(source_name)?,
            RasterActivationSequenceArtifactRef::new(logits_artifact_ref)?,
        )?;
        Ok((roots, logits_ref))
    }
}
