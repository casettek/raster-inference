use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokenizers::Tokenizer;

use crate::input_embedding::raster::auth_source::AuthenticatedGemmaInputEmbeddingSource;
use crate::runtime::checkpoints::{PhaseId, RasterDetourController, RasterDetourSpec, RoutineId};
use crate::runtime::{pipeline, trace};
use crate::shared::api::input::{
    InferenceExecutionMode, InferenceRequest, ModelSpec, PromptPreparationState,
    RasterPromptPreparationState, SamplingConfig,
};
use crate::shared::api::output::OutputDecodeState;
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::integrity_mode::current_raster_integrity_mode;
use crate::shared::artifacts::raster_artifact_store::RasterTokenIdSequenceRef;
use crate::shared::model::gemma_tokenizer::AuthenticatedGemmaTokenizer;
use crate::shared::model::transformer::{Gemma4TransformerModel, TransformerStateTransitionState};
use crate::shared::raster_contracts::pipeline::RasterDecodeLoopState;
use crate::shared::raster_contracts::prefill_layer::AuthenticatedGemmaPrefillLayerSource;
use crate::shared::raster_contracts::prefill_ple::AuthenticatedGemmaPleSource;
use crate::{
    input_embedding, prefill_finalize, prefill_prepare_aux, prefill_range, prompt_prepare,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InputEmbeddingState {
    #[serde(flatten)]
    pub prompt_preparation: PromptPreparationState,
    pub embedded_prompt_activations_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub det_embedded_prompt_activations_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InferenceState {
    pub input_embedding: InputEmbeddingState,
    pub transformer_state_transition: TransformerStateTransitionState,
    pub output_decode: OutputDecodeState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raster_tile_invocations: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InferenceControls {
    pub commit_checkpoints: bool,
    pub terminal_checkpoint: Option<String>,
    pub raster: bool,
    pub raster_detour: Option<RasterDetourSpec>,
    pub raster_tokenizer_source: Option<AuthenticatedGemmaTokenizer>,
    pub raster_projection_rows_per_tile: Option<usize>,
    pub raster_attention_kv_rows_per_tile: Option<usize>,
    pub raster_sequence_rows_per_tile: Option<usize>,
    pub raster_head_rows_per_tile: Option<usize>,
    pub prefill_token_range_width: Option<usize>,
    pub decode_layer_range_width: Option<usize>,
    pub raster_tokenizer_bpe_pairs_per_tile: Option<usize>,
    pub raster_tokenizer_bpe_pieces_per_tile: Option<usize>,
    pub raster_output_byte_flush_bytes_per_tile: Option<usize>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct RasterSizingControls {
    pub projection_rows_per_tile: usize,
    pub attention_kv_rows_per_tile: usize,
    pub sequence_rows_per_tile: usize,
    pub head_rows_per_tile: usize,
    pub prefill_token_range_width: usize,
    pub decode_layer_range_width: usize,
    pub tokenizer_bpe_pairs_per_tile: usize,
    pub tokenizer_bpe_pieces_per_tile: usize,
    pub output_byte_flush_bytes_per_tile: usize,
}

impl InferenceControls {
    pub const DEFAULT_RASTER_PROJECTION_ROWS_PER_TILE: usize = 1;
    pub const DEFAULT_RASTER_ATTENTION_KV_ROWS_PER_TILE: usize = 32;
    pub const DEFAULT_RASTER_SEQUENCE_ROWS_PER_TILE: usize = 1;
    pub const DEFAULT_RASTER_HEAD_ROWS_PER_TILE: usize = 1;
    pub const DEFAULT_PREFILL_TOKEN_RANGE_WIDTH: usize = usize::MAX;
    pub const DEFAULT_DECODE_LAYER_RANGE_WIDTH: usize = usize::MAX;
    pub const DEFAULT_RASTER_TOKENIZER_BPE_PAIRS_PER_TILE: usize =
        crate::prompt_prepare::raster::DEFAULT_BPE_PAIRS_PER_TILE;
    pub const DEFAULT_RASTER_TOKENIZER_BPE_PIECES_PER_TILE: usize =
        crate::prompt_prepare::raster::DEFAULT_BPE_PIECES_PER_TILE;
    pub const DEFAULT_RASTER_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE: usize =
        crate::output_finalize::raster::DEFAULT_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE;

    pub fn raster_projection_rows_per_tile(&self) -> Result<usize> {
        match self.raster_projection_rows_per_tile {
            Some(0) => anyhow::bail!("raster projection rows per tile must be greater than zero"),
            Some(rows) => Ok(rows),
            None => Ok(Self::DEFAULT_RASTER_PROJECTION_ROWS_PER_TILE),
        }
    }

    pub fn raster_attention_kv_rows_per_tile(&self) -> Result<usize> {
        match self.raster_attention_kv_rows_per_tile {
            Some(0) => {
                anyhow::bail!("raster attention KV rows per tile must be greater than zero")
            }
            Some(rows) => Ok(rows),
            None => Ok(Self::DEFAULT_RASTER_ATTENTION_KV_ROWS_PER_TILE),
        }
    }

    pub fn raster_sequence_rows_per_tile(&self) -> Result<usize> {
        match self.raster_sequence_rows_per_tile {
            Some(0) => anyhow::bail!("raster sequence rows per tile must be greater than zero"),
            Some(rows) => Ok(rows),
            None => Ok(Self::DEFAULT_RASTER_SEQUENCE_ROWS_PER_TILE),
        }
    }

    pub fn raster_head_rows_per_tile(&self) -> Result<usize> {
        match self.raster_head_rows_per_tile {
            Some(0) => anyhow::bail!("raster head rows per tile must be greater than zero"),
            Some(rows) => Ok(rows),
            None => Ok(Self::DEFAULT_RASTER_HEAD_ROWS_PER_TILE),
        }
    }

    pub fn prefill_token_range_width(&self) -> Result<usize> {
        match self.prefill_token_range_width {
            Some(0) => anyhow::bail!("prefill token range width must be greater than zero"),
            Some(width) => Ok(width),
            None => Ok(Self::DEFAULT_PREFILL_TOKEN_RANGE_WIDTH),
        }
    }

    pub fn decode_layer_range_width(&self) -> Result<usize> {
        match self.decode_layer_range_width {
            Some(0) => anyhow::bail!("decode layer range width must be greater than zero"),
            Some(width) => Ok(width),
            None => Ok(Self::DEFAULT_DECODE_LAYER_RANGE_WIDTH),
        }
    }

    pub fn raster_tokenizer_bpe_pairs_per_tile(&self) -> Result<usize> {
        match self.raster_tokenizer_bpe_pairs_per_tile {
            Some(0) => {
                anyhow::bail!("raster tokenizer BPE pairs per tile must be greater than zero")
            }
            Some(pairs) => Ok(pairs),
            None => Ok(Self::DEFAULT_RASTER_TOKENIZER_BPE_PAIRS_PER_TILE),
        }
    }

    pub fn raster_tokenizer_bpe_pieces_per_tile(&self) -> Result<usize> {
        match self.raster_tokenizer_bpe_pieces_per_tile {
            Some(0) => {
                anyhow::bail!("raster tokenizer BPE pieces per tile must be greater than zero")
            }
            Some(pieces) => Ok(pieces),
            None => Ok(Self::DEFAULT_RASTER_TOKENIZER_BPE_PIECES_PER_TILE),
        }
    }

    pub fn raster_output_byte_flush_bytes_per_tile(&self) -> Result<usize> {
        match self.raster_output_byte_flush_bytes_per_tile {
            Some(0) => {
                anyhow::bail!("raster output byte flush bytes per tile must be greater than zero")
            }
            Some(bytes) => Ok(bytes),
            None => Ok(Self::DEFAULT_RASTER_OUTPUT_BYTE_FLUSH_BYTES_PER_TILE),
        }
    }

    pub fn raster_sizing_controls(&self) -> Result<RasterSizingControls> {
        Ok(RasterSizingControls {
            projection_rows_per_tile: self.raster_projection_rows_per_tile()?,
            attention_kv_rows_per_tile: self.raster_attention_kv_rows_per_tile()?,
            sequence_rows_per_tile: self.raster_sequence_rows_per_tile()?,
            head_rows_per_tile: self.raster_head_rows_per_tile()?,
            prefill_token_range_width: self.prefill_token_range_width()?,
            decode_layer_range_width: self.decode_layer_range_width()?,
            tokenizer_bpe_pairs_per_tile: self.raster_tokenizer_bpe_pairs_per_tile()?,
            tokenizer_bpe_pieces_per_tile: self.raster_tokenizer_bpe_pieces_per_tile()?,
            output_byte_flush_bytes_per_tile: self.raster_output_byte_flush_bytes_per_tile()?,
        })
    }

    fn terminal_checkpoint_spec(&self) -> Result<Option<trace::TerminalCheckpointSpec>> {
        let spec = self
            .terminal_checkpoint
            .as_deref()
            .map(trace::TerminalCheckpointSpec::parse)
            .transpose()?;
        if spec
            .as_ref()
            .is_some_and(|checkpoint| checkpoint.checkpoint_id() == "decode.layer_range")
        {
            anyhow::bail!(
                "terminal checkpoint decode.layer_range is not supported because paused state cannot yet carry an in-progress decode transition"
            );
        }
        Ok(spec)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PausedInferenceState {
    pub terminal_checkpoint_id: String,
    pub input_embedding: InputEmbeddingState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transformer_state_transition: Option<TransformerStateTransitionState>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_decode: Option<OutputDecodeState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raster_tile_invocations: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RasterPromptPreparedState {
    pub terminal_checkpoint_id: String,
    pub prompt_preparation: RasterPromptPreparationState,
    pub sampling: SamplingConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raster_tile_invocations: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum InferenceRunOutcome {
    Completed(InferenceState),
    Paused(PausedInferenceState),
    RasterPromptPrepared(RasterPromptPreparedState),
}

pub fn run_inference(
    request: &InferenceRequest,
    model: &ModelSpec,
    tokenizer: &Tokenizer,
    transformer_model: &Gemma4TransformerModel,
) -> Result<InferenceState> {
    match run_inference_with_controls(
        request,
        model,
        tokenizer,
        transformer_model,
        &InferenceControls::default(),
    )? {
        InferenceRunOutcome::Completed(state) => Ok(state),
        InferenceRunOutcome::Paused(paused) => anyhow::bail!(
            "inference paused unexpectedly at checkpoint {}",
            paused.terminal_checkpoint_id
        ),
        InferenceRunOutcome::RasterPromptPrepared(state) => anyhow::bail!(
            "raster inference stopped at unsupported routine boundary {}",
            state.terminal_checkpoint_id
        ),
    }
}

pub fn run_inference_with_controls(
    request: &InferenceRequest,
    model: &ModelSpec,
    tokenizer: &Tokenizer,
    transformer_model: &Gemma4TransformerModel,
    controls: &InferenceControls,
) -> Result<InferenceRunOutcome> {
    let terminal_checkpoint = controls.terminal_checkpoint_spec()?;
    trace::with_terminal_checkpoint(terminal_checkpoint.clone(), || {
        trace::with_checkpointing_enabled(controls.commit_checkpoints, || {
            if controls.raster && request.execution_mode != InferenceExecutionMode::Deterministic {
                anyhow::bail!("raster tile inference requires deterministic execution");
            }
            if controls.raster && controls.raster_detour.is_some() {
                anyhow::bail!("--raster and selective raster detour cannot be used together");
            }
            if controls.raster_detour.is_some()
                && request.execution_mode != InferenceExecutionMode::Deterministic
            {
                anyhow::bail!("selective raster detour requires deterministic execution");
            }
            let mut raster_detour_controller = RasterDetourController::new(controls.raster_detour);
            let use_raster_prefill = controls.raster;
            let use_raster_decode = controls.raster;
            let count_raster_tiles = use_raster_decode || raster_detour_controller.is_active();
            let raster_sizing_controls = if controls.raster
                || raster_detour_controller.is_active()
                || controls.prefill_token_range_width.is_some()
                || controls.decode_layer_range_width.is_some()
            {
                Some(controls.raster_sizing_controls()?)
            } else {
                None
            };
            transformer_model.validate_execution_mode(request.execution_mode)?;
            trace::start_inference_trace(&json!({
                "model_id": model.model_id,
                "execution_mode": request.execution_mode,
                "det_num_spec_version": crate::shared::numerics::det_num::DET_NUM_SPEC_VERSION,
                "model_provenance": format!("{:?}", transformer_model.provenance),
                "prompt_bytes_sha256": trace::sha256_hex(&request.prompt_bytes),
                "max_new_tokens": request.sampling.max_new_tokens,
                "transformer_layer_count": transformer_model.layers.len(),
                "terminal_checkpoint": terminal_checkpoint.as_ref().map(|checkpoint| checkpoint.checkpoint_id()),
                "terminal_checkpoint_occurrence": terminal_checkpoint.as_ref().map(|checkpoint| checkpoint.occurrence()),
                "commit_checkpoints": controls.commit_checkpoints,
                "tile_dsl_mode": if use_raster_prefill { "raster" } else { "native" },
                "raster_detour": raster_detour_controller.selected_spec().map(|spec| spec.to_string()),
                "raster_integrity_mode": current_raster_integrity_mode().label(),
                "raster_sizing_controls": raster_sizing_controls,
            }));
            if count_raster_tiles {
                crate::dsl::start_tile_invocation_counting();
            }
            let selected_raster_detour_routine = raster_detour_controller
                .selected_spec()
                .map(|spec| spec.routine_id());
            let deterministic_prompt_checkpoint = |prompt_preparation: &PromptPreparationState| {
                let tokenizer_source = controls.raster_tokenizer_source.as_ref().context(
                    "deterministic CPU prompt.prepare checkpoint requires an authenticated Gemma tokenizer",
                )?;
                prompt_prepare::format_native_prompt_as_raster_checkpoint_for_trace(
                    request,
                    model,
                    tokenizer_source,
                    prompt_preparation,
                )
            };

            let result = (|| {
                let mut raster_prompt_preparation_for_embedding = None;
                let mut raster_prompt_preparation_roots_for_embedding = None;
                raster_detour_controller
                    .reject_if_selected_unsupported(RoutineId::PromptPrepare)?;
                let prompt_preparation = if use_raster_prefill {
                    let tokenizer_source = controls.raster_tokenizer_source.as_ref().context(
                        "raster tile inference requires an authenticated Gemma tokenizer",
                    )?;
                    let raster_sizing_controls = raster_sizing_controls
                        .as_ref()
                        .expect("raster sizing controls should be validated");
                    let raster_prompt_preparation = prompt_prepare::run_raster(
                        request,
                        model,
                        tokenizer_source,
                        *raster_sizing_controls,
                    )?;
                    trace::trace_checkpoint(
                        "prompt.prepare",
                        &json!({
                            "prompt_bytes_root": raster_prompt_preparation.state.prompt_bytes_root.clone(),
                            "prompt_text_root": raster_prompt_preparation.state.prompt_text_root.clone(),
                            "rendered_prompt_root": raster_prompt_preparation.state.rendered_prompt_root.clone(),
                            "normalized_prompt_root": raster_prompt_preparation.state.normalized_prompt_root.clone(),
                            "prompt_token_count": raster_prompt_preparation.state.prompt_token_count,
                            "prompt_token_ids_root": raster_prompt_preparation.state.prompt_token_ids_root.clone(),
                            "sampling": request.sampling.clone(),
                        }),
                    );
                    if let Some(terminal_checkpoint_id) = reached_terminal_checkpoint_id(controls) {
                        return Ok(InferenceRunOutcome::RasterPromptPrepared(
                            RasterPromptPreparedState {
                                terminal_checkpoint_id,
                                prompt_preparation: raster_prompt_preparation.state,
                                sampling: request.sampling.clone(),
                                raster_tile_invocations: None,
                            },
                        ));
                    }
                    raster_prompt_preparation_roots_for_embedding =
                        Some(raster_prompt_preparation.artifact_store_roots.clone());
                    raster_prompt_preparation_for_embedding =
                        Some(raster_prompt_preparation.state.clone());
                    prompt_preparation_from_raster_prompt(
                        request,
                        &raster_prompt_preparation.state,
                    )?
                } else {
                    let prompt_preparation = prompt_prepare::run(request, model, tokenizer)?;
                    if request.execution_mode == InferenceExecutionMode::Deterministic
                        && controls.raster_tokenizer_source.is_some()
                    {
                        let prompt_checkpoint =
                            deterministic_prompt_checkpoint(&prompt_preparation)?;
                        trace::trace_checkpoint(
                            "prompt.prepare",
                            &json!({
                                "prompt_text": prompt_preparation.prompt_text.clone(),
                                "prompt_token_ids": prompt_preparation.prompt_token_ids.clone(),
                                "prompt_token_ids_sha256": prompt_preparation.prompt_token_ids_sha256.clone(),
                                "sampling": request.sampling.clone(),
                            }),
                        );
                        if let Some(terminal_checkpoint_id) =
                            reached_terminal_checkpoint_id(controls)
                        {
                            return Ok(InferenceRunOutcome::RasterPromptPrepared(
                                RasterPromptPreparedState {
                                    terminal_checkpoint_id,
                                    prompt_preparation: prompt_checkpoint,
                                    sampling: request.sampling.clone(),
                                    raster_tile_invocations: None,
                                },
                            ));
                        }
                        raster_prompt_preparation_roots_for_embedding =
                            Some(ArtifactIo::export_store_roots());
                        raster_prompt_preparation_for_embedding = Some(prompt_checkpoint);
                    }
                    prompt_preparation
                };
                let detour_input_embedding =
                    raster_detour_controller.should_detour(RoutineId::InputEmbedding);
                let (token_embeddings, raster_input_embedding_refs) = if use_raster_prefill {
                    let raster_prompt_preparation = raster_prompt_preparation_for_embedding
                        .as_ref()
                        .expect("raster prompt preparation should exist for raster prefill");
                    let raster_prompt_preparation_roots =
                        raster_prompt_preparation_roots_for_embedding
                            .clone()
                            .expect(
                                "raster prompt preparation roots should exist for raster prefill",
                            );
                    let embedding_source = AuthenticatedGemmaInputEmbeddingSource::from_model(
                        model.model_id.clone(),
                        transformer_model,
                    )?;
                    let input_embedding_output = input_embedding::run_raster(
                        raster_prompt_preparation_roots,
                        raster_prompt_preparation,
                        &embedding_source,
                    )?;
                    let token_embeddings =
                        input_embedding::materialize_input_embedding_refs_for_trace(
                            &input_embedding_output.refs,
                        )?;
                    (token_embeddings, Some(input_embedding_output))
                } else if detour_input_embedding {
                    let raster_prompt_preparation = raster_prompt_preparation_for_embedding
                        .as_ref()
                        .context(
                            "selective raster input.embedding detour requires an authenticated Gemma tokenizer",
                        )?;
                    let raster_prompt_preparation_roots =
                        raster_prompt_preparation_roots_for_embedding.clone().context(
                            "selective raster input.embedding detour requires prompt artifact roots",
                        )?;
                    let embedding_source = AuthenticatedGemmaInputEmbeddingSource::from_model(
                        model.model_id.clone(),
                        transformer_model,
                    )?;
                    let input_embedding_output = input_embedding::run_raster(
                        raster_prompt_preparation_roots,
                        raster_prompt_preparation,
                        &embedding_source,
                    )?;
                    let token_embeddings =
                        input_embedding::materialize_input_embedding_refs_for_trace(
                            &input_embedding_output.refs,
                        )?;
                    (token_embeddings, Some(input_embedding_output))
                } else {
                    let token_embeddings = input_embedding::run(
                        &prompt_preparation.prompt_token_ids,
                        transformer_model,
                        request.execution_mode,
                    )?;
                    let input_embedding_output = raster_prompt_preparation_for_embedding
                        .as_ref()
                        .map(|raster_prompt_preparation| {
                            let embedding_source =
                                AuthenticatedGemmaInputEmbeddingSource::from_model(
                                    model.model_id.clone(),
                                    transformer_model,
                                )?;
                            let embedding_source_ref = embedding_source.committed_source_ref()?;
                            input_embedding::format_native_input_embedding_as_raster_checkpoint_for_trace(
                                model.model_id.clone(),
                                embedding_source_ref.root().to_string(),
                                raster_prompt_preparation,
                                &token_embeddings,
                            )
                        })
                        .transpose()?;
                    (token_embeddings, input_embedding_output)
                };
                let input_embedding = InputEmbeddingState {
                    prompt_preparation: prompt_preparation.clone(),
                    embedded_prompt_activations_sha256: token_embeddings.activations_sha256.clone(),
                    det_embedded_prompt_activations_sha256: token_embeddings
                        .det_activations_sha256
                        .clone(),
                };
                if request.execution_mode != InferenceExecutionMode::Deterministic {
                    trace::trace_checkpoint(
                        "prompt.prepare",
                        &json!({
                            "prompt_text": prompt_preparation.prompt_text.clone(),
                            "prompt_token_ids": prompt_preparation.prompt_token_ids.clone(),
                            "prompt_token_ids_sha256": prompt_preparation.prompt_token_ids_sha256.clone(),
                            "embedded_prompt_activations": token_embeddings.activations.clone(),
                            "embedded_prompt_activations_sha256": token_embeddings.activations_sha256.clone(),
                            "det_embedded_prompt_activations_sha256": token_embeddings.det_activations_sha256.clone(),
                            "sampling": request.sampling.clone(),
                        }),
                    );
                }
                let input_embedding_raster_refs_for_checkpoint = if use_raster_prefill
                    || selected_raster_detour_routine == Some(RoutineId::InputEmbedding)
                {
                    raster_input_embedding_refs
                        .as_ref()
                        .map(|output| &output.refs)
                } else {
                    None
                };
                input_embedding::trace_input_embedding_checkpoint(
                    &prompt_preparation.prompt_token_ids,
                    &token_embeddings,
                    input_embedding_raster_refs_for_checkpoint,
                );
                if let Some(terminal_checkpoint_id) = reached_terminal_checkpoint_id(controls) {
                    return Ok(InferenceRunOutcome::Paused(PausedInferenceState {
                        terminal_checkpoint_id,
                        input_embedding,
                        transformer_state_transition: None,
                        output_decode: None,
                        raster_tile_invocations: None,
                    }));
                }
                let mut raster_decode_state_for_output = None;
                let prefill = if use_raster_prefill {
                    let raster_sizing =
                        raster_sizing_controls.expect("raster sizing controls should be validated");
                    raster_detour_controller
                        .reject_if_selected_unsupported(RoutineId::PrefillPrepareAux)?;
                    let ple_source = AuthenticatedGemmaPleSource::from_model(
                        model.model_id.clone(),
                        transformer_model,
                    )?;
                    let input_embedding_output = raster_input_embedding_refs
                        .as_ref()
                        .expect("raster prefill requires raster input embedding refs");
                    let ple_output = prefill_prepare_aux::run_raster(
                        input_embedding_output.artifact_store_roots.clone(),
                        &input_embedding_output.refs,
                        &ple_source,
                        raster_sizing,
                    )?;
                    let (layer_roots, ple_input_manifest_root) = ple_output.into_parts();
                    if let Some(terminal_checkpoint_id) = reached_terminal_checkpoint_id(controls) {
                        trace::phase_paused(PhaseId::TransformerStateTransition);
                        return Ok(InferenceRunOutcome::Paused(PausedInferenceState {
                            terminal_checkpoint_id,
                            input_embedding,
                            transformer_state_transition: None,
                            output_decode: None,
                            raster_tile_invocations: None,
                        }));
                    }
                    let layer_source =
                    crate::shared::raster_contracts::prefill_layer::AuthenticatedGemmaPrefillLayerSource::from_model(
                        model.model_id.clone(),
                        transformer_model,
                    )?;
                    raster_detour_controller
                        .reject_if_selected_unsupported(RoutineId::PrefillRangeFinalize)?;
                    let (layer_roots, layer_refs) = prefill_range::run_raster(
                        layer_roots,
                        &input_embedding_output.refs,
                        &layer_source,
                        ple_input_manifest_root.as_deref(),
                        raster_sizing,
                    )?;
                    if let Some(terminal_checkpoint_id) = reached_terminal_checkpoint_id(controls) {
                        trace::phase_paused(PhaseId::TransformerStateTransition);
                        return Ok(InferenceRunOutcome::Paused(PausedInferenceState {
                            terminal_checkpoint_id,
                            input_embedding,
                            transformer_state_transition: None,
                            output_decode: None,
                            raster_tile_invocations: None,
                        }));
                    }
                    let finalize_source =
                    crate::prefill_finalize::raster::auth_source::AuthenticatedGemmaPrefillFinalizeSource::from_model(
                        model.model_id.clone(),
                        transformer_model,
                    )?;
                    raster_detour_controller
                        .reject_if_selected_unsupported(RoutineId::PrefillFinalize)?;
                    let prefill_output = prefill_finalize::run_raster(
                        layer_roots,
                        prompt_preparation.prompt_token_ids.len(),
                        &finalize_source,
                        layer_refs.final_hidden_states_ref,
                        layer_refs.layer_caches,
                        raster_sizing.projection_rows_per_tile,
                    )?;
                    let full_token_ids_ref =
                        RasterTokenIdSequenceRef::new(ArtifactIo::artifact_ref_for_root(
                            &raster_prompt_preparation_for_embedding
                                .as_ref()
                                .expect("raster prompt preparation should exist for raster prefill")
                                .prompt_token_ids_root,
                        )?)?;
                    raster_decode_state_for_output = Some(RasterDecodeLoopState::new(
                        prefill_output.artifact_store_roots.clone(),
                        Some(full_token_ids_ref),
                        prompt_preparation.prompt_token_ids.len(),
                        None,
                        0,
                        prefill_output.refs.logits_ref.clone(),
                        prefill_output.refs.logit_count,
                        prefill_output
                            .refs
                            .layer_caches
                            .iter()
                            .cloned()
                            .map(Into::into)
                            .collect(),
                        prompt_preparation.prompt_token_ids.len(),
                        prompt_preparation.prompt_token_ids.len(),
                        Some(prefill_output.refs.final_hidden_states_ref.clone()),
                    )?);
                    prefill_finalize::materialize_raster_output_refs_for_api(&prefill_output)?
                } else {
                    let detour_prefill_prepare_aux =
                        raster_detour_controller.should_detour(RoutineId::PrefillPrepareAux);
                    let ple_inputs = if detour_prefill_prepare_aux {
                        let input_embedding_output = raster_input_embedding_refs.as_ref().context(
                            "selective raster prefill.prepare_aux detour requires input embedding raster refs",
                        )?;
                        let raster_sizing = raster_sizing_controls
                            .expect("raster sizing controls should be validated");
                        let ple_source = AuthenticatedGemmaPleSource::from_model(
                            model.model_id.clone(),
                            transformer_model,
                        )?;
                        let ple_output = prefill_prepare_aux::run_raster(
                            input_embedding_output.artifact_store_roots.clone(),
                            &input_embedding_output.refs,
                            &ple_source,
                            raster_sizing,
                        )?;
                        let (artifact_store_roots, ple_input_manifest_root) =
                            ple_output.into_parts();
                        let ple_input_refs =
                            prefill_prepare_aux::prefill_ple_input_refs_from_manifest(
                                artifact_store_roots,
                                ple_input_manifest_root.as_deref(),
                            )?;
                        prefill_prepare_aux::materialize_prefill_ple_inputs(
                            ple_input_refs.as_ref(),
                        )?
                    } else {
                        prefill_prepare_aux::run(
                            &prompt_preparation.prompt_token_ids,
                            transformer_model,
                            &token_embeddings,
                            request.execution_mode,
                        )?
                    };
                    if let Some(terminal_checkpoint_id) = reached_terminal_checkpoint_id(controls) {
                        trace::phase_paused(PhaseId::TransformerStateTransition);
                        return Ok(InferenceRunOutcome::Paused(PausedInferenceState {
                            terminal_checkpoint_id,
                            input_embedding,
                            transformer_state_transition: None,
                            output_decode: None,
                            raster_tile_invocations: None,
                        }));
                    }
                    let prefill_layer_source = if raster_detour_controller
                        .selected_spec()
                        .is_some_and(|spec| spec.routine_id() == RoutineId::PrefillRange)
                    {
                        Some(AuthenticatedGemmaPrefillLayerSource::from_model(
                            model.model_id.clone(),
                            transformer_model,
                        )?)
                    } else {
                        None
                    };
                    let prefill_layer_raster_detour =
                        prefill_layer_source.as_ref().map(|layer_source| {
                            prefill_range::PrefillLayerRasterDetour {
                                layer_source,
                                raster_sizing: raster_sizing_controls
                                    .expect("raster sizing controls should be validated"),
                            }
                        });
                    let (final_hidden_states, layer_caches) =
                        prefill_range::run_with_mode_internal_with_detour(
                            token_embeddings.clone_internal(),
                            transformer_model,
                            ple_inputs.as_ref(),
                            request.execution_mode,
                            Some(&mut raster_detour_controller),
                            prefill_layer_raster_detour,
                            raster_sizing_controls
                                .map(|controls| controls.prefill_token_range_width)
                                .unwrap_or_else(|| {
                                    controls
                                        .prefill_token_range_width()
                                        .expect("prefill token range width should validate")
                                }),
                        )?;
                    if let Some(terminal_checkpoint_id) = reached_terminal_checkpoint_id(controls) {
                        trace::phase_paused(PhaseId::TransformerStateTransition);
                        return Ok(InferenceRunOutcome::Paused(PausedInferenceState {
                            terminal_checkpoint_id,
                            input_embedding,
                            transformer_state_transition: None,
                            output_decode: None,
                            raster_tile_invocations: None,
                        }));
                    }
                    if raster_detour_controller.should_detour(RoutineId::PrefillFinalize) {
                        let raster_sizing = raster_sizing_controls
                            .expect("raster sizing controls should be validated");
                        prefill_finalize::run_selected_raster_detour_from_native_boundary(
                            model.model_id.clone(),
                            transformer_model,
                            prompt_preparation.prompt_token_ids.len(),
                            final_hidden_states,
                            layer_caches,
                            raster_sizing.projection_rows_per_tile,
                        )?
                    } else {
                        prefill_finalize::run(
                            &prompt_preparation.prompt_token_ids,
                            transformer_model,
                            final_hidden_states,
                            layer_caches,
                            request.execution_mode,
                        )?
                    }
                };
                let mut transformer_state_transition = prefill.transformer_state.clone();
                if let Some(terminal_checkpoint_id) = reached_terminal_checkpoint_id(controls) {
                    trace::phase_paused(PhaseId::TransformerStateTransition);
                    return Ok(InferenceRunOutcome::Paused(PausedInferenceState {
                        terminal_checkpoint_id,
                        input_embedding,
                        transformer_state_transition: Some(transformer_state_transition),
                        output_decode: None,
                        raster_tile_invocations: None,
                    }));
                }
                let output_decode = if let Some(raster_decode_state) =
                    raster_decode_state_for_output
                {
                    let tokenizer_source = controls.raster_tokenizer_source.as_ref().context(
                        "raster tile inference requires an authenticated Gemma tokenizer",
                    )?;
                    pipeline::run_output_decode_with_raster_state(
                        raster_decode_state,
                        &request.sampling,
                        tokenizer_source,
                        transformer_model,
                        raster_sizing_controls.expect("raster sizing controls should be validated"),
                    )?
                } else {
                    pipeline::run_output_decode_with_mode_and_detour(
                        &prompt_preparation.prompt_token_ids,
                        &prefill,
                        &request.sampling,
                        tokenizer,
                        controls.raster_tokenizer_source.as_ref(),
                        transformer_model,
                        request.execution_mode,
                        Some(&mut raster_detour_controller),
                        raster_sizing_controls,
                    )?
                };
                transformer_state_transition
                    .activation_states
                    .extend(output_decode.decode_transition_states.iter().cloned());
                if let Some(terminal_checkpoint_id) = reached_terminal_checkpoint_id(controls) {
                    trace::phase_paused(PhaseId::OutputDecode);
                    return Ok(InferenceRunOutcome::Paused(PausedInferenceState {
                        terminal_checkpoint_id,
                        input_embedding,
                        transformer_state_transition: Some(transformer_state_transition),
                        output_decode: Some(output_decode),
                        raster_tile_invocations: None,
                    }));
                }
                raster_detour_controller.ensure_matched_if_active()?;
                Ok(InferenceRunOutcome::Completed(InferenceState {
                    input_embedding,
                    transformer_state_transition,
                    output_decode,
                    raster_tile_invocations: None,
                }))
            })();

            let raster_tile_invocations = count_raster_tiles
                .then(crate::dsl::stop_tile_invocation_counting)
                .flatten();
            if let Some(total) = raster_tile_invocations {
                trace::raster_tile_invocations_finished(total);
            }

            let mut result = result;
            if let Some(total) = raster_tile_invocations {
                match &mut result {
                    Ok(InferenceRunOutcome::Completed(state)) => {
                        state.raster_tile_invocations = Some(total);
                    }
                    Ok(InferenceRunOutcome::Paused(state)) => {
                        state.raster_tile_invocations = Some(total);
                    }
                    Ok(InferenceRunOutcome::RasterPromptPrepared(state)) => {
                        state.raster_tile_invocations = Some(total);
                    }
                    Err(_) => {}
                }
            }

            match &result {
                Ok(InferenceRunOutcome::Completed(state)) => {
                    trace::finish_inference_trace(&json!({
                        "input_embedding_prompt_token_ids_sha256": state.input_embedding.prompt_preparation.prompt_token_ids_sha256,
                        "output_decode_generated_token_ids_sha256": state.output_decode.generated_token_ids_sha256,
                        "output_decode_generated_text_sha256": trace::sha256_hex(&state.output_decode.generated_text),
                        "generated_token_count": state.output_decode.generated_token_count,
                        "raster_tile_invocations": state.raster_tile_invocations,
                    }))
                }
                Ok(InferenceRunOutcome::Paused(state)) => trace::finish_inference_trace(&json!({
                    "terminal_checkpoint_id": state.terminal_checkpoint_id,
                    "input_embedding_prompt_token_ids_sha256": state.input_embedding.prompt_preparation.prompt_token_ids_sha256,
                    "raster_tile_invocations": state.raster_tile_invocations,
                })),
                Ok(InferenceRunOutcome::RasterPromptPrepared(state)) => {
                    trace::finish_inference_trace(&json!({
                        "terminal_checkpoint_id": state.terminal_checkpoint_id,
                        "prompt_token_ids_root": state.prompt_preparation.prompt_token_ids_root,
                        "prompt_token_count": state.prompt_preparation.prompt_token_count,
                        "raster_tile_invocations": state.raster_tile_invocations,
                    }))
                }
                Err(error) => trace::abort_inference_trace(error),
            }

            result
        })
    })
}

fn reached_terminal_checkpoint_id(controls: &InferenceControls) -> Option<String> {
    controls.terminal_checkpoint.as_ref()?;
    trace::reached_terminal_checkpoint_id()
}

fn prompt_preparation_from_raster_prompt(
    request: &InferenceRequest,
    raster_prompt: &RasterPromptPreparationState,
) -> Result<PromptPreparationState> {
    let prompt_text = prompt_prepare::native::decode_prompt_bytes(
        &request.prompt_bytes,
        request.text_decoding_policy,
    )?;
    let prompt_token_ids = materialize_raster_prompt_token_ids(
        &raster_prompt.prompt_token_ids_root,
        raster_prompt.prompt_token_count,
    )?;
    let prompt_token_ids_sha256 =
        prompt_prepare::native::build_prompt_commitment(&prompt_token_ids)?;

    Ok(PromptPreparationState {
        prompt_text,
        prompt_token_ids,
        prompt_token_ids_sha256,
    })
}

fn materialize_raster_prompt_token_ids(
    token_ids_root: &str,
    token_count: usize,
) -> Result<Vec<u32>> {
    let token_ids_ref =
        RasterTokenIdSequenceRef::new(ArtifactIo::artifact_ref_for_root(token_ids_root)?)?;
    if token_ids_ref.token_count() != token_count {
        anyhow::bail!(
            "raster prompt token count mismatch: root has {}, checkpoint expected {}",
            token_ids_ref.token_count(),
            token_count
        );
    }

    (0..token_count)
        .map(|token_idx| {
            ArtifactIo::read_authenticated_leaf(token_ids_ref.artifact_ref(), token_idx)?
                .deserialize()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use tokenizers::{models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace};

    use crate::prefill_finalize::raster::auth_source::AuthenticatedGemmaPrefillFinalizeSource;
    use crate::shared::model::gemma_tokenizer::GemmaAddedToken;
    use crate::shared::model::transformer::{
        DetNumMatrix, DetNumTensorSliceSource, Gemma4LayerMatrixSource, GemmaEmbeddingTensorSource,
        InternalActivationSequence, InternalLogits,
    };
    use crate::shared::numerics::det_num::{f32_to_acc, Act, Wgt};
    use crate::shared::raster_contracts::prefill_layer::AuthenticatedGemmaPrefillLayerSource;
    use crate::shared::raster_kernels::transformer::RasterActivationSequence;
    use crate::shared::tensors::raster_tensor_artifacts::insert_activation_sequence_artifact_ref;
    use crate::{
        decode_step_with_mode, embed_input_tokens, finalize_decode_transition,
        run_decode_select_token, run_inference, run_inference_with_controls, run_output_finalize,
        run_prefill_finalize, run_prefill_prepare_aux, run_prefill_range, run_prompt_prepare,
        AuthenticatedGemmaTokenizer, DecodeState, EmbeddingTable, Gemma4AttentionKind,
        Gemma4LayerWeights, Gemma4LogitsProjection, Gemma4ModelProvenance, Gemma4PleGlobalWeights,
        Gemma4PleLayerWeights, Gemma4TransformerModel, GemmaBpeMerge, GemmaTokenizerSpec,
        GemmaVocabEntry, InferenceControls, InferenceExecutionMode, InferenceRequest,
        InferenceRunOutcome, MatrixF32, ModelSpec, OutputDecodeStopReason, RasterDetourSpec,
        SamplingConfig, TextDecodingPolicy,
    };
    use std::{
        env, fs, process,
        sync::{
            atomic::{AtomicU64, Ordering},
            Arc, Mutex,
        },
    };

    fn trace_test_lock() -> &'static Mutex<()> {
        crate::trace::test_trace_lock()
    }

    fn checkpoint_commitments(payload: &serde_json::Value, checkpoint: &str) -> Vec<String> {
        payload
            .as_array()
            .expect("checkpoint payload should be an array")
            .iter()
            .filter_map(|entry| entry.get(checkpoint)?.as_str().map(ToString::to_string))
            .collect()
    }

    fn checkpoint_commitments_with_prefix(
        payload: &serde_json::Value,
        checkpoint_prefix: &str,
    ) -> Vec<String> {
        payload
            .as_array()
            .expect("checkpoint payload should be an array")
            .iter()
            .filter_map(|entry| {
                let object = entry.as_object()?;
                object.iter().find_map(|(checkpoint, commitment)| {
                    checkpoint
                        .starts_with(checkpoint_prefix)
                        .then(|| commitment.as_str().map(ToString::to_string))
                        .flatten()
                })
            })
            .collect()
    }

    fn checkpoint_name_count(payload: &serde_json::Value, checkpoint: &str) -> usize {
        payload
            .as_array()
            .expect("checkpoint payload should be an array")
            .iter()
            .filter(|entry| entry.get(checkpoint).is_some())
            .count()
    }

    fn checkpoint_entry_name_and_commitment(entry: &serde_json::Value) -> (&str, &str) {
        let object = entry
            .as_object()
            .expect("checkpoint entry should be an object");
        let (checkpoint, commitment) = object
            .iter()
            .next()
            .expect("checkpoint entry should contain a commitment");
        (
            checkpoint.as_str(),
            commitment
                .as_str()
                .expect("checkpoint commitment should be a string"),
        )
    }

    fn assert_checkpoint_payloads_match_except_detour(
        native_payload: &serde_json::Value,
        detour_payload: &serde_json::Value,
        detour_spec: &str,
    ) {
        let detour_spec = RasterDetourSpec::parse(detour_spec).expect("detour spec should parse");
        let native_entries = native_payload
            .as_array()
            .expect("native checkpoint payload should be an array");
        let detour_entries = detour_payload
            .as_array()
            .expect("detour checkpoint payload should be an array");
        assert_eq!(
            native_entries.len(),
            detour_entries.len(),
            "checkpoint payloads should have the same shape"
        );

        let mut selected_seen = 0;
        for (idx, (native_entry, detour_entry)) in
            native_entries.iter().zip(detour_entries).enumerate()
        {
            let (native_checkpoint, native_commitment) =
                checkpoint_entry_name_and_commitment(native_entry);
            let (detour_checkpoint, detour_commitment) =
                checkpoint_entry_name_and_commitment(detour_entry);
            assert_eq!(
                native_checkpoint, detour_checkpoint,
                "checkpoint name mismatch at entry {idx}"
            );

            if native_checkpoint == detour_spec.routine_id().as_str() {
                selected_seen += 1;
                if selected_seen == detour_spec.occurrence() {
                    continue;
                }
            }

            assert_eq!(
                native_commitment, detour_commitment,
                "non-detoured checkpoint {native_checkpoint} differed at entry {idx}"
            );
        }

        assert!(
            selected_seen >= detour_spec.occurrence(),
            "selected checkpoint {} was not present",
            detour_spec
        );
    }

    struct TraceDirGuard {
        previous: Option<std::ffi::OsString>,
    }

    impl TraceDirGuard {
        fn new(test_name: &str) -> Self {
            let previous = env::var_os("RASTER_TRACE_DIR");
            let trace_dir =
                env::temp_dir().join(format!("raster-inference-{test_name}-{}", process::id()));
            fs::create_dir_all(&trace_dir).expect("trace dir should be created");
            env::set_var("RASTER_TRACE_DIR", trace_dir);
            Self { previous }
        }
    }

    impl Drop for TraceDirGuard {
        fn drop(&mut self) {
            if let Some(previous) = self.previous.take() {
                env::set_var("RASTER_TRACE_DIR", previous);
            } else {
                env::remove_var("RASTER_TRACE_DIR");
            }
        }
    }

    fn expect_completed_state(
        outcome: InferenceRunOutcome,
        description: &str,
    ) -> super::InferenceState {
        let InferenceRunOutcome::Completed(state) = outcome else {
            panic!("expected {description} to complete");
        };
        state
    }

    fn assert_output_decode_matches(
        native: &super::InferenceState,
        detour: &super::InferenceState,
    ) {
        assert_eq!(
            native.output_decode.generated_token_ids,
            detour.output_decode.generated_token_ids
        );
        assert_eq!(
            native.output_decode.generated_token_ids_sha256,
            detour.output_decode.generated_token_ids_sha256
        );
        assert_eq!(
            native.output_decode.generated_text,
            detour.output_decode.generated_text
        );
        assert_eq!(
            crate::trace::sha256_hex(&native.output_decode.generated_text),
            crate::trace::sha256_hex(&detour.output_decode.generated_text)
        );
        assert_eq!(
            native.output_decode.generated_token_count,
            detour.output_decode.generated_token_count
        );
    }

    fn deterministic_prompt_request(max_new_tokens: usize) -> InferenceRequest {
        InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: InferenceExecutionMode::Deterministic,
            sampling: SamplingConfig {
                max_new_tokens: Some(max_new_tokens),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
        }
    }

    #[test]
    fn run_inference_generates_greedy_text_for_max_new_tokens() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_model = test_transformer_model();
        let request = InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: InferenceExecutionMode::Fp32,
            sampling: SamplingConfig {
                max_new_tokens: Some(2),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
        };

        let inference_state = run_inference(&request, &model, &tokenizer, &transformer_model)
            .expect("inference should succeed");

        assert_eq!(
            inference_state
                .input_embedding
                .prompt_preparation
                .prompt_token_ids,
            vec![1]
        );
        assert_eq!(
            inference_state.output_decode.generated_token_ids,
            vec![0, 0]
        );
        assert_eq!(inference_state.output_decode.generated_text, "hello hello");
        assert_eq!(inference_state.output_decode.generated_token_count, 2);
        assert_eq!(
            inference_state.output_decode.stop_reason,
            OutputDecodeStopReason::MaxNewTokens
        );
    }

    #[test]
    fn run_inference_returns_empty_generation_when_max_new_tokens_is_zero() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_model = test_transformer_model();
        let request = InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: InferenceExecutionMode::Fp32,
            sampling: SamplingConfig {
                max_new_tokens: Some(0),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
        };

        let inference_state = run_inference(&request, &model, &tokenizer, &transformer_model)
            .expect("inference should succeed");

        assert!(inference_state.output_decode.generated_token_ids.is_empty());
        assert_eq!(inference_state.output_decode.generated_text, "");
        assert_eq!(inference_state.output_decode.generated_token_count, 0);
    }

    #[test]
    fn run_inference_rejects_non_default_sampling_before_decode_loop() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_model = test_transformer_model();
        let request = InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: InferenceExecutionMode::Fp32,
            sampling: SamplingConfig {
                max_new_tokens: Some(1),
                temperature: Some(1.0),
                top_k: Some(5),
                top_p: None,
            },
        };

        let error = run_inference(&request, &model, &tokenizer, &transformer_model)
            .expect_err("top_k should fail");
        assert!(error.to_string().contains("top_k"));
    }

    #[test]
    fn run_inference_rejects_raster_detour_without_deterministic_execution() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_model = test_transformer_model();
        let request = InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: InferenceExecutionMode::Fp32,
            sampling: SamplingConfig {
                max_new_tokens: Some(1),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
        };

        let error = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_model,
            &InferenceControls {
                raster_detour: Some(
                    RasterDetourSpec::parse("input.embedding").expect("detour should parse"),
                ),
                ..InferenceControls::default()
            },
        )
        .expect_err("detour should require deterministic execution");

        assert!(error
            .to_string()
            .contains("selective raster detour requires deterministic execution"));
    }

    #[test]
    fn run_inference_reports_unsupported_selected_raster_detour() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: InferenceExecutionMode::Deterministic,
            sampling: SamplingConfig {
                max_new_tokens: Some(1),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
        };

        let error = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                raster_detour: Some(
                    RasterDetourSpec::parse("prompt.prepare").expect("detour should parse"),
                ),
                raster_projection_rows_per_tile: Some(4),
                ..InferenceControls::default()
            },
        )
        .expect_err("unimplemented detour should fail");

        let message = error.to_string();
        assert!(
            message.contains("selective raster detour for prompt.prepare is not implemented yet"),
            "{message}"
        );
    }

    #[test]
    fn run_inference_reports_unmatched_selected_raster_detour() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: InferenceExecutionMode::Deterministic,
            sampling: SamplingConfig {
                max_new_tokens: Some(0),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
        };

        let error = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                raster_detour: Some(
                    RasterDetourSpec::parse("prefill.range:999").expect("detour should parse"),
                ),
                ..InferenceControls::default()
            },
        )
        .expect_err("unmatched detour should fail");

        assert!(error
            .to_string()
            .contains("selective raster detour target prefill.range:999 was not reached"));
    }

    #[test]
    fn run_inference_reports_unmatched_decode_select_token_detour() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = deterministic_prompt_request(1);

        let error = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                raster_detour: Some(
                    RasterDetourSpec::parse("decode.select_token:2").expect("detour should parse"),
                ),
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                ..InferenceControls::default()
            },
        )
        .expect_err("second decode select token detour should be unmatched");

        assert!(error
            .to_string()
            .contains("selective raster detour target decode.select_token:2 was not reached"));
    }

    #[test]
    fn run_inference_validates_sequence_sizing_for_decode_select_token_detour() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = deterministic_prompt_request(1);

        let error = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                raster_detour: Some(
                    RasterDetourSpec::parse("decode.select_token").expect("detour should parse"),
                ),
                raster_sequence_rows_per_tile: Some(0),
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                ..InferenceControls::default()
            },
        )
        .expect_err("zero sequence rows should fail");

        assert!(error
            .to_string()
            .contains("raster sequence rows per tile must be greater than zero"));
    }

    #[test]
    fn run_inference_counts_prefill_layer_detour_occurrences() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let mut transformer_fixture = deterministic_no_ple_model_fixture();
        transformer_fixture
            .model
            .layers
            .push(transformer_fixture.model.layers[0].clone());
        let request = InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: InferenceExecutionMode::Deterministic,
            sampling: SamplingConfig {
                max_new_tokens: Some(0),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
        };

        crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
        let native = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls::default(),
        )
        .expect("native inference should complete");

        crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
        let detour = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                raster_detour: Some(
                    RasterDetourSpec::parse("prefill.range:2").expect("detour should parse"),
                ),
                ..InferenceControls::default()
            },
        )
        .expect("second prefill layer detour should complete");

        let native = expect_completed_state(native, "native inference");
        let detour = expect_completed_state(detour, "second prefill layer detour inference");
        assert_output_decode_matches(&native, &detour);
        assert!(
            detour.raster_tile_invocations.unwrap_or(0) > 0,
            "prefill layer detour should count raster tiles"
        );
    }

    #[test]
    fn run_inference_rejects_full_raster_with_raster_detour() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: InferenceExecutionMode::Deterministic,
            sampling: SamplingConfig {
                max_new_tokens: Some(0),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
        };

        let error = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                raster: true,
                raster_detour: Some(
                    RasterDetourSpec::parse("prompt.prepare").expect("detour should parse"),
                ),
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                ..InferenceControls::default()
            },
        )
        .expect_err("full raster and detour should conflict");

        assert!(error
            .to_string()
            .contains("--raster and selective raster detour cannot be used together"));
    }

    #[test]
    fn run_inference_validates_raster_sizing_for_detour() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: InferenceExecutionMode::Deterministic,
            sampling: SamplingConfig {
                max_new_tokens: Some(0),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
        };

        let error = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                raster_detour: Some(
                    RasterDetourSpec::parse("prompt.prepare").expect("detour should parse"),
                ),
                raster_projection_rows_per_tile: Some(0),
                ..InferenceControls::default()
            },
        )
        .expect_err("zero raster sizing should fail for detour");

        assert!(error
            .to_string()
            .contains("raster projection rows per tile must be greater than zero"));
    }

    #[test]
    fn run_inference_validates_attention_sizing_for_prefill_layer_detour() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = deterministic_prompt_request(0);

        let error = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                raster_detour: Some(
                    RasterDetourSpec::parse("prefill.range").expect("detour should parse"),
                ),
                raster_attention_kv_rows_per_tile: Some(0),
                ..InferenceControls::default()
            },
        )
        .expect_err("zero attention KV rows should fail for prefill layer detour");

        assert!(error
            .to_string()
            .contains("raster attention KV rows per tile must be greater than zero"));
    }

    #[test]
    fn run_inference_executes_input_embedding_raster_detour() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = deterministic_prompt_request(1);

        crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
        let native = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls::default(),
        )
        .expect("native deterministic inference should complete");
        crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
        let detour = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                raster_detour: Some(
                    RasterDetourSpec::parse("input.embedding").expect("detour should parse"),
                ),
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                ..InferenceControls::default()
            },
        )
        .expect("input embedding detour should complete");

        let InferenceRunOutcome::Completed(native) = native else {
            panic!("expected native inference to complete");
        };
        let InferenceRunOutcome::Completed(detour) = detour else {
            panic!("expected detour inference to complete");
        };
        assert_eq!(
            native.input_embedding.prompt_preparation.prompt_token_ids,
            detour.input_embedding.prompt_preparation.prompt_token_ids
        );
        assert_eq!(
            native.input_embedding.embedded_prompt_activations_sha256,
            detour.input_embedding.embedded_prompt_activations_sha256
        );
        assert_eq!(
            native
                .input_embedding
                .det_embedded_prompt_activations_sha256,
            detour
                .input_embedding
                .det_embedded_prompt_activations_sha256
        );
        assert_eq!(
            native.output_decode.generated_token_ids,
            detour.output_decode.generated_token_ids
        );
        assert_eq!(
            native.output_decode.generated_token_ids_sha256,
            detour.output_decode.generated_token_ids_sha256
        );
        assert_eq!(
            native.output_decode.generated_text,
            detour.output_decode.generated_text
        );
        assert_eq!(
            native.output_decode.generated_token_count,
            detour.output_decode.generated_token_count
        );
        assert!(
            detour.raster_tile_invocations.unwrap_or(0) > 0,
            "input embedding detour should count raster tiles"
        );
    }

    #[test]
    fn run_inference_input_embedding_detour_can_pause_after_input_embedding() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = deterministic_prompt_request(1);

        crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
        let paused = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                terminal_checkpoint: Some("input.embedding".to_string()),
                raster_detour: Some(
                    RasterDetourSpec::parse("input.embedding").expect("detour should parse"),
                ),
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                ..InferenceControls::default()
            },
        )
        .expect("input embedding detour should pause");

        match paused {
            InferenceRunOutcome::Paused(state) => {
                assert_eq!(state.terminal_checkpoint_id, "input.embedding");
                assert_eq!(
                    state.input_embedding.prompt_preparation.prompt_token_ids,
                    vec![1]
                );
                assert!(state
                    .input_embedding
                    .det_embedded_prompt_activations_sha256
                    .is_some());
                assert!(state.transformer_state_transition.is_none());
                assert!(state.output_decode.is_none());
                assert!(
                    state.raster_tile_invocations.unwrap_or(0) > 0,
                    "input embedding detour should count raster tiles"
                );
            }
            InferenceRunOutcome::Completed(_) | InferenceRunOutcome::RasterPromptPrepared(_) => {
                panic!("expected paused detour inference")
            }
        }
    }

    #[test]
    fn run_inference_prefill_layer_detour_can_pause_after_selected_layer() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let mut transformer_fixture = deterministic_no_ple_model_fixture();
        transformer_fixture
            .model
            .layers
            .push(transformer_fixture.model.layers[0].clone());
        let request = deterministic_prompt_request(1);

        crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
        let paused = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                terminal_checkpoint: Some("prefill.range_finalize:2".to_string()),
                raster_detour: Some(
                    RasterDetourSpec::parse("prefill.range:2").expect("detour should parse"),
                ),
                raster_projection_rows_per_tile: Some(2),
                ..InferenceControls::default()
            },
        )
        .expect("prefill layer detour should pause after selected layer");

        match paused {
            InferenceRunOutcome::Paused(state) => {
                assert_eq!(state.terminal_checkpoint_id, "prefill.range_finalize");
                assert!(state.transformer_state_transition.is_none());
                assert!(state.output_decode.is_none());
                assert!(
                    state.raster_tile_invocations.unwrap_or(0) > 0,
                    "prefill layer detour should count raster tiles"
                );
            }
            InferenceRunOutcome::Completed(_) | InferenceRunOutcome::RasterPromptPrepared(_) => {
                panic!("expected paused prefill layer detour inference")
            }
        }
    }

    #[test]
    fn run_inference_executes_prefill_finalize_raster_detour() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = deterministic_prompt_request(1);

        crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
        let native = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls::default(),
        )
        .expect("native deterministic inference should complete");

        crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
        let detour = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                raster_detour: Some(
                    RasterDetourSpec::parse("prefill.finalize").expect("detour should parse"),
                ),
                raster_projection_rows_per_tile: Some(2),
                ..InferenceControls::default()
            },
        )
        .expect("prefill finalize detour should complete");

        let native = expect_completed_state(native, "native inference");
        let detour = expect_completed_state(detour, "prefill finalize detour inference");
        assert_output_decode_matches(&native, &detour);
        assert_eq!(
            native.transformer_state_transition.prefill_logits,
            detour.transformer_state_transition.prefill_logits
        );
        assert!(
            detour.raster_tile_invocations.unwrap_or(0) > 0,
            "prefill finalize detour should count raster tiles"
        );
    }

    #[test]
    fn run_inference_prefill_finalize_detour_can_pause_after_finalize() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = deterministic_prompt_request(1);

        crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
        let paused = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                terminal_checkpoint: Some("prefill.finalize".to_string()),
                raster_detour: Some(
                    RasterDetourSpec::parse("prefill.finalize").expect("detour should parse"),
                ),
                raster_projection_rows_per_tile: Some(2),
                ..InferenceControls::default()
            },
        )
        .expect("prefill finalize detour should pause after finalize");

        match paused {
            InferenceRunOutcome::Paused(state) => {
                assert_eq!(state.terminal_checkpoint_id, "prefill.finalize");
                assert!(state.transformer_state_transition.is_some());
                assert!(state.output_decode.is_none());
                assert!(
                    state.raster_tile_invocations.unwrap_or(0) > 0,
                    "prefill finalize detour should count raster tiles"
                );
            }
            InferenceRunOutcome::Completed(_) | InferenceRunOutcome::RasterPromptPrepared(_) => {
                panic!("expected paused prefill finalize detour inference")
            }
        }
    }

    #[test]
    fn run_inference_reports_unmatched_second_prefill_finalize_detour() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = deterministic_prompt_request(0);

        let error = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                raster_detour: Some(
                    RasterDetourSpec::parse("prefill.finalize:2").expect("detour should parse"),
                ),
                ..InferenceControls::default()
            },
        )
        .expect_err("second prefill finalize detour should be unmatched");

        assert!(error
            .to_string()
            .contains("selective raster detour target prefill.finalize:2 was not reached"));
    }

    #[test]
    fn run_inference_prefill_finalize_detour_uses_projection_tile_sizing() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = deterministic_prompt_request(0);

        crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
        let single_row_chunks = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                raster_detour: Some(
                    RasterDetourSpec::parse("prefill.finalize").expect("detour should parse"),
                ),
                raster_projection_rows_per_tile: Some(1),
                ..InferenceControls::default()
            },
        )
        .expect("single-row projection chunks should complete");

        crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
        let multi_row_chunks = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                raster_detour: Some(
                    RasterDetourSpec::parse("prefill.finalize").expect("detour should parse"),
                ),
                raster_projection_rows_per_tile: Some(2),
                ..InferenceControls::default()
            },
        )
        .expect("multi-row projection chunks should complete");

        let single_row_chunks = expect_completed_state(
            single_row_chunks,
            "single-row prefill finalize detour inference",
        );
        let multi_row_chunks = expect_completed_state(
            multi_row_chunks,
            "multi-row prefill finalize detour inference",
        );
        assert_output_decode_matches(&single_row_chunks, &multi_row_chunks);
        assert!(
            single_row_chunks.raster_tile_invocations.unwrap_or(0)
                > multi_row_chunks.raster_tile_invocations.unwrap_or(0),
            "smaller projection chunks should invoke more raster tiles"
        );
    }

    #[cfg(feature = "unchecked-raster-integrity")]
    #[test]
    fn run_inference_prefill_finalize_detour_runs_in_unchecked_integrity_mode() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = deterministic_prompt_request(1);

        crate::shared::artifacts::integrity_mode::with_raster_integrity_mode(
            crate::RasterIntegrityMode::UncheckedTestOnly,
            || {
                crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
                let detour = run_inference_with_controls(
                    &request,
                    &model,
                    &tokenizer,
                    &transformer_fixture.model,
                    &InferenceControls {
                        raster_detour: Some(
                            RasterDetourSpec::parse("prefill.finalize")
                                .expect("detour should parse"),
                        ),
                        raster_projection_rows_per_tile: Some(2),
                        ..InferenceControls::default()
                    },
                )
                .expect("unchecked prefill finalize detour should complete");

                let detour = expect_completed_state(detour, "unchecked prefill finalize detour");
                assert!(
                    detour.raster_tile_invocations.unwrap_or(0) > 0,
                    "unchecked prefill finalize detour should count raster tiles"
                );
            },
        );
    }

    #[test]
    fn run_inference_executes_output_finalize_raster_detour() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = deterministic_prompt_request(1);

        crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
        let native = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls::default(),
        )
        .expect("native deterministic inference should complete");

        crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
        let detour = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                raster_detour: Some(
                    RasterDetourSpec::parse("output.finalize").expect("detour should parse"),
                ),
                raster_tokenizer_source: Some(test_native_matching_gemma_tokenizer_source()),
                ..InferenceControls::default()
            },
        )
        .expect("output finalize detour should complete");

        let native = expect_completed_state(native, "native inference");
        let detour = expect_completed_state(detour, "output finalize detour inference");
        assert_output_decode_matches(&native, &detour);
        assert!(
            detour.raster_tile_invocations.unwrap_or(0) > 0,
            "output finalize detour should count raster tiles"
        );
    }

    #[test]
    fn run_inference_output_finalize_detour_requires_authenticated_tokenizer_source() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = deterministic_prompt_request(1);

        let error = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                raster_detour: Some(
                    RasterDetourSpec::parse("output.finalize").expect("detour should parse"),
                ),
                ..InferenceControls::default()
            },
        )
        .expect_err("output finalize detour requires tokenizer source");

        assert!(error.to_string().contains(
            "selective raster output.finalize detour requires an authenticated Gemma tokenizer"
        ));
    }

    #[test]
    fn run_inference_reports_unmatched_second_output_finalize_detour() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = deterministic_prompt_request(0);

        let error = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                raster_detour: Some(
                    RasterDetourSpec::parse("output.finalize:2").expect("detour should parse"),
                ),
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                ..InferenceControls::default()
            },
        )
        .expect_err("second output finalize detour should be unmatched");

        assert!(error
            .to_string()
            .contains("selective raster detour target output.finalize:2 was not reached"));
    }

    #[test]
    fn run_inference_validates_output_byte_flush_sizing_for_output_finalize_detour() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = deterministic_prompt_request(1);

        let error = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                raster_detour: Some(
                    RasterDetourSpec::parse("output.finalize").expect("detour should parse"),
                ),
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_output_byte_flush_bytes_per_tile: Some(0),
                ..InferenceControls::default()
            },
        )
        .expect_err("zero output byte flush bytes should fail for output finalize detour");

        assert!(error
            .to_string()
            .contains("raster output byte flush bytes per tile must be greater than zero"));
    }

    #[cfg(feature = "unchecked-raster-integrity")]
    #[test]
    fn run_inference_output_finalize_detour_runs_in_unchecked_integrity_mode() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = deterministic_prompt_request(1);

        crate::shared::artifacts::integrity_mode::with_raster_integrity_mode(
            crate::RasterIntegrityMode::UncheckedTestOnly,
            || {
                crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
                let detour =
                    run_inference_with_controls(
                        &request,
                        &model,
                        &tokenizer,
                        &transformer_fixture.model,
                        &InferenceControls {
                            raster_detour: Some(
                                RasterDetourSpec::parse("output.finalize")
                                    .expect("detour should parse"),
                            ),
                            raster_tokenizer_source: Some(
                                test_native_matching_gemma_tokenizer_source(),
                            ),
                            ..InferenceControls::default()
                        },
                    )
                    .expect("unchecked output finalize detour should complete");

                let detour = expect_completed_state(detour, "unchecked output finalize detour");
                assert!(
                    detour.raster_tile_invocations.unwrap_or(0) > 0,
                    "unchecked output finalize detour should count raster tiles"
                );
            },
        );
    }

    #[test]
    fn run_inference_input_embedding_detour_requires_prompt_artifact_roots() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = deterministic_prompt_request(1);

        let error = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                raster_detour: Some(
                    RasterDetourSpec::parse("input.embedding").expect("detour should parse"),
                ),
                ..InferenceControls::default()
            },
        )
        .expect_err("input embedding detour requires prompt artifact roots");

        assert!(error.to_string().contains(
            "selective raster input.embedding detour requires an authenticated Gemma tokenizer"
        ));
    }

    #[test]
    fn run_inference_reports_unmatched_second_input_embedding_detour() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = deterministic_prompt_request(0);

        let error = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                raster_detour: Some(
                    RasterDetourSpec::parse("input.embedding:2").expect("detour should parse"),
                ),
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                ..InferenceControls::default()
            },
        )
        .expect_err("second input embedding detour should be unmatched");

        assert!(error
            .to_string()
            .contains("selective raster detour target input.embedding:2 was not reached"));
    }

    #[test]
    fn run_inference_executes_prefill_prepare_aux_raster_detour() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_ple_model_fixture();
        let tokenizer_source = test_gemma_tokenizer_source();
        let request = deterministic_prompt_request(1);

        crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
        let native = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls::default(),
        )
        .expect("native deterministic inference should complete");

        crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
        let detour = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                raster_detour: Some(
                    RasterDetourSpec::parse("prefill.prepare_aux").expect("detour should parse"),
                ),
                raster_tokenizer_source: Some(tokenizer_source),
                raster_projection_rows_per_tile: Some(2),
                raster_sequence_rows_per_tile: Some(2),
                ..InferenceControls::default()
            },
        )
        .expect("prefill prepare aux detour should complete");

        let native = expect_completed_state(native, "native inference");
        let detour = expect_completed_state(detour, "prefill prepare aux detour inference");
        assert_output_decode_matches(&native, &detour);
        assert!(
            detour.raster_tile_invocations.unwrap_or(0) > 0,
            "prefill prepare aux detour should count raster tiles"
        );
    }

    #[test]
    fn run_inference_prefill_prepare_aux_detour_requires_input_embedding_refs() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = deterministic_prompt_request(1);

        let error = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                raster_detour: Some(
                    RasterDetourSpec::parse("prefill.prepare_aux").expect("detour should parse"),
                ),
                ..InferenceControls::default()
            },
        )
        .expect_err("prefill prepare aux detour requires input embedding refs");

        assert!(error.to_string().contains(
            "selective raster prefill.prepare_aux detour requires input embedding raster refs"
        ));
    }

    #[test]
    fn run_inference_reports_unmatched_second_prefill_prepare_aux_detour() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = deterministic_prompt_request(0);

        let error = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                raster_detour: Some(
                    RasterDetourSpec::parse("prefill.prepare_aux:2").expect("detour should parse"),
                ),
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                ..InferenceControls::default()
            },
        )
        .expect_err("second prefill prepare aux detour should be unmatched");

        assert!(error
            .to_string()
            .contains("selective raster detour target prefill.prepare_aux:2 was not reached"));
    }

    #[test]
    fn run_inference_validates_sequence_sizing_for_prefill_prepare_aux_detour() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = deterministic_prompt_request(0);

        let error = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                raster_detour: Some(
                    RasterDetourSpec::parse("prefill.prepare_aux").expect("detour should parse"),
                ),
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_sequence_rows_per_tile: Some(0),
                ..InferenceControls::default()
            },
        )
        .expect_err("zero sequence rows should fail for prepare aux detour");

        assert!(error
            .to_string()
            .contains("raster sequence rows per tile must be greater than zero"));
    }

    #[test]
    fn deterministic_cpu_trace_matches_input_embedding_detour_trace() {
        let _trace_guard = trace_test_lock().lock().expect("trace test lock");
        let _trace_dir = TraceDirGuard::new("input-embedding-detour-trace");
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let tokenizer_source = test_gemma_tokenizer_source();
        let request = deterministic_prompt_request(1);

        crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
        let native = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                commit_checkpoints: true,
                raster_tokenizer_source: Some(tokenizer_source.clone()),
                ..InferenceControls::default()
            },
        )
        .expect("native deterministic inference should complete");
        let native_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

        crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
        let detour = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                commit_checkpoints: true,
                raster_detour: Some(
                    RasterDetourSpec::parse("input.embedding").expect("detour should parse"),
                ),
                raster_tokenizer_source: Some(tokenizer_source),
                raster_projection_rows_per_tile: Some(2),
                raster_sequence_rows_per_tile: Some(2),
                ..InferenceControls::default()
            },
        )
        .expect("input embedding detour inference should complete");
        let detour_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

        assert_checkpoint_payloads_match_except_detour(
            &native_payload,
            &detour_payload,
            "input.embedding",
        );
        assert_eq!(
            checkpoint_commitments(&native_payload, "input.embedding").len(),
            1
        );
        assert_eq!(
            checkpoint_commitments(&detour_payload, "input.embedding").len(),
            1
        );

        let native = expect_completed_state(native, "native inference");
        let detour = expect_completed_state(detour, "prefill prepare aux detour inference");
        assert_output_decode_matches(&native, &detour);
        assert!(
            detour.raster_tile_invocations.unwrap_or(0) > 0,
            "detour should expose raster tile telemetry outside committed checkpoints"
        );
    }

    #[test]
    fn deterministic_cpu_trace_matches_prefill_prepare_aux_detour_trace() {
        let _trace_guard = trace_test_lock().lock().expect("trace test lock");
        let _trace_dir = TraceDirGuard::new("prefill-prepare-aux-detour-trace");
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_ple_model_fixture();
        let tokenizer_source = test_gemma_tokenizer_source();
        let request = deterministic_prompt_request(1);

        crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
        let native = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                commit_checkpoints: true,
                raster_tokenizer_source: Some(tokenizer_source.clone()),
                ..InferenceControls::default()
            },
        )
        .expect("native deterministic inference should complete");
        let native_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

        crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
        let detour = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                commit_checkpoints: true,
                raster_detour: Some(
                    RasterDetourSpec::parse("prefill.prepare_aux").expect("detour should parse"),
                ),
                raster_tokenizer_source: Some(tokenizer_source),
                raster_projection_rows_per_tile: Some(2),
                raster_attention_kv_rows_per_tile: Some(2),
                raster_sequence_rows_per_tile: Some(2),
                raster_head_rows_per_tile: Some(2),
                raster_tokenizer_bpe_pairs_per_tile: Some(2),
                raster_tokenizer_bpe_pieces_per_tile: Some(2),
                raster_output_byte_flush_bytes_per_tile: Some(2),
                ..InferenceControls::default()
            },
        )
        .expect("prefill prepare aux detour inference should complete");
        let detour_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

        assert_checkpoint_payloads_match_except_detour(
            &native_payload,
            &detour_payload,
            "prefill.prepare_aux",
        );
        assert_eq!(
            checkpoint_commitments(&native_payload, "prefill.prepare_aux").len(),
            1
        );
        assert_eq!(
            checkpoint_commitments(&detour_payload, "prefill.prepare_aux").len(),
            1
        );

        let InferenceRunOutcome::Completed(native) = native else {
            panic!("expected native inference to complete");
        };
        let InferenceRunOutcome::Completed(detour) = detour else {
            panic!("expected detour inference to complete");
        };
        assert_eq!(
            native.output_decode.generated_token_ids,
            detour.output_decode.generated_token_ids
        );
        assert_eq!(
            native.output_decode.generated_token_ids_sha256,
            detour.output_decode.generated_token_ids_sha256
        );
        assert_eq!(
            native.output_decode.generated_text,
            detour.output_decode.generated_text
        );
        assert_eq!(
            crate::trace::sha256_hex(&native.output_decode.generated_text),
            crate::trace::sha256_hex(&detour.output_decode.generated_text)
        );
        assert_eq!(
            native.output_decode.generated_token_count,
            detour.output_decode.generated_token_count
        );
        assert!(
            detour.raster_tile_invocations.unwrap_or(0) > 0,
            "detour should expose raster tile telemetry outside committed checkpoints"
        );
    }

    #[test]
    fn deterministic_cpu_trace_matches_no_ple_prefill_prepare_aux_detour_trace() {
        let _trace_guard = trace_test_lock().lock().expect("trace test lock");
        let _trace_dir = TraceDirGuard::new("no-ple-prefill-prepare-aux-detour-trace");
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let tokenizer_source = test_gemma_tokenizer_source();
        let request = deterministic_prompt_request(1);

        crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
        let native = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                commit_checkpoints: true,
                raster_tokenizer_source: Some(tokenizer_source.clone()),
                ..InferenceControls::default()
            },
        )
        .expect("native deterministic inference should complete");
        let native_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

        crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
        let detour = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                commit_checkpoints: true,
                raster_detour: Some(
                    RasterDetourSpec::parse("prefill.prepare_aux").expect("detour should parse"),
                ),
                raster_tokenizer_source: Some(tokenizer_source),
                ..InferenceControls::default()
            },
        )
        .expect("no-PLE prefill prepare aux detour should complete");
        let detour_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

        assert_checkpoint_payloads_match_except_detour(
            &native_payload,
            &detour_payload,
            "prefill.prepare_aux",
        );
        assert_eq!(
            checkpoint_commitments(&native_payload, "prefill.prepare_aux").len(),
            1
        );
        assert_eq!(
            checkpoint_commitments(&detour_payload, "prefill.prepare_aux").len(),
            1
        );

        let native = expect_completed_state(native, "native inference");
        let detour = expect_completed_state(detour, "no-PLE prefill prepare aux detour inference");
        assert_output_decode_matches(&native, &detour);
    }

    #[test]
    fn deterministic_cpu_trace_matches_prefill_layer_detour_trace() {
        let _trace_guard = trace_test_lock().lock().expect("trace test lock");
        let _trace_dir = TraceDirGuard::new("prefill-layer-detour-trace");
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let mut transformer_fixture = deterministic_no_ple_model_fixture();
        transformer_fixture
            .model
            .layers
            .push(transformer_fixture.model.layers[0].clone());
        let tokenizer_source = test_gemma_tokenizer_source();
        let request = deterministic_prompt_request(1);

        crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
        let native = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                commit_checkpoints: true,
                raster_tokenizer_source: Some(tokenizer_source.clone()),
                ..InferenceControls::default()
            },
        )
        .expect("native deterministic inference should complete");
        let native_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

        crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
        let detour = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                commit_checkpoints: true,
                raster_detour: Some(
                    RasterDetourSpec::parse("prefill.range:2").expect("detour should parse"),
                ),
                raster_tokenizer_source: Some(tokenizer_source),
                raster_projection_rows_per_tile: Some(2),
                raster_attention_kv_rows_per_tile: Some(2),
                raster_sequence_rows_per_tile: Some(2),
                raster_head_rows_per_tile: Some(2),
                ..InferenceControls::default()
            },
        )
        .expect("prefill layer detour inference should complete");
        let detour_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

        assert_checkpoint_payloads_match_except_detour(
            &native_payload,
            &detour_payload,
            "prefill.range_finalize:2",
        );
        assert_eq!(
            checkpoint_commitments(&native_payload, "prefill.range_finalize").len(),
            2
        );
        assert_eq!(
            checkpoint_commitments(&detour_payload, "prefill.range_finalize").len(),
            2
        );

        let native = expect_completed_state(native, "native inference");
        let detour = expect_completed_state(detour, "prefill layer detour inference");
        assert_output_decode_matches(&native, &detour);
        assert!(
            detour.raster_tile_invocations.unwrap_or(0) > 0,
            "detour should expose raster tile telemetry outside committed checkpoints"
        );
    }

    #[test]
    fn deterministic_cpu_trace_matches_ple_prefill_layer_detour_trace() {
        let _trace_guard = trace_test_lock().lock().expect("trace test lock");
        let _trace_dir = TraceDirGuard::new("ple-prefill-layer-detour-trace");
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_ple_model_fixture();
        let tokenizer_source = test_gemma_tokenizer_source();
        let request = deterministic_prompt_request(1);

        crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
        let native = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                commit_checkpoints: true,
                raster_tokenizer_source: Some(tokenizer_source.clone()),
                ..InferenceControls::default()
            },
        )
        .expect("native deterministic inference should complete");
        let native_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

        crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
        let detour = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                commit_checkpoints: true,
                raster_detour: Some(
                    RasterDetourSpec::parse("prefill.range").expect("detour should parse"),
                ),
                raster_tokenizer_source: Some(tokenizer_source),
                raster_projection_rows_per_tile: Some(2),
                raster_attention_kv_rows_per_tile: Some(2),
                raster_sequence_rows_per_tile: Some(2),
                raster_head_rows_per_tile: Some(2),
                ..InferenceControls::default()
            },
        )
        .expect("PLE prefill layer detour inference should complete");
        let detour_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

        assert_checkpoint_payloads_match_except_detour(
            &native_payload,
            &detour_payload,
            "prefill.range_finalize",
        );
        assert_eq!(
            checkpoint_commitments(&native_payload, "prefill.range_finalize").len(),
            1
        );
        assert_eq!(
            checkpoint_commitments(&detour_payload, "prefill.range_finalize").len(),
            1
        );

        let native = expect_completed_state(native, "native inference");
        let detour = expect_completed_state(detour, "PLE prefill layer detour inference");
        assert_output_decode_matches(&native, &detour);
        assert!(
            detour.raster_tile_invocations.unwrap_or(0) > 0,
            "detour should expose raster tile telemetry outside committed checkpoints"
        );
    }

    #[test]
    fn deterministic_cpu_trace_matches_prefill_finalize_detour_trace() {
        let _trace_guard = trace_test_lock().lock().expect("trace test lock");
        let _trace_dir = TraceDirGuard::new("prefill-finalize-detour-trace");
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let tokenizer_source = test_gemma_tokenizer_source();
        let request = deterministic_prompt_request(1);

        crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
        let native = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                commit_checkpoints: true,
                raster_tokenizer_source: Some(tokenizer_source.clone()),
                ..InferenceControls::default()
            },
        )
        .expect("native deterministic inference should complete");
        let native_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

        crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
        let detour = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                commit_checkpoints: true,
                raster_detour: Some(
                    RasterDetourSpec::parse("prefill.finalize").expect("detour should parse"),
                ),
                raster_tokenizer_source: Some(tokenizer_source),
                raster_projection_rows_per_tile: Some(2),
                ..InferenceControls::default()
            },
        )
        .expect("prefill finalize detour inference should complete");
        let detour_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

        assert_checkpoint_payloads_match_except_detour(
            &native_payload,
            &detour_payload,
            "prefill.finalize",
        );
        assert_eq!(
            checkpoint_commitments(&native_payload, "prefill.finalize").len(),
            1
        );
        assert_eq!(
            checkpoint_commitments(&detour_payload, "prefill.finalize").len(),
            1
        );

        let native = expect_completed_state(native, "native inference");
        let detour = expect_completed_state(detour, "prefill finalize detour inference");
        assert_output_decode_matches(&native, &detour);
        assert!(
            detour.raster_tile_invocations.unwrap_or(0) > 0,
            "detour should expose raster tile telemetry outside committed checkpoints"
        );
    }

    #[test]
    fn deterministic_cpu_trace_matches_output_finalize_detour_trace() {
        let _trace_guard = trace_test_lock().lock().expect("trace test lock");
        let _trace_dir = TraceDirGuard::new("output-finalize-detour-trace");
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let tokenizer_source = test_native_matching_gemma_tokenizer_source();
        let request = deterministic_prompt_request(1);

        crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
        let native = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                commit_checkpoints: true,
                raster_tokenizer_source: Some(tokenizer_source.clone()),
                ..InferenceControls::default()
            },
        )
        .expect("native deterministic inference should complete");
        let native_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

        crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
        let detour = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                commit_checkpoints: true,
                raster_detour: Some(
                    RasterDetourSpec::parse("output.finalize").expect("detour should parse"),
                ),
                raster_tokenizer_source: Some(tokenizer_source),
                raster_output_byte_flush_bytes_per_tile: Some(1),
                ..InferenceControls::default()
            },
        )
        .expect("output finalize detour inference should complete");
        let detour_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

        assert_checkpoint_payloads_match_except_detour(
            &native_payload,
            &detour_payload,
            "output.finalize",
        );
        assert_eq!(
            checkpoint_commitments(&native_payload, "output.finalize").len(),
            1
        );
        assert_eq!(
            checkpoint_commitments(&detour_payload, "output.finalize").len(),
            1
        );

        let native = expect_completed_state(native, "native inference");
        let detour = expect_completed_state(detour, "output finalize detour inference");
        assert_output_decode_matches(&native, &detour);
        assert!(
            detour.raster_tile_invocations.unwrap_or(0) > 0,
            "detour should expose raster tile telemetry outside committed checkpoints"
        );
    }

    fn assert_decode_select_detour_matches_native(
        test_name: &str,
        max_new_tokens: usize,
        detour_spec: &str,
        expected_decode_select_checkpoints: usize,
    ) {
        let _trace_guard = trace_test_lock().lock().expect("trace test lock");
        let _trace_dir = TraceDirGuard::new(test_name);
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let tokenizer_source = test_gemma_tokenizer_source();
        let request = deterministic_prompt_request(max_new_tokens);

        crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
        let native = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                commit_checkpoints: true,
                raster_tokenizer_source: Some(tokenizer_source.clone()),
                ..InferenceControls::default()
            },
        )
        .expect("native deterministic inference should complete");
        let native_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

        crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
        let detour = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                commit_checkpoints: true,
                raster_detour: Some(
                    RasterDetourSpec::parse(detour_spec).expect("detour should parse"),
                ),
                raster_tokenizer_source: Some(tokenizer_source),
                raster_sequence_rows_per_tile: Some(2),
                ..InferenceControls::default()
            },
        )
        .expect("decode select token detour inference should complete");
        let detour_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

        assert_checkpoint_payloads_match_except_detour(
            &native_payload,
            &detour_payload,
            detour_spec,
        );
        assert_eq!(
            checkpoint_commitments(&native_payload, "decode.select_token").len(),
            expected_decode_select_checkpoints
        );

        let native = expect_completed_state(native, "native inference");
        let detour = expect_completed_state(detour, "decode select token detour inference");
        assert_output_decode_matches(&native, &detour);
        assert!(
            detour.raster_tile_invocations.unwrap_or(0) > 0,
            "detour should expose raster tile telemetry outside committed checkpoints"
        );
    }

    #[test]
    fn deterministic_cpu_trace_matches_decode_select_token_detour_trace() {
        assert_decode_select_detour_matches_native(
            "decode-select-token-detour-trace",
            2,
            "decode.select_token",
            2,
        );
    }

    #[test]
    fn deterministic_cpu_trace_matches_second_decode_select_token_detour_trace() {
        assert_decode_select_detour_matches_native(
            "second-decode-select-token-detour-trace",
            2,
            "decode.select_token:2",
            2,
        );
    }

    fn assert_decode_transition_detour_matches_native(
        test_name: &str,
        max_new_tokens: usize,
        detour_spec: &str,
        expected_decode_transitions: usize,
    ) {
        let _trace_guard = trace_test_lock().lock().expect("trace test lock");
        let _trace_dir = TraceDirGuard::new(test_name);
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = deterministic_prompt_request(max_new_tokens);

        crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
        let native = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                commit_checkpoints: true,
                ..InferenceControls::default()
            },
        )
        .expect("native deterministic inference should complete");
        let native_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

        crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
        let detour = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                commit_checkpoints: true,
                raster_detour: Some(
                    RasterDetourSpec::parse(detour_spec).expect("detour should parse"),
                ),
                raster_projection_rows_per_tile: Some(2),
                raster_attention_kv_rows_per_tile: Some(2),
                ..InferenceControls::default()
            },
        )
        .expect("decode transition detour inference should complete");
        let detour_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

        assert_checkpoint_payloads_match_except_detour(
            &native_payload,
            &detour_payload,
            detour_spec,
        );
        assert_eq!(
            checkpoint_commitments_with_prefix(&native_payload, "decode.layer_token.").len(),
            0
        );
        assert_eq!(
            checkpoint_commitments_with_prefix(&detour_payload, "decode.layer_token.").len(),
            0
        );
        assert_eq!(
            checkpoint_name_count(&native_payload, "decode.layer_range"),
            expected_decode_transitions
        );
        assert_eq!(
            checkpoint_name_count(&detour_payload, "decode.layer_range"),
            expected_decode_transitions
        );
        assert_eq!(
            checkpoint_name_count(&native_payload, "decode.transition_finalize"),
            expected_decode_transitions
        );
        assert_eq!(
            checkpoint_name_count(&detour_payload, "decode.transition_finalize"),
            expected_decode_transitions
        );
        assert!(
            checkpoint_name_count(&native_payload, "decode.finalize") == 0
                && checkpoint_name_count(&detour_payload, "decode.finalize") == 0,
            "decode.transition_finalize is the decode finalization checkpoint; decode.finalize should not be committed"
        );

        let native = expect_completed_state(native, "native inference");
        let detour = expect_completed_state(detour, "decode transition detour inference");
        assert_output_decode_matches(&native, &detour);
        assert!(
            detour.raster_tile_invocations.unwrap_or(0) > 0,
            "detour should expose raster tile telemetry outside committed checkpoints"
        );
    }

    #[test]
    fn deterministic_cpu_trace_matches_decode_transition_detour_trace() {
        assert_decode_transition_detour_matches_native(
            "decode-transition-detour-trace",
            2,
            "decode.layer_range",
            2,
        );
    }

    #[test]
    fn deterministic_cpu_trace_matches_second_decode_transition_detour_trace() {
        assert_decode_transition_detour_matches_native(
            "second-decode-transition-detour-trace",
            2,
            "decode.layer_range:2",
            2,
        );
    }

    #[test]
    fn deterministic_cpu_decode_transition_non_detoured_checkpoints_match_raster_detour() {
        let _trace_guard = trace_test_lock().lock().expect("trace test lock");
        let _trace_dir = TraceDirGuard::new("decode-transition-checkpoint-commitment");
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = deterministic_prompt_request(1);

        crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
        run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                commit_checkpoints: true,
                ..InferenceControls::default()
            },
        )
        .expect("native deterministic inference should complete");
        let deterministic_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

        crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
        run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                commit_checkpoints: true,
                raster_detour: Some(
                    RasterDetourSpec::parse("decode.layer_range").expect("detour should parse"),
                ),
                raster_projection_rows_per_tile: Some(3),
                raster_attention_kv_rows_per_tile: Some(2),
                ..InferenceControls::default()
            },
        )
        .expect("decode transition raster detour inference should complete");
        let raster_payload = crate::trace::take_completed_checkpoint_payload_for_tests();

        assert_eq!(
            checkpoint_commitments_with_prefix(&deterministic_payload, "decode.layer_token.").len(),
            0
        );
        assert_checkpoint_payloads_match_except_detour(
            &deterministic_payload,
            &raster_payload,
            "decode.layer_range",
        );
    }

    #[test]
    fn deterministic_cpu_decode_layer_range_width_splits_checkpoints() {
        let _trace_guard = trace_test_lock().lock().expect("trace test lock");
        let _trace_dir = TraceDirGuard::new("decode-layer-range-width");
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let mut transformer_fixture = deterministic_no_ple_model_fixture();
        transformer_fixture
            .model
            .layers
            .push(transformer_fixture.model.layers[0].clone());
        let request = deterministic_prompt_request(1);

        crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
        run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                commit_checkpoints: true,
                decode_layer_range_width: Some(1),
                ..InferenceControls::default()
            },
        )
        .expect("native deterministic inference should complete");
        let payload = crate::trace::take_completed_checkpoint_payload_for_tests();

        assert_eq!(checkpoint_name_count(&payload, "decode.layer_range"), 2);
        assert_eq!(
            checkpoint_name_count(&payload, "decode.transition_finalize"),
            1
        );
    }

    #[test]
    fn run_inference_reports_unmatched_decode_transition_detour() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = deterministic_prompt_request(1);

        let error = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                raster_detour: Some(
                    RasterDetourSpec::parse("decode.layer_range:2").expect("detour should parse"),
                ),
                ..InferenceControls::default()
            },
        )
        .expect_err("second decode transition detour should be unmatched");

        assert!(error
            .to_string()
            .contains("selective raster detour target decode.layer_range:2 was not reached"));
    }

    #[test]
    fn run_inference_rejects_decode_transition_detour_without_deterministic_execution() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_model = test_transformer_model();
        let request = InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: InferenceExecutionMode::Fp32,
            sampling: SamplingConfig {
                max_new_tokens: Some(1),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
        };

        let error = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_model,
            &InferenceControls {
                raster_detour: Some(
                    RasterDetourSpec::parse("decode.layer_range").expect("detour should parse"),
                ),
                ..InferenceControls::default()
            },
        )
        .expect_err("decode transition detour should require deterministic execution");

        assert!(error
            .to_string()
            .contains("selective raster detour requires deterministic execution"));
    }

    #[test]
    fn run_inference_validates_projection_sizing_for_decode_transition_detour() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = deterministic_prompt_request(1);

        let error = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                raster_detour: Some(
                    RasterDetourSpec::parse("decode.layer_range").expect("detour should parse"),
                ),
                raster_projection_rows_per_tile: Some(0),
                ..InferenceControls::default()
            },
        )
        .expect_err("zero projection rows should fail for decode transition detour");

        assert!(error
            .to_string()
            .contains("raster projection rows per tile must be greater than zero"));
    }

    #[test]
    fn run_inference_validates_attention_sizing_for_decode_transition_detour() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = deterministic_prompt_request(1);

        let error = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                raster_detour: Some(
                    RasterDetourSpec::parse("decode.layer_range").expect("detour should parse"),
                ),
                raster_attention_kv_rows_per_tile: Some(0),
                ..InferenceControls::default()
            },
        )
        .expect_err("zero attention rows should fail for decode transition detour");

        assert!(error
            .to_string()
            .contains("raster attention KV rows per tile must be greater than zero"));
    }

    #[test]
    fn run_inference_decode_transition_finalize_detour_matches_native() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = deterministic_prompt_request(1);

        let native = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls::default(),
        )
        .expect("native inference should complete");

        let detour = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                raster_detour: Some(
                    RasterDetourSpec::parse("decode.transition_finalize")
                        .expect("detour should parse"),
                ),
                ..InferenceControls::default()
            },
        )
        .expect("decode transition finalize detour should complete");

        let native = expect_completed_state(native, "native inference");
        let detour = expect_completed_state(detour, "decode transition finalize detour");
        assert_output_decode_matches(&native, &detour);
        assert!(
            detour.raster_tile_invocations.unwrap_or(0) > 0,
            "finalize detour should execute raster tiles"
        );
    }

    #[cfg(feature = "unchecked-raster-integrity")]
    #[test]
    fn run_inference_decode_transition_detour_runs_in_unchecked_integrity_mode() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = deterministic_prompt_request(1);

        crate::shared::artifacts::integrity_mode::with_raster_integrity_mode(
            crate::RasterIntegrityMode::UncheckedTestOnly,
            || {
                crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
                let detour = run_inference_with_controls(
                    &request,
                    &model,
                    &tokenizer,
                    &transformer_fixture.model,
                    &InferenceControls {
                        raster_detour: Some(
                            RasterDetourSpec::parse("decode.layer_range")
                                .expect("detour should parse"),
                        ),
                        raster_projection_rows_per_tile: Some(1),
                        raster_attention_kv_rows_per_tile: Some(1),
                        ..InferenceControls::default()
                    },
                )
                .expect("unchecked decode transition detour should complete");

                let detour = expect_completed_state(detour, "unchecked decode transition detour");
                assert!(
                    detour.raster_tile_invocations.unwrap_or(0) > 0,
                    "unchecked decode transition detour should count raster tiles"
                );
            },
        );
    }

    #[cfg(feature = "unchecked-raster-integrity")]
    #[test]
    fn run_inference_decode_select_token_detour_runs_in_unchecked_integrity_mode() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = deterministic_prompt_request(1);

        crate::shared::artifacts::integrity_mode::with_raster_integrity_mode(
            crate::RasterIntegrityMode::UncheckedTestOnly,
            || {
                crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
                let detour = run_inference_with_controls(
                    &request,
                    &model,
                    &tokenizer,
                    &transformer_fixture.model,
                    &InferenceControls {
                        raster_detour: Some(
                            RasterDetourSpec::parse("decode.select_token")
                                .expect("detour should parse"),
                        ),
                        raster_sequence_rows_per_tile: Some(1),
                        raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                        ..InferenceControls::default()
                    },
                )
                .expect("unchecked decode select token detour should complete");

                let detour = expect_completed_state(detour, "unchecked decode select token detour");
                assert!(
                    detour.raster_tile_invocations.unwrap_or(0) > 0,
                    "unchecked decode select token detour should count raster tiles"
                );
            },
        );
    }

    #[test]
    fn run_inference_with_controls_pauses_after_prompt_prepare_checkpoint() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_model = test_transformer_model();
        let request = InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: InferenceExecutionMode::Fp32,
            sampling: SamplingConfig {
                max_new_tokens: Some(2),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
        };

        let paused = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_model,
            &InferenceControls {
                commit_checkpoints: false,
                terminal_checkpoint: Some("prompt.prepare".to_string()),
                raster: false,
                raster_detour: None,
                raster_tokenizer_source: None,
                raster_projection_rows_per_tile: None,
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
                prefill_token_range_width: None,
                decode_layer_range_width: None,
                raster_tokenizer_bpe_pairs_per_tile: None,
                raster_tokenizer_bpe_pieces_per_tile: None,
                raster_output_byte_flush_bytes_per_tile: None,
            },
        )
        .expect("inference should pause");

        match paused {
            InferenceRunOutcome::Paused(state) => {
                assert_eq!(state.terminal_checkpoint_id, "prompt.prepare");
                assert_eq!(
                    state.input_embedding.prompt_preparation.prompt_token_ids,
                    vec![1]
                );
                assert!(state.transformer_state_transition.is_none());
                assert!(state.output_decode.is_none());
            }
            InferenceRunOutcome::Completed(_) => panic!("expected paused inference"),
            InferenceRunOutcome::RasterPromptPrepared(_) => panic!("expected paused inference"),
        }
    }

    #[test]
    fn deterministic_cpu_prompt_prepare_checkpoint_matches_raster_shape() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let tokenizer_source = test_gemma_tokenizer_source();
        let request = InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: InferenceExecutionMode::Deterministic,
            sampling: SamplingConfig {
                max_new_tokens: Some(2),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
        };

        let paused = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                commit_checkpoints: false,
                terminal_checkpoint: Some("prompt.prepare".to_string()),
                raster: false,
                raster_detour: None,
                raster_tokenizer_source: Some(tokenizer_source.clone()),
                raster_projection_rows_per_tile: None,
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
                prefill_token_range_width: None,
                decode_layer_range_width: None,
                raster_tokenizer_bpe_pairs_per_tile: None,
                raster_tokenizer_bpe_pieces_per_tile: None,
                raster_output_byte_flush_bytes_per_tile: None,
            },
        )
        .expect("deterministic inference should stop after prompt prepare");
        let expected = crate::prompt_prepare::run_raster(
            &request,
            &model,
            &tokenizer_source,
            InferenceControls::default()
                .raster_sizing_controls()
                .expect("default sizing"),
        )
        .expect("raster prompt prepare should run");

        match paused {
            InferenceRunOutcome::RasterPromptPrepared(state) => {
                assert_eq!(state.terminal_checkpoint_id, "prompt.prepare");
                assert_eq!(state.prompt_preparation, expected.state);
                assert_eq!(state.sampling, request.sampling);
                assert_eq!(state.raster_tile_invocations, None);
            }
            InferenceRunOutcome::Completed(_) | InferenceRunOutcome::Paused(_) => {
                panic!("expected deterministic CPU prompt boundary")
            }
        }
    }

    #[test]
    fn deterministic_cpu_prefill_layer_checkpoint_commitment_matches_raster() {
        let _trace_guard = trace_test_lock().lock().expect("trace test lock");
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let input_rows = vec![
            vec![
                Act::from_num(1.0),
                Act::from_num(0.0),
                Act::from_num(0.0),
                Act::from_num(0.0),
            ],
            vec![
                Act::from_num(0.0),
                Act::from_num(1.0),
                Act::from_num(0.0),
                Act::from_num(0.0),
            ],
        ];

        let deterministic_payload = crate::trace::with_checkpointing_enabled(true, || {
            crate::trace::start_inference_trace(&json!({ "test": "deterministic-prefill-layer" }));
            crate::prefill_range::run_with_mode_internal_with_detour(
                InternalActivationSequence::from_det_values(input_rows.clone()),
                &transformer_fixture.model,
                None,
                InferenceExecutionMode::Deterministic,
                None,
                None,
                1,
            )
            .expect("deterministic prefill layer should run");
            crate::trace::checkpoint_payload_for_tests()
        });

        crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
        let input_ref = insert_activation_sequence_artifact_ref(
            "test.prefill.layer.input_embedding",
            RasterActivationSequence::from_acts(input_rows),
        )
        .expect("input embedding activation ref");
        let input_embedding_roots =
            crate::shared::artifacts::artifact_io::ArtifactIo::export_store_roots();
        let input_embedding_refs = crate::input_embedding::raster::RasterInputEmbeddingRefs {
            source_id: "embedding-fixture".to_string(),
            embedding_source_root: "embedding-root".to_string(),
            prompt_token_ids_root: "token-root".to_string(),
            prompt_token_count: input_ref.row_count(),
            embedded_prompt_activations_ref: input_ref,
        };
        let layer_source = AuthenticatedGemmaPrefillLayerSource::from_model(
            "prefill-layer",
            &transformer_fixture.model,
        )
        .expect("prefill layer source");
        let raster_payload = crate::trace::with_checkpointing_enabled(true, || {
            crate::trace::start_inference_trace(&json!({ "test": "raster-prefill-layer" }));
            crate::prefill_range::run_raster(
                input_embedding_roots.clone(),
                &input_embedding_refs,
                &layer_source,
                None,
                InferenceControls {
                    prefill_token_range_width: Some(1),
                    ..InferenceControls::default()
                }
                .raster_sizing_controls()
                .expect("default sizing"),
            )
            .expect("raster prefill layer should run");
            crate::trace::checkpoint_payload_for_tests()
        });

        assert_eq!(
            checkpoint_commitments(&deterministic_payload, "prefill.range_finalize"),
            checkpoint_commitments(&raster_payload, "prefill.range_finalize")
        );
        assert_eq!(
            checkpoint_commitments(&deterministic_payload, "prefill.range").len(),
            2
        );
        assert_eq!(
            checkpoint_commitments(&deterministic_payload, "prefill.range"),
            checkpoint_commitments(&raster_payload, "prefill.range")
        );
    }

    #[test]
    fn deterministic_cpu_prefill_finalize_checkpoint_commitment_matches_raster() {
        let _trace_guard = trace_test_lock().lock().expect("trace test lock");
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let token_ids = vec![1, 2];
        let input_rows = vec![
            vec![
                Act::from_num(1.0),
                Act::from_num(0.0),
                Act::from_num(0.0),
                Act::from_num(0.0),
            ],
            vec![
                Act::from_num(0.0),
                Act::from_num(1.0),
                Act::from_num(0.0),
                Act::from_num(0.0),
            ],
        ];

        let deterministic_payload = crate::trace::with_checkpointing_enabled(true, || {
            crate::trace::start_inference_trace(
                &json!({ "test": "deterministic-prefill-finalize" }),
            );
            let (final_hidden_states, layer_caches) = crate::prefill_range::run_with_mode_internal(
                InternalActivationSequence::from_det_values(input_rows.clone()),
                &transformer_fixture.model,
                None,
                InferenceExecutionMode::Deterministic,
            )
            .expect("deterministic prefill layer should run");
            crate::prefill_finalize::run(
                &token_ids,
                &transformer_fixture.model,
                final_hidden_states,
                layer_caches,
                InferenceExecutionMode::Deterministic,
            )
            .expect("deterministic prefill finalize should run");
            crate::trace::checkpoint_payload_for_tests()
        });

        crate::shared::artifacts::artifact_io::ArtifactIo::reset_store();
        let input_ref = insert_activation_sequence_artifact_ref(
            "test.prefill.finalize.input_embedding",
            RasterActivationSequence::from_acts(input_rows),
        )
        .expect("input embedding activation ref");
        let input_embedding_roots =
            crate::shared::artifacts::artifact_io::ArtifactIo::export_store_roots();
        let input_embedding_refs = crate::input_embedding::raster::RasterInputEmbeddingRefs {
            source_id: "embedding-fixture".to_string(),
            embedding_source_root: "embedding-root".to_string(),
            prompt_token_ids_root: "token-root".to_string(),
            prompt_token_count: input_ref.row_count(),
            embedded_prompt_activations_ref: input_ref,
        };
        let layer_source = AuthenticatedGemmaPrefillLayerSource::from_model(
            "prefill-finalize-layer",
            &transformer_fixture.model,
        )
        .expect("prefill layer source");
        let finalize_source = AuthenticatedGemmaPrefillFinalizeSource::from_model(
            "prefill-finalize",
            &transformer_fixture.model,
        )
        .expect("prefill finalize source");
        let raster_payload = crate::trace::with_checkpointing_enabled(true, || {
            crate::trace::start_inference_trace(&json!({ "test": "raster-prefill-finalize" }));
            let (layer_roots, layer_refs) = crate::prefill_range::run_raster(
                input_embedding_roots.clone(),
                &input_embedding_refs,
                &layer_source,
                None,
                InferenceControls::default()
                    .raster_sizing_controls()
                    .expect("default sizing"),
            )
            .expect("raster prefill layer should run");
            crate::prefill_finalize::materialize_raster_input_roots_for_api(
                layer_roots,
                token_ids.len(),
                &finalize_source,
                layer_refs.final_hidden_states_ref,
                layer_refs.layer_caches,
                InferenceControls::default()
                    .raster_sizing_controls()
                    .expect("default sizing")
                    .projection_rows_per_tile,
            )
            .expect("raster prefill finalize should run");
            crate::trace::checkpoint_payload_for_tests()
        });

        assert_eq!(
            checkpoint_commitments(&deterministic_payload, "prefill.finalize"),
            checkpoint_commitments(&raster_payload, "prefill.finalize")
        );
    }

    #[test]
    fn deterministic_cpu_decode_select_checkpoint_commitment_matches_raster() {
        let _trace_guard = trace_test_lock().lock().expect("trace test lock");
        let det_logits = vec![Act::from_bits(2), Act::from_bits(5), Act::from_bits(3)];
        let internal_logits = InternalLogits::from_det_values(det_logits);

        let deterministic_payload = crate::trace::with_checkpointing_enabled(true, || {
            crate::trace::start_inference_trace(&json!({ "test": "deterministic-decode-select" }));
            let mut decode_state = DecodeState::new(
                vec![7],
                internal_logits.clone_f32(),
                crate::shared::model::transformer::TransformerDecodeState::default(),
            );
            decode_state.set_internal_logits(internal_logits.clone());
            run_decode_select_token(&mut decode_state, 1, InferenceExecutionMode::Deterministic)
                .expect("deterministic decode select should run");
            crate::trace::checkpoint_payload_for_tests()
        });

        let raster_payload = crate::trace::with_checkpointing_enabled(true, || {
            crate::trace::start_inference_trace(&json!({ "test": "raster-decode-select" }));
            let mut decode_state = DecodeState::new(
                vec![7],
                internal_logits.clone_f32(),
                crate::shared::model::transformer::TransformerDecodeState::default(),
            );
            decode_state.set_internal_logits(internal_logits.clone());
            crate::decode_select_token::materialize_run_raster_for_api(&mut decode_state, 1)
                .expect("raster decode select should run");
            crate::trace::checkpoint_payload_for_tests()
        });

        assert_eq!(
            checkpoint_commitments(&deterministic_payload, "decode.select_token"),
            checkpoint_commitments(&raster_payload, "decode.select_token")
        );
    }

    #[test]
    fn run_inference_with_controls_raster_can_pause_after_input_embedding() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: InferenceExecutionMode::Deterministic,
            sampling: SamplingConfig {
                max_new_tokens: Some(2),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
        };

        let paused = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                commit_checkpoints: false,
                terminal_checkpoint: Some("input.embedding".to_string()),
                raster: true,
                raster_detour: None,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: None,
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
                prefill_token_range_width: None,
                decode_layer_range_width: None,
                raster_tokenizer_bpe_pairs_per_tile: None,
                raster_tokenizer_bpe_pieces_per_tile: None,
                raster_output_byte_flush_bytes_per_tile: None,
            },
        )
        .expect("raster inference should stop after input embedding");

        match paused {
            InferenceRunOutcome::Paused(state) => {
                assert_eq!(state.terminal_checkpoint_id, "input.embedding");
                assert_eq!(
                    state.input_embedding.prompt_preparation.prompt_token_ids,
                    vec![1]
                );
                assert!(state
                    .input_embedding
                    .det_embedded_prompt_activations_sha256
                    .is_some());
                assert!(state.transformer_state_transition.is_none());
                assert!(state.output_decode.is_none());
            }
            InferenceRunOutcome::Completed(_) | InferenceRunOutcome::RasterPromptPrepared(_) => {
                panic!("expected paused inference")
            }
        }
    }

    #[test]
    fn run_inference_with_controls_raster_can_pause_after_prefill_prepare_aux() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: InferenceExecutionMode::Deterministic,
            sampling: SamplingConfig {
                max_new_tokens: Some(2),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
        };

        let paused = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                commit_checkpoints: false,
                terminal_checkpoint: Some("prefill.prepare_aux".to_string()),
                raster: true,
                raster_detour: None,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: None,
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
                prefill_token_range_width: None,
                decode_layer_range_width: None,
                raster_tokenizer_bpe_pairs_per_tile: None,
                raster_tokenizer_bpe_pieces_per_tile: None,
                raster_output_byte_flush_bytes_per_tile: None,
            },
        )
        .expect("raster inference should stop after prefill prepare aux");

        match paused {
            InferenceRunOutcome::Paused(state) => {
                assert_eq!(state.terminal_checkpoint_id, "prefill.prepare_aux");
                assert_eq!(
                    state.input_embedding.prompt_preparation.prompt_token_ids,
                    vec![1]
                );
                assert!(state.transformer_state_transition.is_none());
                assert!(state.output_decode.is_none());
                assert!(
                    state.raster_tile_invocations.unwrap_or(0) > 0,
                    "raster prefill checkpoint should include tile invocation count"
                );
            }
            InferenceRunOutcome::Completed(_) | InferenceRunOutcome::RasterPromptPrepared(_) => {
                panic!("expected paused raster inference")
            }
        }
    }

    #[test]
    fn run_inference_with_controls_raster_can_pause_after_prefill_layer() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: InferenceExecutionMode::Deterministic,
            sampling: SamplingConfig {
                max_new_tokens: Some(2),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
        };

        let paused = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                commit_checkpoints: false,
                terminal_checkpoint: Some("prefill.range_finalize".to_string()),
                raster: true,
                raster_detour: None,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: Some(2),
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
                prefill_token_range_width: None,
                decode_layer_range_width: None,
                raster_tokenizer_bpe_pairs_per_tile: None,
                raster_tokenizer_bpe_pieces_per_tile: None,
                raster_output_byte_flush_bytes_per_tile: None,
            },
        )
        .expect("raster inference should stop after prefill layer");

        match paused {
            InferenceRunOutcome::Paused(state) => {
                assert_eq!(state.terminal_checkpoint_id, "prefill.range_finalize");
                assert!(state.transformer_state_transition.is_none());
                assert!(state.output_decode.is_none());
            }
            InferenceRunOutcome::Completed(_) | InferenceRunOutcome::RasterPromptPrepared(_) => {
                panic!("expected paused raster inference")
            }
        }
    }

    #[test]
    fn run_inference_with_controls_pauses_after_second_prefill_layer_checkpoint() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let mut transformer_model = test_transformer_model();
        transformer_model
            .layers
            .push(transformer_model.layers[0].clone());
        let request = InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: InferenceExecutionMode::Fp32,
            sampling: SamplingConfig {
                max_new_tokens: Some(1),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
        };

        let paused = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_model,
            &InferenceControls {
                commit_checkpoints: false,
                terminal_checkpoint: Some("prefill.range_finalize:2".to_string()),
                raster: false,
                raster_detour: None,
                raster_tokenizer_source: None,
                raster_projection_rows_per_tile: None,
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
                prefill_token_range_width: None,
                decode_layer_range_width: None,
                raster_tokenizer_bpe_pairs_per_tile: None,
                raster_tokenizer_bpe_pieces_per_tile: None,
                raster_output_byte_flush_bytes_per_tile: None,
            },
        )
        .expect("inference should pause after the second prefill layer checkpoint");

        match paused {
            InferenceRunOutcome::Paused(state) => {
                assert_eq!(state.terminal_checkpoint_id, "prefill.range_finalize");
                assert!(state.transformer_state_transition.is_none());
                assert!(state.output_decode.is_none());
            }
            InferenceRunOutcome::Completed(_) => panic!("expected paused inference"),
            InferenceRunOutcome::RasterPromptPrepared(_) => panic!("expected paused inference"),
        }
    }

    #[test]
    fn run_inference_with_controls_raster_can_pause_after_prefill_finalize() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: InferenceExecutionMode::Deterministic,
            sampling: SamplingConfig {
                max_new_tokens: Some(2),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
        };

        let paused = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                commit_checkpoints: false,
                terminal_checkpoint: Some("prefill.finalize".to_string()),
                raster: true,
                raster_detour: None,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: Some(2),
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
                prefill_token_range_width: None,
                decode_layer_range_width: None,
                raster_tokenizer_bpe_pairs_per_tile: None,
                raster_tokenizer_bpe_pieces_per_tile: None,
                raster_output_byte_flush_bytes_per_tile: None,
            },
        )
        .expect("raster inference should stop after prefill finalize");

        match paused {
            InferenceRunOutcome::Paused(state) => {
                assert_eq!(state.terminal_checkpoint_id, "prefill.finalize");
                assert!(
                    state.raster_tile_invocations.unwrap_or(0) > 0,
                    "raster prefill should include tile invocation count"
                );
                assert!(state.transformer_state_transition.is_some());
                assert!(state.output_decode.is_none());
            }
            InferenceRunOutcome::Completed(_) | InferenceRunOutcome::RasterPromptPrepared(_) => {
                panic!("expected paused raster inference")
            }
        }
    }

    #[test]
    fn run_inference_with_controls_raster_runs_decode_select_token() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: InferenceExecutionMode::Deterministic,
            sampling: SamplingConfig {
                max_new_tokens: Some(2),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
        };

        let outcome = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                commit_checkpoints: false,
                terminal_checkpoint: None,
                raster: true,
                raster_detour: None,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: Some(2),
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
                prefill_token_range_width: None,
                decode_layer_range_width: None,
                raster_tokenizer_bpe_pairs_per_tile: None,
                raster_tokenizer_bpe_pieces_per_tile: None,
                raster_output_byte_flush_bytes_per_tile: None,
            },
        )
        .expect("raster inference should complete");

        match outcome {
            InferenceRunOutcome::Completed(state) => {
                assert_eq!(state.output_decode.generated_token_ids, vec![0, 0]);
                assert_eq!(
                    state.output_decode.generated_text,
                    "raster-helloraster-hello"
                );
                assert_eq!(state.output_decode.generated_token_count, 2);
                assert_eq!(state.output_decode.decode_transition_states.len(), 2);
                assert!(
                    state.raster_tile_invocations.expect("tile count") > 0,
                    "full raster inference should count raster tiles"
                );
            }
            InferenceRunOutcome::Paused(_) | InferenceRunOutcome::RasterPromptPrepared(_) => {
                panic!("expected completed raster inference")
            }
        }
    }

    #[test]
    fn run_inference_with_controls_raster_can_pause_after_decode_transition() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: InferenceExecutionMode::Deterministic,
            sampling: SamplingConfig {
                max_new_tokens: Some(2),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
        };

        let paused = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                commit_checkpoints: false,
                terminal_checkpoint: Some("decode.transition_finalize".to_string()),
                raster: true,
                raster_detour: None,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: Some(2),
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
                prefill_token_range_width: None,
                decode_layer_range_width: None,
                raster_tokenizer_bpe_pairs_per_tile: None,
                raster_tokenizer_bpe_pieces_per_tile: None,
                raster_output_byte_flush_bytes_per_tile: None,
            },
        )
        .expect("raster inference should pause after decode transition finalize");

        match paused {
            InferenceRunOutcome::Paused(state) => {
                assert_eq!(state.terminal_checkpoint_id, "decode.transition_finalize");
                let output_decode = state
                    .output_decode
                    .expect("partial output decode state should be present");
                assert_eq!(output_decode.generated_token_ids, vec![0]);
                assert_eq!(output_decode.generated_text, "raster-hello");
                assert_eq!(output_decode.generated_token_count, 1);
                assert_eq!(output_decode.decode_transition_states.len(), 1);
            }
            InferenceRunOutcome::Completed(_) | InferenceRunOutcome::RasterPromptPrepared(_) => {
                panic!("expected paused raster inference")
            }
        }
    }

    #[test]
    fn run_inference_rejects_decode_layer_range_terminal_checkpoint() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = deterministic_prompt_request(1);

        let error = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                terminal_checkpoint: Some("decode.layer_range".to_string()),
                decode_layer_range_width: Some(1),
                ..InferenceControls::default()
            },
        )
        .expect_err("decode layer range terminal checkpoint should be rejected");

        assert!(error
            .to_string()
            .contains("terminal checkpoint decode.layer_range is not supported"));
    }

    #[test]
    fn run_inference_with_controls_raster_uses_ple_ref_bridge() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_ple_model_fixture();
        let request = InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: InferenceExecutionMode::Deterministic,
            sampling: SamplingConfig {
                max_new_tokens: Some(1),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
        };

        let native = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls::default(),
        )
        .expect("native deterministic inference should complete");
        let raster = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                commit_checkpoints: false,
                terminal_checkpoint: None,
                raster: true,
                raster_detour: None,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: Some(2),
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
                prefill_token_range_width: None,
                decode_layer_range_width: None,
                raster_tokenizer_bpe_pairs_per_tile: None,
                raster_tokenizer_bpe_pieces_per_tile: None,
                raster_output_byte_flush_bytes_per_tile: None,
            },
        )
        .expect("raster inference should complete");

        let InferenceRunOutcome::Completed(native) = native else {
            panic!("expected native inference to complete");
        };
        let InferenceRunOutcome::Completed(raster) = raster else {
            panic!("expected completed raster inference");
        };
        assert_eq!(
            raster.output_decode.generated_token_ids,
            native.output_decode.generated_token_ids
        );
        assert!(raster.raster_tile_invocations.expect("tile count") > 0);
    }

    #[test]
    fn run_inference_with_controls_raster_can_pause_after_output_finalize() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: InferenceExecutionMode::Deterministic,
            sampling: SamplingConfig {
                max_new_tokens: Some(1),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
        };

        let paused = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                commit_checkpoints: false,
                terminal_checkpoint: Some("output.finalize".to_string()),
                raster: true,
                raster_detour: None,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: Some(2),
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
                prefill_token_range_width: None,
                decode_layer_range_width: None,
                raster_tokenizer_bpe_pairs_per_tile: None,
                raster_tokenizer_bpe_pieces_per_tile: None,
                raster_output_byte_flush_bytes_per_tile: None,
            },
        )
        .expect("raster inference should pause after output finalize");

        match paused {
            InferenceRunOutcome::Paused(state) => {
                assert_eq!(state.terminal_checkpoint_id, "output.finalize");
                assert!(state.transformer_state_transition.is_some());
                let output_decode = state.output_decode.expect("output phase should be present");
                assert_eq!(output_decode.generated_token_count, 1);
            }
            InferenceRunOutcome::Completed(_) | InferenceRunOutcome::RasterPromptPrepared(_) => {
                panic!("expected paused raster inference")
            }
        }
    }

    #[test]
    fn run_inference_with_controls_raster_rejects_non_deterministic_request() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_model = test_transformer_model();
        let request = InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: InferenceExecutionMode::Fp32,
            sampling: SamplingConfig {
                max_new_tokens: Some(0),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
        };

        let error = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_model,
            &InferenceControls {
                commit_checkpoints: false,
                terminal_checkpoint: None,
                raster: true,
                raster_detour: None,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: None,
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
                prefill_token_range_width: None,
                decode_layer_range_width: None,
                raster_tokenizer_bpe_pairs_per_tile: None,
                raster_tokenizer_bpe_pieces_per_tile: None,
                raster_output_byte_flush_bytes_per_tile: None,
            },
        )
        .expect_err("raster inference should reject fp32 requests");

        assert!(error
            .to_string()
            .contains("requires deterministic execution"));
    }

    #[test]
    fn run_inference_with_controls_raster_rejects_non_deterministic_model() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_model = test_transformer_model();
        let request = InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: InferenceExecutionMode::Deterministic,
            sampling: SamplingConfig {
                max_new_tokens: Some(0),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
        };

        let error = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_model,
            &InferenceControls {
                commit_checkpoints: false,
                terminal_checkpoint: None,
                raster: true,
                raster_detour: None,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: None,
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
                prefill_token_range_width: None,
                decode_layer_range_width: None,
                raster_tokenizer_bpe_pairs_per_tile: None,
                raster_tokenizer_bpe_pieces_per_tile: None,
                raster_output_byte_flush_bytes_per_tile: None,
            },
        )
        .expect_err("raster inference should reject fp32 models");

        assert!(error.to_string().contains(".detwgt artifact"));
    }

    #[test]
    fn run_inference_with_controls_raster_rejects_zero_projection_rows_per_tile() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: InferenceExecutionMode::Deterministic,
            sampling: SamplingConfig {
                max_new_tokens: Some(0),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
        };

        let error = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                commit_checkpoints: false,
                terminal_checkpoint: None,
                raster: true,
                raster_detour: None,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: Some(0),
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
                prefill_token_range_width: None,
                decode_layer_range_width: None,
                raster_tokenizer_bpe_pairs_per_tile: None,
                raster_tokenizer_bpe_pieces_per_tile: None,
                raster_output_byte_flush_bytes_per_tile: None,
            },
        )
        .expect_err("zero raster projection rows per tile should fail");

        assert!(error.to_string().contains("greater than zero"));
    }

    #[test]
    fn run_inference_with_controls_raster_rejects_zero_attention_kv_rows_per_tile() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: InferenceExecutionMode::Deterministic,
            sampling: SamplingConfig {
                max_new_tokens: Some(0),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
        };

        let error = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                commit_checkpoints: false,
                terminal_checkpoint: None,
                raster: true,
                raster_detour: None,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: None,
                raster_attention_kv_rows_per_tile: Some(0),
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
                prefill_token_range_width: None,
                decode_layer_range_width: None,
                raster_tokenizer_bpe_pairs_per_tile: None,
                raster_tokenizer_bpe_pieces_per_tile: None,
                raster_output_byte_flush_bytes_per_tile: None,
            },
        )
        .expect_err("zero raster attention KV rows per tile should fail");

        assert!(error.to_string().contains("greater than zero"));
    }

    #[test]
    fn run_inference_with_controls_raster_rejects_zero_sequence_rows_per_tile() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: InferenceExecutionMode::Deterministic,
            sampling: SamplingConfig {
                max_new_tokens: Some(0),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
        };

        let error = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                commit_checkpoints: false,
                terminal_checkpoint: None,
                raster: true,
                raster_detour: None,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: None,
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: Some(0),
                raster_head_rows_per_tile: None,
                prefill_token_range_width: None,
                decode_layer_range_width: None,
                raster_tokenizer_bpe_pairs_per_tile: None,
                raster_tokenizer_bpe_pieces_per_tile: None,
                raster_output_byte_flush_bytes_per_tile: None,
            },
        )
        .expect_err("zero raster sequence rows per tile should fail");

        assert!(error.to_string().contains("greater than zero"));
    }

    #[test]
    fn run_inference_with_controls_raster_rejects_zero_head_rows_per_tile() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_fixture = deterministic_no_ple_model_fixture();
        let request = InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: InferenceExecutionMode::Deterministic,
            sampling: SamplingConfig {
                max_new_tokens: Some(0),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
        };

        let error = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_fixture.model,
            &InferenceControls {
                commit_checkpoints: false,
                terminal_checkpoint: None,
                raster: true,
                raster_detour: None,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: None,
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: Some(0),
                prefill_token_range_width: None,
                decode_layer_range_width: None,
                raster_tokenizer_bpe_pairs_per_tile: None,
                raster_tokenizer_bpe_pieces_per_tile: None,
                raster_output_byte_flush_bytes_per_tile: None,
            },
        )
        .expect_err("zero raster head rows per tile should fail");

        assert!(error.to_string().contains("greater than zero"));
    }

    #[test]
    fn raster_sizing_controls_reject_zero_output_tokenizer_chunks() {
        let pair_error = InferenceControls {
            raster_tokenizer_bpe_pairs_per_tile: Some(0),
            ..InferenceControls::default()
        }
        .raster_sizing_controls()
        .expect_err("zero tokenizer pair chunk should fail");
        assert!(pair_error
            .to_string()
            .contains("BPE pairs per tile must be greater than zero"));

        let piece_error = InferenceControls {
            raster_tokenizer_bpe_pieces_per_tile: Some(0),
            ..InferenceControls::default()
        }
        .raster_sizing_controls()
        .expect_err("zero tokenizer piece chunk should fail");
        assert!(piece_error
            .to_string()
            .contains("BPE pieces per tile must be greater than zero"));

        let output_error = InferenceControls {
            raster_output_byte_flush_bytes_per_tile: Some(0),
            ..InferenceControls::default()
        }
        .raster_sizing_controls()
        .expect_err("zero output byte flush chunk should fail");
        assert!(output_error
            .to_string()
            .contains("output byte flush bytes per tile must be greater than zero"));
    }

    #[test]
    fn run_inference_with_controls_pauses_after_prefill_finalize_checkpoint() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_model = test_transformer_model();
        let request = InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: InferenceExecutionMode::Fp32,
            sampling: SamplingConfig {
                max_new_tokens: Some(2),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
        };

        let paused = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_model,
            &InferenceControls {
                commit_checkpoints: false,
                terminal_checkpoint: Some("prefill.finalize".to_string()),
                raster: false,
                raster_detour: None,
                raster_tokenizer_source: None,
                raster_projection_rows_per_tile: Some(0),
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
                prefill_token_range_width: None,
                decode_layer_range_width: None,
                raster_tokenizer_bpe_pairs_per_tile: None,
                raster_tokenizer_bpe_pieces_per_tile: None,
                raster_output_byte_flush_bytes_per_tile: None,
            },
        )
        .expect("inference should pause");

        match paused {
            InferenceRunOutcome::Paused(state) => {
                assert_eq!(state.terminal_checkpoint_id, "prefill.finalize");
                assert_eq!(state.raster_tile_invocations, None);
                assert_eq!(
                    state
                        .transformer_state_transition
                        .expect("transformer phase should be present")
                        .activation_states
                        .len(),
                    1
                );
                assert!(state.output_decode.is_none());
            }
            InferenceRunOutcome::Completed(_) => panic!("expected paused inference"),
            InferenceRunOutcome::RasterPromptPrepared(_) => panic!("expected paused inference"),
        }
    }

    #[test]
    fn inference_state_serializes_with_protocol_phase_keys() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_model = test_transformer_model();
        let request = InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: InferenceExecutionMode::Fp32,
            sampling: SamplingConfig {
                max_new_tokens: Some(1),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
        };

        let inference_state = run_inference(&request, &model, &tokenizer, &transformer_model)
            .expect("inference should succeed");
        let serialized = serde_json::to_value(&inference_state).expect("serialize inference state");
        let object = serialized
            .as_object()
            .expect("serialized inference state should be an object");

        assert!(object.contains_key("input_embedding"));
        assert!(object.contains_key("transformer_state_transition"));
        assert!(object.contains_key("output_decode"));
    }

    #[test]
    fn inference_state_deserializes_protocol_taxonomy_keys() {
        let serialized_shape = json!({
            "input_embedding": {
                "prompt_text": "prompt",
                "prompt_token_ids": [1],
                "prompt_token_ids_sha256": "prompt-digest",
                "embedded_prompt_activations_sha256": "embed-digest"
            },
            "transformer_state_transition": {
                "activation_states": [
                    {
                        "activations_sha256": "hidden-digest"
                    }
                ],
                "prefill_logits": {
                    "final_logits_sha256": "logits-digest"
                }
            },
            "output_decode": {
                "generated_token_ids": [0],
                "generated_token_ids_sha256": "generated-digest",
                "generated_text": "hello"
            }
        });

        let inference_state: super::InferenceState =
            serde_json::from_value(serialized_shape).expect("serialized shape should deserialize");

        assert_eq!(
            inference_state
                .input_embedding
                .prompt_preparation
                .prompt_token_ids,
            vec![1]
        );
        assert_eq!(
            inference_state
                .input_embedding
                .embedded_prompt_activations_sha256,
            "embed-digest"
        );
        assert_eq!(
            inference_state
                .transformer_state_transition
                .prefill_logits
                .final_logits_sha256,
            "logits-digest"
        );
        assert_eq!(inference_state.output_decode.generated_text, "hello");
    }

    #[test]
    fn routine_exports_support_manual_inference_orchestration() {
        let tokenizer = test_tokenizer();
        let model = test_model_spec();
        let transformer_model = test_transformer_model();
        let request = InferenceRequest {
            prompt_bytes: b"prompt".to_vec(),
            text_decoding_policy: TextDecodingPolicy::Utf8,
            add_generation_prompt: false,
            add_special_tokens: false,
            execution_mode: InferenceExecutionMode::Fp32,
            sampling: SamplingConfig {
                max_new_tokens: Some(1),
                temperature: Some(1.0),
                top_k: None,
                top_p: None,
            },
        };

        let prompt_preparation =
            run_prompt_prepare(&request, &model, &tokenizer).expect("prompt prepare");
        let token_embeddings = embed_input_tokens(
            &prompt_preparation.prompt_token_ids,
            transformer_model
                .embedding_table
                .as_ref()
                .expect("embedding table"),
        )
        .expect("embed tokens");
        let ple_inputs = run_prefill_prepare_aux(
            &prompt_preparation.prompt_token_ids,
            &transformer_model,
            &token_embeddings,
            InferenceExecutionMode::Fp32,
        )
        .expect("prefill prepare aux");
        let (final_hidden_states, layer_caches) = run_prefill_range(
            &token_embeddings.activations,
            &transformer_model,
            ple_inputs.as_ref(),
        )
        .expect("prefill layer");
        let prefill = run_prefill_finalize(
            &prompt_preparation.prompt_token_ids,
            &transformer_model,
            final_hidden_states,
            layer_caches,
            InferenceExecutionMode::Fp32,
        )
        .expect("prefill finalize");

        let mut decode_state = DecodeState::new(
            prompt_preparation.prompt_token_ids.clone(),
            prefill.transformer_state.prefill_logits.logits.clone(),
            prefill.transformer_decode_state.clone(),
        );
        decode_state.set_internal_logits(prefill.transformer_state.prefill_logits.clone_internal());
        let next_token =
            run_decode_select_token(&mut decode_state, 1, InferenceExecutionMode::Fp32)
                .expect("decode select token");
        let next_token = next_token.expect("should select a token");
        let decode_transition = decode_step_with_mode(
            std::mem::take(&mut decode_state.transformer_decode_state),
            next_token,
            &transformer_model,
            InferenceExecutionMode::Fp32,
        )
        .expect("decode transition");
        decode_state.set_internal_logits(decode_transition.prefill_logits.clone_internal());
        decode_state.transformer_decode_state = decode_transition.transformer_decode_state;
        finalize_decode_transition(&decode_state).expect("decode finalize trace");

        let output = run_output_finalize(decode_state, &tokenizer).expect("output finalize");
        assert_eq!(output.generated_token_ids, vec![0]);
        assert_eq!(output.generated_text, "hello");
        assert_eq!(output.stop_reason, OutputDecodeStopReason::MaxNewTokens);
    }

    fn test_model_spec() -> ModelSpec {
        ModelSpec {
            model_id: "gemma-4-test".to_string(),
            tokenizer_path: "tokenizer.json".into(),
            chat_template: "{{ messages[0].content }}".to_string(),
            bos_token: None,
            eos_token: None,
            unk_token: Some("<unk>".to_string()),
        }
    }

    fn test_tokenizer() -> tokenizers::Tokenizer {
        let vocab = [
            ("hello".to_string(), 0),
            ("prompt".to_string(), 1),
            ("<unk>".to_string(), 2),
        ]
        .into_iter()
        .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("<unk>".to_string())
            .build()
            .expect("word level tokenizer");
        let mut tokenizer = tokenizers::Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        tokenizer
    }

    fn test_gemma_tokenizer_source() -> AuthenticatedGemmaTokenizer {
        AuthenticatedGemmaTokenizer::new(test_gemma_tokenizer_spec())
    }

    fn test_native_matching_gemma_tokenizer_source() -> AuthenticatedGemmaTokenizer {
        AuthenticatedGemmaTokenizer::new(test_gemma_tokenizer_spec_with_output_token("hello"))
    }

    fn test_gemma_tokenizer_spec() -> GemmaTokenizerSpec {
        test_gemma_tokenizer_spec_with_output_token("raster-hello")
    }

    fn test_gemma_tokenizer_spec_with_output_token(output_token: &str) -> GemmaTokenizerSpec {
        GemmaTokenizerSpec::new(
            "digest".to_string(),
            vec![
                GemmaVocabEntry {
                    token: output_token.to_string(),
                    id: 0,
                },
                GemmaVocabEntry {
                    token: "prompt".to_string(),
                    id: 1,
                },
                GemmaVocabEntry {
                    token: "<unk>".to_string(),
                    id: 2,
                },
                GemmaVocabEntry {
                    token: "p".to_string(),
                    id: 3,
                },
                GemmaVocabEntry {
                    token: "r".to_string(),
                    id: 4,
                },
                GemmaVocabEntry {
                    token: "o".to_string(),
                    id: 5,
                },
                GemmaVocabEntry {
                    token: "m".to_string(),
                    id: 6,
                },
                GemmaVocabEntry {
                    token: "t".to_string(),
                    id: 7,
                },
                GemmaVocabEntry {
                    token: "pr".to_string(),
                    id: 8,
                },
                GemmaVocabEntry {
                    token: "pro".to_string(),
                    id: 9,
                },
                GemmaVocabEntry {
                    token: "prom".to_string(),
                    id: 10,
                },
                GemmaVocabEntry {
                    token: "promp".to_string(),
                    id: 11,
                },
            ],
            vec![
                GemmaBpeMerge {
                    left: "p".to_string(),
                    right: "r".to_string(),
                    merged: "pr".to_string(),
                    rank: 0,
                },
                GemmaBpeMerge {
                    left: "pr".to_string(),
                    right: "o".to_string(),
                    merged: "pro".to_string(),
                    rank: 1,
                },
                GemmaBpeMerge {
                    left: "pro".to_string(),
                    right: "m".to_string(),
                    merged: "prom".to_string(),
                    rank: 2,
                },
                GemmaBpeMerge {
                    left: "prom".to_string(),
                    right: "p".to_string(),
                    merged: "promp".to_string(),
                    rank: 3,
                },
                GemmaBpeMerge {
                    left: "promp".to_string(),
                    right: "t".to_string(),
                    merged: "prompt".to_string(),
                    rank: 4,
                },
            ],
            vec![GemmaAddedToken {
                id: 2,
                content: "<unk>".to_string(),
                special: true,
            }],
            "<unk>".to_string(),
            true,
            "▁".to_string(),
            " ".to_string(),
        )
        .expect("test Gemma tokenizer spec should build")
    }

    struct DeterministicModelFixture {
        model: Gemma4TransformerModel,
        weights_files: Vec<std::path::PathBuf>,
    }

    impl Drop for DeterministicModelFixture {
        fn drop(&mut self) {
            for weights_file in &self.weights_files {
                let _ = std::fs::remove_file(weights_file);
            }
        }
    }

    fn deterministic_no_ple_model_fixture() -> DeterministicModelFixture {
        let embedding_rows = vec![
            vec![Act::from_num(0.0); 4],
            vec![Act::from_num(0.0); 4],
            vec![Act::from_num(0.0); 4],
        ];
        let (weights_file, source) = write_det_embedding_weights(embedding_rows);
        let mut model = test_transformer_model();
        let (layer_weights_file, layer_sources) = write_det_layer_weights(vec![
            det_zero_matrix(4, 4),
            det_zero_matrix(2, 4),
            det_zero_matrix(2, 4),
            det_zero_matrix(4, 4),
            det_zero_matrix(8, 4),
            det_zero_matrix(8, 4),
            det_zero_matrix(4, 8),
        ]);
        let mut layer_sources = layer_sources.into_iter();
        model.provenance = Gemma4ModelProvenance::DetNumWgt;
        model.embedding_table = None;
        model.embedding_source = Some(GemmaEmbeddingTensorSource::Deterministic {
            source,
            scale: 1.0,
            det_cache: Arc::new(Mutex::new(None::<Arc<DetNumMatrix>>)),
        });
        model.logits_projection = Gemma4LogitsProjection::UntiedLmHead {
            weight: zero_matrix(3, 4),
            det_weight: Some(Arc::new(DetNumMatrix {
                rows: 3,
                cols: 4,
                values: vec![Wgt::from_num(0.0).to_bits(); 12],
            })),
        };
        model.rms_norm_eps_det = Some(f32_to_acc(model.rms_norm_eps));
        model.final_norm_weight_det = Some(vec![Wgt::from_num(1.0); 4]);
        let layer = &mut model.layers[0];
        layer.q_proj =
            Gemma4LayerMatrixSource::from_det_num_source(layer_sources.next().expect("q source"));
        layer.k_proj =
            Gemma4LayerMatrixSource::from_det_num_source(layer_sources.next().expect("k source"));
        layer.v_proj = Some(Gemma4LayerMatrixSource::from_det_num_source(
            layer_sources.next().expect("v source"),
        ));
        layer.o_proj =
            Gemma4LayerMatrixSource::from_det_num_source(layer_sources.next().expect("o source"));
        layer.gate_proj = Gemma4LayerMatrixSource::from_det_num_source(
            layer_sources.next().expect("gate source"),
        );
        layer.up_proj =
            Gemma4LayerMatrixSource::from_det_num_source(layer_sources.next().expect("up source"));
        layer.down_proj = Gemma4LayerMatrixSource::from_det_num_source(
            layer_sources.next().expect("down source"),
        );
        layer.rms_norm_eps_det = Some(f32_to_acc(layer.rms_norm_eps));
        layer.rope_base_det = Some(f32_to_acc(layer.rope_base));
        layer.q_norm_weight_det = Some(vec![Wgt::from_num(1.0); 2]);
        layer.k_norm_weight_det = Some(vec![Wgt::from_num(1.0); 2]);
        layer.input_layernorm_weight_det = Some(vec![Wgt::from_num(1.0); 4]);
        layer.post_attention_layernorm_weight_det = Some(vec![Wgt::from_num(1.0); 4]);
        layer.pre_feedforward_layernorm_weight_det = Some(vec![Wgt::from_num(1.0); 4]);
        layer.post_feedforward_layernorm_weight_det = Some(vec![Wgt::from_num(1.0); 4]);
        DeterministicModelFixture {
            model,
            weights_files: vec![weights_file, layer_weights_file],
        }
    }

    fn deterministic_ple_model_fixture() -> DeterministicModelFixture {
        let mut fixture = deterministic_no_ple_model_fixture();
        let (ple_layer_weights_file, ple_layer_sources) =
            write_det_layer_weights(vec![det_zero_matrix(2, 4), det_zero_matrix(4, 2)]);
        let mut ple_layer_sources = ple_layer_sources.into_iter();
        fixture.model.layers[0].ple = Some(Gemma4PleLayerWeights {
            input_gate: Gemma4LayerMatrixSource::from_det_num_source(
                ple_layer_sources
                    .next()
                    .expect("PLE input gate layer source"),
            ),
            layer_projection: Gemma4LayerMatrixSource::from_det_num_source(
                ple_layer_sources
                    .next()
                    .expect("PLE layer projection source"),
            ),
            post_input_norm_weight: vec![1.0; 4],
            post_input_norm_weight_det: Some(vec![Wgt::from_num(1.0); 4]),
        });

        let (ple_global_weights_file, ple_global_sources) =
            write_det_layer_weights(vec![det_zero_matrix(3, 2), det_zero_matrix(2, 4)]);
        let mut ple_global_sources = ple_global_sources.into_iter();
        fixture.model.ple_global =
            Some(Gemma4PleGlobalWeights::from_det_num_sources_with_canonical(
                vec![ple_global_sources.next().expect("PLE token embeddings")],
                vec![ple_global_sources.next().expect("PLE model projection")],
                vec![1.0; 2],
                vec![Wgt::from_num(1.0); 2],
                1.0,
                Act::from_num(1.0),
                1.0,
                Act::from_num(1.0),
                1.0,
                Act::from_num(1.0),
            ));
        fixture.weights_files.push(ple_layer_weights_file);
        fixture.weights_files.push(ple_global_weights_file);
        fixture
    }

    fn write_det_embedding_weights(
        rows: Vec<Vec<Act>>,
    ) -> (std::path::PathBuf, DetNumTensorSliceSource) {
        let unique_suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        let unique_counter = next_fixture_counter();
        let path = std::env::temp_dir().join(format!(
            "raster-lib-det-embedding-{}-{unique_suffix}-{unique_counter}.detwgt",
            std::process::id(),
        ));
        let mut bytes = Vec::new();
        for row in &rows {
            for value in row {
                bytes.extend(value.to_bits().to_le_bytes());
            }
        }
        std::fs::write(&path, bytes).expect("det embedding fixture should write");
        let row_count = rows.len();
        let col_count = rows.first().map(Vec::len).unwrap_or(0);
        (
            path.clone(),
            DetNumTensorSliceSource {
                weights_path: path,
                total_rows: row_count,
                total_cols: col_count,
                data_offset: 0,
                row_offset: 0,
                row_count,
                col_offset: 0,
                col_count,
            },
        )
    }

    fn write_det_layer_weights(
        matrices: Vec<Vec<Vec<Wgt>>>,
    ) -> (std::path::PathBuf, Vec<DetNumTensorSliceSource>) {
        let unique_suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        let unique_counter = next_fixture_counter();
        let path = std::env::temp_dir().join(format!(
            "raster-lib-det-layer-{}-{unique_suffix}-{unique_counter}.detwgt",
            std::process::id(),
        ));
        let mut bytes = Vec::new();
        let mut sources = Vec::new();
        for matrix in matrices {
            let data_offset = bytes.len();
            for row in &matrix {
                for value in row {
                    bytes.extend(value.to_bits().to_le_bytes());
                }
            }
            sources.push(DetNumTensorSliceSource {
                weights_path: path.clone(),
                total_rows: matrix.len(),
                total_cols: matrix.first().map(Vec::len).unwrap_or(0),
                data_offset,
                row_offset: 0,
                row_count: matrix.len(),
                col_offset: 0,
                col_count: matrix.first().map(Vec::len).unwrap_or(0),
            });
        }
        std::fs::write(&path, bytes).expect("det layer fixture should write");
        (path, sources)
    }

    fn det_zero_matrix(rows: usize, cols: usize) -> Vec<Vec<Wgt>> {
        vec![vec![Wgt::from_num(0.0); cols]; rows]
    }

    fn next_fixture_counter() -> u64 {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        COUNTER.fetch_add(1, Ordering::Relaxed)
    }

    fn test_transformer_model() -> Gemma4TransformerModel {
        Gemma4TransformerModel {
            provenance: Gemma4ModelProvenance::Fp32,
            embedding_table: Some(EmbeddingTable {
                rows: vec![
                    vec![0.0, 0.0, 0.0, 0.0],
                    vec![0.0, 0.0, 0.0, 0.0],
                    vec![0.0, 0.0, 0.0, 0.0],
                ],
                scale: 1.0,
            }),
            embedding_source: None,
            layers: vec![Gemma4LayerWeights {
                attention_kind: Gemma4AttentionKind::Sliding,
                hidden_size: 4,
                num_heads: 2,
                num_kv_heads: 1,
                head_dim: 2,
                sliding_window: Some(2),
                cache_sliding_window: Some(2),
                rms_norm_eps: 1e-6,
                rms_norm_eps_det: None,
                rope_base: 10_000.0,
                rope_base_det: None,
                partial_rotary_dim: 2,
                rope_freq_base_dim: 2,
                kv_shared_layer_index: None,
                attention_k_eq_v: false,
                q_proj: zero_matrix(4, 4).into(),
                k_proj: zero_matrix(2, 4).into(),
                v_proj: Some(zero_matrix(2, 4).into()),
                o_proj: zero_matrix(4, 4).into(),
                q_norm_weight: vec![1.0, 1.0],
                q_norm_weight_det: None,
                k_norm_weight: vec![1.0, 1.0],
                k_norm_weight_det: None,
                input_layernorm_weight: vec![1.0; 4],
                input_layernorm_weight_det: None,
                post_attention_layernorm_weight: vec![1.0; 4],
                post_attention_layernorm_weight_det: None,
                pre_feedforward_layernorm_weight: vec![1.0; 4],
                pre_feedforward_layernorm_weight_det: None,
                post_feedforward_layernorm_weight: vec![1.0; 4],
                post_feedforward_layernorm_weight_det: None,
                gate_proj: zero_matrix(8, 4).into(),
                up_proj: zero_matrix(8, 4).into(),
                down_proj: zero_matrix(4, 8).into(),
                ple: None,
                layer_scalar: None,
                layer_scalar_det: None,
            }],
            ple_global: None,
            final_norm_weight: vec![1.0; 4],
            final_norm_weight_det: None,
            logits_projection: Gemma4LogitsProjection::UntiedLmHead {
                weight: zero_matrix(3, 4),
                det_weight: None,
            },
            final_logit_softcapping: None,
            final_logit_softcapping_det: None,
            rms_norm_eps: 1e-6,
            rms_norm_eps_det: None,
        }
    }

    fn zero_matrix(rows: usize, cols: usize) -> MatrixF32 {
        MatrixF32 {
            rows,
            cols,
            values: vec![0.0; rows * cols],
        }
    }
}
