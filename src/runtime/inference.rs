use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokenizers::Tokenizer;

use crate::input_embedding::raster::auth_source::AuthenticatedGemmaInputEmbeddingSource;
use crate::runtime::checkpoints::PhaseId;
use crate::runtime::{pipeline, trace};
use crate::shared::api::input::{
    InferenceExecutionMode, InferenceRequest, ModelSpec, PromptPreparationState,
    RasterPromptPreparationState, SamplingConfig,
};
use crate::shared::api::output::OutputDecodeState;
use crate::shared::artifacts::artifact_io::ArtifactIo;
use crate::shared::artifacts::raster_artifact_store::RasterTokenIdSequenceRef;
use crate::shared::model::gemma_tokenizer::AuthenticatedGemmaTokenizer;
use crate::shared::model::transformer::{Gemma4TransformerModel, TransformerStateTransitionState};
use crate::shared::raster_contracts::prefill_ple::AuthenticatedGemmaPleSource;
use crate::{
    input_embedding, prefill_finalize, prefill_layer, prefill_prepare_aux, prompt_prepare,
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
    pub raster_decode_only: bool,
    pub raster_tokenizer_source: Option<AuthenticatedGemmaTokenizer>,
    pub raster_projection_rows_per_tile: Option<usize>,
    pub raster_attention_kv_rows_per_tile: Option<usize>,
    pub raster_sequence_rows_per_tile: Option<usize>,
    pub raster_head_rows_per_tile: Option<usize>,
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
    pub tokenizer_bpe_pairs_per_tile: usize,
    pub tokenizer_bpe_pieces_per_tile: usize,
    pub output_byte_flush_bytes_per_tile: usize,
}

impl InferenceControls {
    pub const DEFAULT_RASTER_PROJECTION_ROWS_PER_TILE: usize = 1;
    pub const DEFAULT_RASTER_ATTENTION_KV_ROWS_PER_TILE: usize = 32;
    pub const DEFAULT_RASTER_SEQUENCE_ROWS_PER_TILE: usize = 1;
    pub const DEFAULT_RASTER_HEAD_ROWS_PER_TILE: usize = 1;
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
            tokenizer_bpe_pairs_per_tile: self.raster_tokenizer_bpe_pairs_per_tile()?,
            tokenizer_bpe_pieces_per_tile: self.raster_tokenizer_bpe_pieces_per_tile()?,
            output_byte_flush_bytes_per_tile: self.raster_output_byte_flush_bytes_per_tile()?,
        })
    }

    fn terminal_checkpoint_spec(&self) -> Result<Option<trace::TerminalCheckpointSpec>> {
        self.terminal_checkpoint
            .as_deref()
            .map(trace::TerminalCheckpointSpec::parse)
            .transpose()
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
            if controls.raster_decode_only && !controls.raster {
                anyhow::bail!("raster decode-only inference requires raster tile inference");
            }
            if controls.raster && request.execution_mode != InferenceExecutionMode::Deterministic {
                anyhow::bail!("raster tile inference requires deterministic execution");
            }
            let use_raster_prefill = controls.raster && !controls.raster_decode_only;
            let use_raster_decode = controls.raster;
            let raster_sizing_controls = if controls.raster {
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
                "tile_dsl_mode": if use_raster_prefill { "raster" } else if use_raster_decode { "raster_decode_only" } else { "native" },
                "raster_decode_only": controls.raster_decode_only,
                "raster_sizing_controls": raster_sizing_controls,
            }));
            if use_raster_decode {
                crate::dsl::start_tile_invocation_counting();
            }
            let deterministic_prompt_checkpoint = |prompt_preparation: &PromptPreparationState| {
                let tokenizer_source = controls.raster_tokenizer_source.as_ref().context(
                    "deterministic CPU prompt.prepare checkpoint requires an authenticated Gemma tokenizer",
                )?;
                prompt_prepare::format_native_prompt_as_raster_checkpoint(
                    request,
                    model,
                    tokenizer_source,
                    prompt_preparation,
                )
            };

            let result = (|| {
                trace::phase_started(PhaseId::InputEmbedding);
                let mut raster_prompt_preparation_for_embedding = None;
                let mut raster_prompt_preparation_roots_for_embedding = None;
                let prompt_preparation = if use_raster_prefill {
                    let tokenizer_source = controls.raster_tokenizer_source.as_ref().context(
                        "raster tile inference requires an authenticated Gemma tokenizer",
                    )?;
                    let raster_sizing_controls = raster_sizing_controls
                        .as_ref()
                        .expect("raster sizing controls should be validated");
                    let raster_prompt_preparation =
                        prompt_prepare::run_raster_with_tokenizer_controls(
                            request,
                            model,
                            tokenizer_source,
                            raster_sizing_controls.tokenizer_bpe_pairs_per_tile,
                            raster_sizing_controls.tokenizer_bpe_pieces_per_tile,
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
                                "prompt_bytes_root": prompt_checkpoint.prompt_bytes_root.clone(),
                                "prompt_text_root": prompt_checkpoint.prompt_text_root.clone(),
                                "rendered_prompt_root": prompt_checkpoint.rendered_prompt_root.clone(),
                                "normalized_prompt_root": prompt_checkpoint.normalized_prompt_root.clone(),
                                "prompt_token_count": prompt_checkpoint.prompt_token_count,
                                "prompt_token_ids_root": prompt_checkpoint.prompt_token_ids_root.clone(),
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
                    let input_embedding_output = input_embedding::run_raster_output_with_roots(
                        raster_prompt_preparation_roots,
                        raster_prompt_preparation,
                        &embedding_source,
                    )?;
                    let token_embeddings = input_embedding::materialize_input_embedding_refs(
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
                            input_embedding::format_native_input_embedding_as_raster_checkpoint(
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
                input_embedding::trace_input_embedding_checkpoint(
                    &prompt_preparation.prompt_token_ids,
                    &token_embeddings,
                    raster_input_embedding_refs
                        .as_ref()
                        .map(|output| &output.refs),
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
                trace::phase_finished(PhaseId::InputEmbedding);

                trace::phase_started(PhaseId::TransformerStateTransition);
                let prefill = if use_raster_prefill {
                    let raster_sizing =
                        raster_sizing_controls.expect("raster sizing controls should be validated");
                    let ple_source = AuthenticatedGemmaPleSource::from_model(
                        model.model_id.clone(),
                        transformer_model,
                    )?;
                    let input_embedding_output = raster_input_embedding_refs
                        .as_ref()
                        .expect("raster prefill requires raster input embedding refs");
                    let (layer_roots, ple_input_manifest_root) =
                        prefill_prepare_aux::run_raster_refs_from_input_embedding(
                            input_embedding_output.artifact_store_roots.clone(),
                            &input_embedding_output.refs,
                            &ple_source,
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
                    let layer_source =
                    crate::shared::raster_contracts::prefill_layer::AuthenticatedGemmaPrefillLayerSource::from_model(
                        model.model_id.clone(),
                        transformer_model,
                    )?;
                    let (layer_roots, layer_refs) =
                        prefill_layer::run_raster_refs_from_input_embedding(
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
                    prefill_finalize::run_raster_refs_with_roots(
                        layer_roots,
                        prompt_preparation.prompt_token_ids.len(),
                        &finalize_source,
                        layer_refs.final_hidden_states_ref,
                        layer_refs.layer_caches,
                        raster_sizing.projection_rows_per_tile,
                    )?
                } else {
                    let ple_inputs = if let Some(input_embedding_output) =
                        raster_input_embedding_refs.as_ref()
                    {
                        let ple_source = AuthenticatedGemmaPleSource::from_model(
                            model.model_id.clone(),
                            transformer_model,
                        )?;
                        prefill_prepare_aux::run_with_input_embedding_checkpoint(
                            &prompt_preparation.prompt_token_ids,
                            transformer_model,
                            &token_embeddings,
                            request.execution_mode,
                            input_embedding_output.artifact_store_roots.clone(),
                            &input_embedding_output.refs,
                            &ple_source,
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
                    let (final_hidden_states, layer_caches) =
                        prefill_layer::run_with_mode_internal(
                            token_embeddings.clone_internal(),
                            transformer_model,
                            ple_inputs.as_ref(),
                            request.execution_mode,
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
                    prefill_finalize::run(
                        &prompt_preparation.prompt_token_ids,
                        transformer_model,
                        final_hidden_states,
                        layer_caches,
                        request.execution_mode,
                    )?
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
                trace::phase_finished(PhaseId::TransformerStateTransition);

                trace::phase_started(PhaseId::OutputDecode);
                let output_decode = if use_raster_decode {
                    let tokenizer_source = controls.raster_tokenizer_source.as_ref().context(
                        "raster tile inference requires an authenticated Gemma tokenizer",
                    )?;
                    pipeline::run_output_decode_with_mode_and_raster(
                        &prompt_preparation.prompt_token_ids,
                        &prefill,
                        &request.sampling,
                        tokenizer,
                        tokenizer_source,
                        transformer_model,
                        request.execution_mode,
                        raster_sizing_controls.expect("raster sizing controls should be validated"),
                    )?
                } else {
                    pipeline::run_output_decode_with_mode(
                        &prompt_preparation.prompt_token_ids,
                        &prefill,
                        &request.sampling,
                        tokenizer,
                        transformer_model,
                        request.execution_mode,
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
                trace::phase_finished(PhaseId::OutputDecode);

                Ok(InferenceRunOutcome::Completed(InferenceState {
                    input_embedding,
                    transformer_state_transition,
                    output_decode,
                    raster_tile_invocations: None,
                }))
            })();

            let raster_tile_invocations = use_raster_decode
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
            let read = ArtifactIo::read_leaf(token_ids_ref.artifact_ref(), token_idx)?;
            ArtifactIo::verify_artifact_read(token_ids_ref.artifact_ref(), &read)?;
            let payload = read.payload();
            let bytes: [u8; 4] = payload
                .try_into()
                .map_err(|_| anyhow::anyhow!("token-id leaf payload must be exactly four bytes"))?;
            Ok(u32::from_le_bytes(bytes))
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
    use crate::shared::tensors::raster_row_store::insert_activation_sequence_artifact_ref;
    use crate::{
        embed_input_tokens, finalize_decode_transition, run_decode_select_token,
        run_decode_transition, run_inference, run_inference_with_controls, run_output_finalize,
        run_prefill_finalize, run_prefill_layer, run_prefill_prepare_aux, run_prompt_prepare,
        AuthenticatedGemmaTokenizer, DecodeState, EmbeddingTable, Gemma4AttentionKind,
        Gemma4LayerWeights, Gemma4LogitsProjection, Gemma4ModelProvenance, Gemma4PleGlobalWeights,
        Gemma4PleLayerWeights, Gemma4TransformerModel, GemmaBpeMerge, GemmaTokenizerSpec,
        GemmaVocabEntry, InferenceControls, InferenceExecutionMode, InferenceRequest,
        InferenceRunOutcome, MatrixF32, ModelSpec, OutputDecodeStopReason, SamplingConfig,
        TextDecodingPolicy,
    };
    use std::sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, OnceLock,
    };

    fn trace_test_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn checkpoint_commitments(payload: &serde_json::Value, checkpoint: &str) -> Vec<String> {
        payload
            .as_array()
            .expect("checkpoint payload should be an array")
            .iter()
            .filter_map(|entry| entry.get(checkpoint)?.as_str().map(ToString::to_string))
            .collect()
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
                raster_decode_only: false,
                raster_tokenizer_source: None,
                raster_projection_rows_per_tile: None,
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
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
                raster_decode_only: false,
                raster_tokenizer_source: Some(tokenizer_source.clone()),
                raster_projection_rows_per_tile: None,
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
                raster_tokenizer_bpe_pairs_per_tile: None,
                raster_tokenizer_bpe_pieces_per_tile: None,
                raster_output_byte_flush_bytes_per_tile: None,
            },
        )
        .expect("deterministic inference should stop after prompt prepare");
        let expected = crate::prompt_prepare::run_raster(&request, &model, &tokenizer_source)
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
            crate::prefill_layer::run_with_mode_internal(
                InternalActivationSequence::from_det_values(input_rows.clone()),
                &transformer_fixture.model,
                None,
                InferenceExecutionMode::Deterministic,
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
            crate::prefill_layer::run_raster_refs_from_input_embedding(
                input_embedding_roots.clone(),
                &input_embedding_refs,
                &layer_source,
                None,
                InferenceControls::default()
                    .raster_sizing_controls()
                    .expect("default sizing"),
            )
            .expect("raster prefill layer should run");
            crate::trace::checkpoint_payload_for_tests()
        });

        assert_eq!(
            checkpoint_commitments(&deterministic_payload, "prefill.layer"),
            checkpoint_commitments(&raster_payload, "prefill.layer")
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
            let (final_hidden_states, layer_caches) = crate::prefill_layer::run_with_mode_internal(
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
            let (layer_roots, layer_refs) =
                crate::prefill_layer::run_raster_refs_from_input_embedding(
                    input_embedding_roots.clone(),
                    &input_embedding_refs,
                    &layer_source,
                    None,
                    InferenceControls::default()
                        .raster_sizing_controls()
                        .expect("default sizing"),
                )
                .expect("raster prefill layer should run");
            crate::prefill_finalize::run_raster_refs_with_roots(
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
            crate::decode_select_token::run_raster(&mut decode_state, 1)
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
                raster_decode_only: false,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: None,
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
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
                raster_decode_only: false,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: None,
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
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
                terminal_checkpoint: Some("prefill.layer".to_string()),
                raster: true,
                raster_decode_only: false,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: Some(2),
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
                raster_tokenizer_bpe_pairs_per_tile: None,
                raster_tokenizer_bpe_pieces_per_tile: None,
                raster_output_byte_flush_bytes_per_tile: None,
            },
        )
        .expect("raster inference should stop after prefill layer");

        match paused {
            InferenceRunOutcome::Paused(state) => {
                assert_eq!(state.terminal_checkpoint_id, "prefill.layer");
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
                terminal_checkpoint: Some("prefill.layer:2".to_string()),
                raster: false,
                raster_decode_only: false,
                raster_tokenizer_source: None,
                raster_projection_rows_per_tile: None,
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
                raster_tokenizer_bpe_pairs_per_tile: None,
                raster_tokenizer_bpe_pieces_per_tile: None,
                raster_output_byte_flush_bytes_per_tile: None,
            },
        )
        .expect("inference should pause after the second prefill layer checkpoint");

        match paused {
            InferenceRunOutcome::Paused(state) => {
                assert_eq!(state.terminal_checkpoint_id, "prefill.layer");
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
                raster_decode_only: false,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: Some(2),
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
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
    fn run_inference_with_controls_raster_decode_only_can_pause_after_prefill_finalize() {
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
                raster_decode_only: true,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: Some(2),
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
                raster_tokenizer_bpe_pairs_per_tile: None,
                raster_tokenizer_bpe_pieces_per_tile: None,
                raster_output_byte_flush_bytes_per_tile: None,
            },
        )
        .expect("hybrid raster inference should pause after prefill finalize");

        match paused {
            InferenceRunOutcome::Paused(state) => {
                assert_eq!(state.terminal_checkpoint_id, "prefill.finalize");
                assert_eq!(
                    state.raster_tile_invocations,
                    Some(0),
                    "decode-only raster should not invoke raster tiles before decode"
                );
                let transformer_state = state
                    .transformer_state_transition
                    .expect("transformer phase should be present");
                assert_eq!(transformer_state.activation_states.len(), 1);
                assert!(transformer_state
                    .prefill_logits
                    .det_final_logits_sha256
                    .is_some());
                assert!(state.output_decode.is_none());
            }
            InferenceRunOutcome::Completed(_) => panic!("expected paused hybrid raster inference"),
            InferenceRunOutcome::RasterPromptPrepared(_) => {
                panic!("expected paused hybrid raster inference")
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
                raster_decode_only: false,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: Some(2),
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
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
    fn run_inference_with_controls_raster_decode_only_runs_raster_decode() {
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
                raster_decode_only: true,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: Some(2),
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
                raster_tokenizer_bpe_pairs_per_tile: None,
                raster_tokenizer_bpe_pieces_per_tile: None,
                raster_output_byte_flush_bytes_per_tile: None,
            },
        )
        .expect("hybrid raster inference should complete");

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
                    "decode-only raster should count raster decode tiles"
                );
            }
            InferenceRunOutcome::Paused(_) => panic!("expected completed hybrid raster inference"),
            InferenceRunOutcome::RasterPromptPrepared(_) => {
                panic!("expected completed hybrid raster inference")
            }
        }
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
                raster_decode_only: false,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: Some(2),
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
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
                raster_decode_only: false,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: Some(2),
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
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
    fn run_inference_with_controls_raster_decode_only_can_pause_after_output_finalize() {
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
                raster_decode_only: true,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: Some(2),
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
                raster_tokenizer_bpe_pairs_per_tile: None,
                raster_tokenizer_bpe_pieces_per_tile: None,
                raster_output_byte_flush_bytes_per_tile: None,
            },
        )
        .expect("hybrid raster inference should pause after output finalize");

        match paused {
            InferenceRunOutcome::Paused(state) => {
                assert_eq!(state.terminal_checkpoint_id, "output.finalize");
                assert!(
                    state.raster_tile_invocations.expect("tile count") > 0,
                    "hybrid raster inference should count output decode tiles"
                );
                let output_decode = state
                    .output_decode
                    .expect("output decode state should be present");
                assert_eq!(output_decode.generated_token_ids, vec![0]);
                assert_eq!(output_decode.generated_text, "raster-hello");
                assert_eq!(output_decode.generated_token_count, 1);
            }
            InferenceRunOutcome::Completed(_) => panic!("expected paused hybrid raster inference"),
            InferenceRunOutcome::RasterPromptPrepared(_) => {
                panic!("expected paused hybrid raster inference")
            }
        }
    }

    #[test]
    fn run_inference_with_controls_raster_decode_only_can_pause_after_decode_finalize() {
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
                terminal_checkpoint: Some("decode.finalize".to_string()),
                raster: true,
                raster_decode_only: true,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: Some(2),
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
                raster_tokenizer_bpe_pairs_per_tile: None,
                raster_tokenizer_bpe_pieces_per_tile: None,
                raster_output_byte_flush_bytes_per_tile: None,
            },
        )
        .expect("hybrid raster inference should pause after decode finalize");

        match paused {
            InferenceRunOutcome::Paused(state) => {
                assert_eq!(state.terminal_checkpoint_id, "decode.finalize");
                let output_decode = state
                    .output_decode
                    .expect("partial output decode state should be present");
                assert_eq!(output_decode.generated_token_ids, vec![0]);
                assert_eq!(output_decode.generated_text, "raster-hello");
                assert_eq!(output_decode.generated_token_count, 1);
                assert_eq!(output_decode.decode_transition_states.len(), 1);
            }
            InferenceRunOutcome::Completed(_) => panic!("expected paused hybrid raster inference"),
            InferenceRunOutcome::RasterPromptPrepared(_) => {
                panic!("expected paused hybrid raster inference")
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
                raster_decode_only: false,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: None,
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
                raster_tokenizer_bpe_pairs_per_tile: None,
                raster_tokenizer_bpe_pieces_per_tile: None,
                raster_output_byte_flush_bytes_per_tile: None,
            },
        )
        .expect_err("raster inference should reject fp32 requests");

        assert!(error
            .to_string()
            .contains("requires deterministic execution"));

        let error = run_inference_with_controls(
            &request,
            &model,
            &tokenizer,
            &transformer_model,
            &InferenceControls {
                commit_checkpoints: false,
                terminal_checkpoint: None,
                raster: true,
                raster_decode_only: true,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: None,
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
                raster_tokenizer_bpe_pairs_per_tile: None,
                raster_tokenizer_bpe_pieces_per_tile: None,
                raster_output_byte_flush_bytes_per_tile: None,
            },
        )
        .expect_err("raster decode-only inference should reject fp32 requests");

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
                raster_decode_only: false,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: None,
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
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
                raster_decode_only: false,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: Some(0),
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
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
                raster_decode_only: false,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: None,
                raster_attention_kv_rows_per_tile: Some(0),
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
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
                raster_decode_only: false,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: None,
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: Some(0),
                raster_head_rows_per_tile: None,
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
                raster_decode_only: false,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: None,
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: Some(0),
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
                raster_decode_only: false,
                raster_tokenizer_source: None,
                raster_projection_rows_per_tile: Some(0),
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
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
        let (final_hidden_states, layer_caches) = run_prefill_layer(
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
        let decode_transition = run_decode_transition(
            std::mem::take(&mut decode_state.transformer_decode_state),
            next_token,
            &transformer_model,
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

    fn test_gemma_tokenizer_spec() -> GemmaTokenizerSpec {
        GemmaTokenizerSpec::new(
            "digest".to_string(),
            vec![
                GemmaVocabEntry {
                    token: "raster-hello".to_string(),
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
