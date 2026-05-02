use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokenizers::Tokenizer;

pub mod checkpoints;
pub mod decode_select_token;
pub mod decode_transition;
pub mod io;
pub mod output_finalize;
mod pipeline;
pub mod prefill_finalize;
pub mod prefill_layer;
pub mod prefill_prepare_aux;
pub mod prompt_prepare;
pub mod raster_authoring;
pub mod shared;
pub mod trace;

pub use checkpoints::{classify_checkpoint, CheckpointTaxonomy, PhaseId, RoutineId};
pub use decode_select_token::run as run_decode_select_token;
pub use decode_transition::tiles::run_text_layers_decode_step;
pub use decode_transition::{finalize as finalize_decode_transition, run as run_decode_transition};
pub use io::{
    load_chat_template, load_embedding_table_from_gemma_model_path, load_embedding_table_from_path,
    load_gemma_tokenizer_spec_from_path, load_tokenizer_from_path,
    load_transformer_state_model_from_det_num_wgt_path,
    load_transformer_state_model_from_gemma_model_path,
};
pub use output_finalize::run as run_output_finalize;
pub use pipeline::{
    decode_step, decode_step_with_mode, run_output_decode, run_output_decode_with_mode,
    run_prefill_pass, run_prefill_pass_with_mode, run_transformer_state_transition,
    run_transformer_state_transition_for_token_ids, validate_sampling_config,
};
pub use prefill_finalize::run as run_prefill_finalize;
pub use prefill_layer::run as run_prefill_layer;
pub use prefill_layer::run_with_mode as run_prefill_layer_with_mode;
pub use prefill_layer::tiles::{run_text_layers_prefill, run_text_layers_prefill_with_cache};
pub use prefill_prepare_aux::run as run_prefill_prepare_aux;
pub use prompt_prepare::run as run_prompt_prepare;
pub use shared::gemma_tokenizer::{
    AuthenticatedGemmaTokenizer, GemmaAddedToken, GemmaBpeMerge, GemmaBpeState, GemmaTokenizerSpec,
    GemmaVocabEntry,
};
pub use shared::input::{
    Gemma4Prompt, InferenceExecutionMode, InferenceRequest, MessageRole, ModelSpec,
    PromptPreparationState, SamplingConfig, TextDecodingPolicy, TextMessage,
};
pub use shared::output::{DecodeState, OutputDecodeState, OutputDecodeStopReason};
pub use shared::transformer::{
    ActivationSequence, EmbeddedTokenSequence, EmbeddingTable, Gemma4AttentionKind,
    Gemma4LayerWeights, Gemma4LogitsProjection, Gemma4ModelProvenance, Gemma4PleGlobalWeights,
    Gemma4PleLayerWeights, Gemma4PrefillPleInputs, Gemma4TransformerModel,
    GemmaEmbeddingTensorSource, LayerKvCache, MatrixF32, PrefillLogits, TransformerDecodeState,
    TransformerDecodeStepResult, TransformerPrefillResult, TransformerStateTransitionState,
};
pub use shared::transformer_kernels::{
    append_kv_cache, apply_final_logit_softcapping, apply_final_norm, compute_decode_ple_input,
    compute_prefill_ple_inputs, embed_input_token, embed_input_token_with_mode, embed_input_tokens,
    embed_input_tokens_with_mode, extract_prefill_logits, project_decode_hidden_to_logits,
    project_to_logits, run_gemma4_layer, run_gemma4_layer_decode, select_final_position,
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
    pub raster_tiles: bool,
    pub raster_decode_only: bool,
    pub raster_tokenizer_source: Option<AuthenticatedGemmaTokenizer>,
    pub raster_projection_rows_per_tile: Option<usize>,
    pub raster_attention_kv_rows_per_tile: Option<usize>,
    pub raster_sequence_rows_per_tile: Option<usize>,
    pub raster_head_rows_per_tile: Option<usize>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct RasterSizingControls {
    pub projection_rows_per_tile: usize,
    pub attention_kv_rows_per_tile: usize,
    pub sequence_rows_per_tile: usize,
    pub head_rows_per_tile: usize,
}

impl InferenceControls {
    pub const DEFAULT_RASTER_PROJECTION_ROWS_PER_TILE: usize = 1;
    pub const DEFAULT_RASTER_ATTENTION_KV_ROWS_PER_TILE: usize = 32;
    pub const DEFAULT_RASTER_SEQUENCE_ROWS_PER_TILE: usize = 1;
    pub const DEFAULT_RASTER_HEAD_ROWS_PER_TILE: usize = 1;

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

    pub fn raster_sizing_controls(&self) -> Result<RasterSizingControls> {
        Ok(RasterSizingControls {
            projection_rows_per_tile: self.raster_projection_rows_per_tile()?,
            attention_kv_rows_per_tile: self.raster_attention_kv_rows_per_tile()?,
            sequence_rows_per_tile: self.raster_sequence_rows_per_tile()?,
            head_rows_per_tile: self.raster_head_rows_per_tile()?,
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

#[derive(Debug, Clone, PartialEq)]
pub enum InferenceRunOutcome {
    Completed(InferenceState),
    Paused(PausedInferenceState),
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
            if controls.raster_decode_only && !controls.raster_tiles {
                anyhow::bail!("raster decode-only inference requires raster tile inference");
            }
            if controls.raster_tiles
                && request.execution_mode != InferenceExecutionMode::Deterministic
            {
                anyhow::bail!("raster tile inference requires deterministic execution");
            }
            let use_raster_prefill = controls.raster_tiles && !controls.raster_decode_only;
            let use_raster_decode = controls.raster_tiles;
            let raster_sizing_controls = if controls.raster_tiles {
                Some(controls.raster_sizing_controls()?)
            } else {
                None
            };
            transformer_model.validate_execution_mode(request.execution_mode)?;
            trace::start_inference_trace(&json!({
                "model_id": model.model_id,
                "execution_mode": request.execution_mode,
                "det_num_spec_version": crate::shared::det_num::DET_NUM_SPEC_VERSION,
                "model_provenance": format!("{:?}", transformer_model.provenance),
                "prompt_bytes_sha256": trace::sha256_hex(&request.prompt_bytes),
                "max_new_tokens": request.sampling.max_new_tokens,
                "transformer_layer_count": transformer_model.layers.len(),
                "terminal_checkpoint": terminal_checkpoint.as_ref().map(|checkpoint| checkpoint.checkpoint_id()),
                "terminal_checkpoint_occurrence": terminal_checkpoint.as_ref().map(|checkpoint| checkpoint.occurrence()),
                "commit_checkpoints": controls.commit_checkpoints,
                "tile_authoring_mode": if use_raster_prefill { "raster" } else if use_raster_decode { "raster_decode_only" } else { "native" },
                "raster_decode_only": controls.raster_decode_only,
                "raster_sizing_controls": raster_sizing_controls,
            }));
            if use_raster_decode {
                crate::raster_authoring::start_tile_invocation_counting();
            }

            let result = (|| {
                trace::phase_started(PhaseId::InputEmbedding);
                let prompt_preparation = if use_raster_prefill {
                    let tokenizer_source = controls.raster_tokenizer_source.as_ref().context(
                        "raster tile inference requires an authenticated Gemma tokenizer",
                    )?;
                    prompt_prepare::run_raster(request, model, tokenizer_source)?
                } else {
                    run_prompt_prepare(request, model, tokenizer)?
                };
                let token_embeddings = if let Some(embedding_table) =
                    transformer_model.embedding_table.as_ref()
                {
                    embed_input_tokens_with_mode(
                        &prompt_preparation.prompt_token_ids,
                        embedding_table,
                        request.execution_mode,
                    )?
                } else if let Some(embedding_source) = transformer_model.embedding_source.as_ref() {
                    io::embed_input_tokens_from_gemma_source_with_mode(
                        &prompt_preparation.prompt_token_ids,
                        embedding_source,
                        request.execution_mode,
                    )?
                } else {
                    anyhow::bail!(
                    "transformer state model is missing both embedding_table and embedding_source"
                )
                };
                let input_embedding = InputEmbeddingState {
                    prompt_preparation: prompt_preparation.clone(),
                    embedded_prompt_activations_sha256: token_embeddings.activations_sha256.clone(),
                    det_embedded_prompt_activations_sha256: token_embeddings
                        .det_activations_sha256
                        .clone(),
                };
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
                let (final_hidden_states, layer_caches) = if use_raster_prefill {
                    let mut raster_prefill_store =
                        crate::shared::raster_row_store::AuthenticatedRasterTensorStore::new();
                    let ple_source =
                        crate::shared::raster_prefill_ple::AuthenticatedGemmaPleSource::from_model(
                            model.model_id.clone(),
                            transformer_model,
                        )?;
                    let ple_input_refs = prefill_prepare_aux::run_raster_refs_with_store(
                        &prompt_preparation.prompt_token_ids,
                        &ple_source,
                        &token_embeddings,
                        raster_sizing_controls
                            .as_ref()
                            .map(|controls| controls.projection_rows_per_tile)
                            .expect("raster projection rows per tile should be validated"),
                        &mut raster_prefill_store,
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
                    crate::shared::raster_prefill_layer::AuthenticatedGemmaPrefillLayerSource::from_model(
                        model.model_id.clone(),
                        transformer_model,
                    )?;
                    prefill_layer::run_raster_with_store(
                        &mut raster_prefill_store,
                        &token_embeddings,
                        &layer_source,
                        ple_input_refs.as_ref(),
                        raster_sizing_controls.expect("raster sizing controls should be validated"),
                    )?
                } else {
                    let ple_inputs = run_prefill_prepare_aux(
                        &prompt_preparation.prompt_token_ids,
                        transformer_model,
                        &token_embeddings,
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
                    prefill_layer::run_with_mode_internal(
                        token_embeddings.clone_internal(),
                        transformer_model,
                        ple_inputs.as_ref(),
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
                let prefill = if use_raster_prefill {
                    let finalize_source =
                    crate::shared::raster_prefill_finalize::AuthenticatedGemmaPrefillFinalizeSource::from_model(
                        model.model_id.clone(),
                        transformer_model,
                    )?;
                    prefill_finalize::run_raster(
                        &prompt_preparation.prompt_token_ids,
                        &finalize_source,
                        final_hidden_states,
                        layer_caches,
                        raster_sizing_controls
                            .as_ref()
                            .map(|controls| controls.projection_rows_per_tile)
                            .expect("raster projection rows per tile should be validated"),
                    )?
                } else {
                    run_prefill_finalize(
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
                    pipeline::run_output_decode_with_mode_and_raster_tiles(
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
                    run_output_decode_with_mode(
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
                .then(crate::raster_authoring::stop_tile_invocation_counting)
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

#[cfg(test)]
mod tests {
    use serde_json::json;
    use tokenizers::{models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace};

    use super::{
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
    use crate::shared::det_num::{f32_to_acc, Act, Wgt};
    use crate::shared::gemma_tokenizer::GemmaAddedToken;
    use crate::shared::transformer::{
        DetNumMatrix, DetNumTensorSliceSource, Gemma4LayerMatrixSource, GemmaEmbeddingTensorSource,
    };
    use std::sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    };

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
                raster_tiles: false,
                raster_decode_only: false,
                raster_tokenizer_source: None,
                raster_projection_rows_per_tile: None,
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
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
        }
    }

    #[test]
    fn run_inference_with_controls_raster_tiles_can_pause_after_prefill_prepare_aux() {
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
                raster_tiles: true,
                raster_decode_only: false,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: None,
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
            },
        )
        .expect("raster inference should pause after prefill prepare aux");

        match paused {
            InferenceRunOutcome::Paused(state) => {
                assert_eq!(state.terminal_checkpoint_id, "prefill.prepare_aux");
                assert_eq!(
                    state.input_embedding.prompt_preparation.prompt_token_ids,
                    vec![1]
                );
                assert!(state.transformer_state_transition.is_none());
                assert!(state.output_decode.is_none());
            }
            InferenceRunOutcome::Completed(_) => panic!("expected paused raster inference"),
        }
    }

    #[test]
    fn run_inference_with_controls_raster_tiles_can_pause_after_prefill_layer() {
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
                raster_tiles: true,
                raster_decode_only: false,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: Some(2),
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
            },
        )
        .expect("raster inference should pause after prefill layer");

        match paused {
            InferenceRunOutcome::Paused(state) => {
                assert_eq!(state.terminal_checkpoint_id, "prefill.layer");
                assert_eq!(
                    state.input_embedding.prompt_preparation.prompt_token_ids,
                    vec![1]
                );
                assert!(state.transformer_state_transition.is_none());
                assert!(state.output_decode.is_none());
            }
            InferenceRunOutcome::Completed(_) => panic!("expected paused raster inference"),
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
                raster_tiles: false,
                raster_decode_only: false,
                raster_tokenizer_source: None,
                raster_projection_rows_per_tile: None,
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
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
        }
    }

    #[test]
    fn run_inference_with_controls_raster_tiles_can_pause_after_prefill_finalize() {
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
                raster_tiles: true,
                raster_decode_only: false,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: Some(2),
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
            },
        )
        .expect("raster inference should pause after prefill finalize");

        match paused {
            InferenceRunOutcome::Paused(state) => {
                assert_eq!(state.terminal_checkpoint_id, "prefill.finalize");
                assert!(
                    state.raster_tile_invocations.unwrap_or(0) > 0,
                    "paused raster inference should include tile invocation count"
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
            InferenceRunOutcome::Completed(_) => panic!("expected paused raster inference"),
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
                raster_tiles: true,
                raster_decode_only: true,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: Some(2),
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
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
        }
    }

    #[test]
    fn run_inference_with_controls_raster_tiles_runs_decode_select_token() {
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
                raster_tiles: true,
                raster_decode_only: false,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: Some(2),
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
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
            }
            InferenceRunOutcome::Paused(_) => panic!("expected completed raster inference"),
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
                raster_tiles: true,
                raster_decode_only: true,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: Some(2),
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
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
        }
    }

    #[test]
    fn run_inference_with_controls_raster_tiles_uses_ple_ref_bridge() {
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
                raster_tiles: true,
                raster_decode_only: false,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: Some(2),
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
            },
        )
        .expect("raster inference should complete");

        let InferenceRunOutcome::Completed(native) = native else {
            panic!("expected native inference to complete");
        };
        let InferenceRunOutcome::Completed(raster) = raster else {
            panic!("expected raster inference to complete");
        };
        assert_eq!(
            raster.output_decode.generated_token_ids,
            native.output_decode.generated_token_ids
        );
        assert_eq!(
            raster.output_decode.generated_token_count,
            native.output_decode.generated_token_count
        );
        assert!(raster.raster_tile_invocations.expect("tile count") > 0);
    }

    #[test]
    fn run_inference_with_controls_raster_tiles_can_pause_after_output_finalize() {
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
                raster_tiles: true,
                raster_decode_only: false,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: Some(2),
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
            },
        )
        .expect("raster inference should pause after output finalize");

        match paused {
            InferenceRunOutcome::Paused(state) => {
                assert_eq!(state.terminal_checkpoint_id, "output.finalize");
                let output_decode = state
                    .output_decode
                    .expect("output decode state should be present");
                assert_eq!(output_decode.generated_token_ids, vec![0]);
                assert_eq!(output_decode.generated_text, "raster-hello");
                assert_eq!(output_decode.generated_token_count, 1);
            }
            InferenceRunOutcome::Completed(_) => panic!("expected paused raster inference"),
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
                raster_tiles: true,
                raster_decode_only: true,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: Some(2),
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
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
                raster_tiles: true,
                raster_decode_only: true,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: Some(2),
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
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
        }
    }

    #[test]
    fn run_inference_with_controls_raster_tiles_rejects_non_deterministic_request() {
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
                raster_tiles: true,
                raster_decode_only: false,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: None,
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
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
                raster_tiles: true,
                raster_decode_only: true,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: None,
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
            },
        )
        .expect_err("raster decode-only inference should reject fp32 requests");

        assert!(error
            .to_string()
            .contains("requires deterministic execution"));
    }

    #[test]
    fn run_inference_with_controls_raster_tiles_rejects_non_deterministic_model() {
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
                raster_tiles: true,
                raster_decode_only: false,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: None,
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
            },
        )
        .expect_err("raster inference should reject fp32 models");

        assert!(error.to_string().contains(".detwgt artifact"));
    }

    #[test]
    fn run_inference_with_controls_raster_tiles_rejects_zero_projection_rows_per_tile() {
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
                raster_tiles: true,
                raster_decode_only: false,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: Some(0),
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
            },
        )
        .expect_err("zero raster projection rows per tile should fail");

        assert!(error.to_string().contains("greater than zero"));
    }

    #[test]
    fn run_inference_with_controls_raster_tiles_rejects_zero_attention_kv_rows_per_tile() {
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
                raster_tiles: true,
                raster_decode_only: false,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: None,
                raster_attention_kv_rows_per_tile: Some(0),
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
            },
        )
        .expect_err("zero raster attention KV rows per tile should fail");

        assert!(error.to_string().contains("greater than zero"));
    }

    #[test]
    fn run_inference_with_controls_raster_tiles_rejects_zero_sequence_rows_per_tile() {
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
                raster_tiles: true,
                raster_decode_only: false,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: None,
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: Some(0),
                raster_head_rows_per_tile: None,
            },
        )
        .expect_err("zero raster sequence rows per tile should fail");

        assert!(error.to_string().contains("greater than zero"));
    }

    #[test]
    fn run_inference_with_controls_raster_tiles_rejects_zero_head_rows_per_tile() {
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
                raster_tiles: true,
                raster_decode_only: false,
                raster_tokenizer_source: Some(test_gemma_tokenizer_source()),
                raster_projection_rows_per_tile: None,
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: Some(0),
            },
        )
        .expect_err("zero raster head rows per tile should fail");

        assert!(error.to_string().contains("greater than zero"));
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
                raster_tiles: false,
                raster_decode_only: false,
                raster_tokenizer_source: None,
                raster_projection_rows_per_tile: Some(0),
                raster_attention_kv_rows_per_tile: None,
                raster_sequence_rows_per_tile: None,
                raster_head_rows_per_tile: None,
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
