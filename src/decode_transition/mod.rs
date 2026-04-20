use anyhow::Result;
use serde_json::json;

use crate::shared::output::DecodeState;
use crate::shared::transformer::{Gemma4TransformerModel, TransformerDecodeStepResult, TransformerDecodeState};

pub mod tiles;

pub fn run(
    transformer_decode_state: TransformerDecodeState,
    next_token: u32,
    model: &Gemma4TransformerModel,
) -> Result<TransformerDecodeStepResult> {
    let TransformerDecodeState {
        layer_caches,
        position,
        token_count,
    } = transformer_decode_state;
    let embedded_token = if let Some(ref embedding_table) = model.embedding_table {
        crate::shared::transformer_kernels::embed_input_token(next_token, embedding_table)?
    } else if let Some(ref embedding_source) = model.embedding_source {
        let embedded =
            crate::io::embed_input_tokens_from_gemma_source(&[next_token], embedding_source)?;
        embedded
            .activations
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("transformer embedding returned no activation rows"))?
    } else {
        anyhow::bail!(
            "transformer state model is missing both embedding_table and embedding_source"
        )
    };
    let final_hidden_state =
        tiles::run_text_layers_decode_step(&embedded_token, next_token, model, layer_caches, position)?;
    let prefill_logits = crate::shared::transformer_kernels::project_decode_hidden_to_logits(
        &final_hidden_state.activation_state.activations[0],
        &model.final_norm_weight,
        model.rms_norm_eps,
        &model.logits_projection,
        model.final_logit_softcapping,
    )?;

    Ok(TransformerDecodeStepResult {
        transformer_decode_state: TransformerDecodeState {
            layer_caches: final_hidden_state.layer_caches,
            position: position + 1,
            token_count: token_count + 1,
        },
        activation_state: final_hidden_state.activation_state,
        prefill_logits,
    })
}

pub fn finalize(decode_state: &DecodeState) -> Result<()> {
    crate::trace::trace_checkpoint(
        "decode.finalize",
        &json!({
            "full_token_ids": decode_state.full_token_ids.clone(),
            "full_token_ids_sha256": crate::trace::sha256_hex(&decode_state.full_token_ids),
            "generated_token_ids": decode_state.generated_token_ids.clone(),
            "generated_token_ids_sha256": crate::output_decode::build_output_decode_commitment(&decode_state.generated_token_ids)?,
            "current_logits": decode_state.current_logits.clone(),
            "current_logits_sha256": crate::trace::sha256_hex(&decode_state.current_logits),
            "decode_position": decode_state.transformer_decode_state.position,
            "decode_token_count": decode_state.transformer_decode_state.token_count,
            "layer_caches": crate::trace::serialize_layer_caches(&decode_state.transformer_decode_state.layer_caches),
        }),
    );
    Ok(())
}
