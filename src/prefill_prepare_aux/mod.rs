use anyhow::Result;
use serde_json::json;

use crate::shared::input::InferenceExecutionMode;
use crate::shared::transformer::{
    ActivationSequence, Gemma4PrefillPleInputs, Gemma4TransformerModel,
};

pub fn run(
    prompt_token_ids: &[u32],
    model: &Gemma4TransformerModel,
    token_embeddings: &ActivationSequence,
    execution_mode: InferenceExecutionMode,
) -> Result<Option<Gemma4PrefillPleInputs>> {
    let ple_inputs = model
        .ple_global
        .as_ref()
        .map(|ple_global| {
            crate::shared::transformer_kernels::compute_prefill_ple_inputs_internal(
                prompt_token_ids,
                token_embeddings.clone_internal(),
                &model.layers,
                ple_global,
                model.rms_norm_eps,
                execution_mode,
            )
        })
        .transpose()?;
    crate::trace::trace_checkpoint(
        "prefill.prepare_aux",
        &json!({
            "prompt_token_ids": prompt_token_ids,
            "prompt_token_ids_sha256": crate::trace::sha256_hex(&prompt_token_ids),
            "embedded_prompt_activations": token_embeddings.activations.clone(),
            "embedded_prompt_activations_sha256": token_embeddings.activations_sha256.clone(),
            "per_layer_prefill_inputs": ple_inputs.as_ref().map(|inputs| inputs.per_layer_inputs.clone()),
            "per_layer_prefill_input_sha256s": ple_inputs.as_ref().map(|inputs| {
                inputs
                    .per_layer_inputs
                    .iter()
                    .map(|input| input.as_ref().map(crate::trace::sha256_hex))
                    .collect::<Vec<_>>()
            }),
        }),
    );
    Ok(ple_inputs)
}
