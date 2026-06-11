use serde_json::json;

use crate::runtime::checkpoints::RoutineId;
use crate::shared::model::transformer::{ActivationSequence, LayerKvCache};

pub mod raster;

pub(crate) struct PrefillRangeFinalizeCheckpoint<'a> {
    pub execution_mode: Option<&'a str>,
    pub layer_idx: usize,
    pub current_activations: &'a ActivationSequence,
    pub layer_caches: &'a [LayerKvCache],
    pub completed_layer_output_sha256s: Vec<String>,
    pub completed_layer_output_det_sha256s: Option<Vec<Option<String>>>,
}

pub(crate) fn trace_checkpoint(input: PrefillRangeFinalizeCheckpoint<'_>) -> bool {
    let deterministic = input.execution_mode == Some("deterministic");
    let current_internal = input.current_activations.clone_internal();
    let token_count = if deterministic {
        current_internal
            .det_values()
            .map(<[Vec<_>]>::len)
            .unwrap_or(0)
    } else {
        input.current_activations.activations.len()
    };
    let _routine = crate::trace::routine_scope(
        RoutineId::PrefillRangeFinalize,
        format!(
            "{}layer={} tokens={token_count}",
            input
                .execution_mode
                .map(|mode| format!("mode={mode} "))
                .unwrap_or_default(),
            input.layer_idx,
        ),
    );
    crate::trace::trace_checkpoint_lazy("prefill.range_finalize", || {
        let det_current_activations_sha256 = current_internal
            .det_values()
            .map(crate::shared::numerics::transformer_kernels::build_det_activation_commitment);
        let mut payload = json!({
            "next_layer_idx": input.layer_idx + 1,
            "det_current_activations_sha256": det_current_activations_sha256,
            "det_layer_caches_sha256": crate::shared::numerics::transformer_kernels::build_det_kv_cache_commitment(input.layer_caches),
        });
        if !deterministic {
            // Deterministic-mode payloads carry only canonical commitments
            // (spec v1); fp32 mode keeps the compatibility fields.
            let current_activations = input.current_activations.activations.clone();
            payload["current_activations_sha256"] = json!(
                crate::shared::numerics::transformer_kernels::build_activation_commitment(
                    &current_activations
                )
            );
            payload["current_activations"] = json!(current_activations);
            payload["layer_caches"] =
                json!(crate::trace::serialize_layer_caches(input.layer_caches));
            payload["completed_layer_output_sha256s"] = json!(input.completed_layer_output_sha256s);
        }
        if let Some(execution_mode) = input.execution_mode {
            payload["execution_mode"] = json!(execution_mode);
        }
        if let Some(completed_layer_output_det_sha256s) =
            input.completed_layer_output_det_sha256s.as_ref()
        {
            payload["completed_layer_output_det_sha256s"] =
                json!(completed_layer_output_det_sha256s);
        }
        payload
    })
}
