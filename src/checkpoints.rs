use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PhaseId {
    InputEmbedding,
    TransformerStateTransition,
    OutputDecode,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RoutineId {
    PromptPrepare,
    PrefillPrepareAux,
    PrefillLayer,
    PrefillFinalize,
    SelectOutputToken,
    DecodeTransition,
    FinalizeOutput,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct CheckpointTaxonomy {
    pub phase_id: PhaseId,
    pub routine_id: RoutineId,
}

pub fn classify_checkpoint(checkpoint_name: &str) -> Option<CheckpointTaxonomy> {
    match checkpoint_name {
        "prompt.prepare" => Some(CheckpointTaxonomy {
            phase_id: PhaseId::InputEmbedding,
            routine_id: RoutineId::PromptPrepare,
        }),
        "prefill.prepare_aux" => Some(CheckpointTaxonomy {
            phase_id: PhaseId::TransformerStateTransition,
            routine_id: RoutineId::PrefillPrepareAux,
        }),
        "prefill.layer" => Some(CheckpointTaxonomy {
            phase_id: PhaseId::TransformerStateTransition,
            routine_id: RoutineId::PrefillLayer,
        }),
        "prefill.finalize" => Some(CheckpointTaxonomy {
            phase_id: PhaseId::TransformerStateTransition,
            routine_id: RoutineId::PrefillFinalize,
        }),
        "decode.select_token" => Some(CheckpointTaxonomy {
            phase_id: PhaseId::OutputDecode,
            routine_id: RoutineId::SelectOutputToken,
        }),
        "decode.finalize" => Some(CheckpointTaxonomy {
            phase_id: PhaseId::TransformerStateTransition,
            routine_id: RoutineId::DecodeTransition,
        }),
        "output.finalize" => Some(CheckpointTaxonomy {
            phase_id: PhaseId::OutputDecode,
            routine_id: RoutineId::FinalizeOutput,
        }),
        _ if checkpoint_name.starts_with("prefill.layer_token.") => Some(CheckpointTaxonomy {
            phase_id: PhaseId::TransformerStateTransition,
            routine_id: RoutineId::PrefillLayer,
        }),
        _ if checkpoint_name.starts_with("decode.layer_token.") => Some(CheckpointTaxonomy {
            phase_id: PhaseId::TransformerStateTransition,
            routine_id: RoutineId::DecodeTransition,
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{classify_checkpoint, CheckpointTaxonomy, PhaseId, RoutineId};

    #[test]
    fn classify_prefill_layer_token_checkpoint() {
        assert_eq!(
            classify_checkpoint("prefill.layer_token.layer_3.token_9"),
            Some(CheckpointTaxonomy {
                phase_id: PhaseId::TransformerStateTransition,
                routine_id: RoutineId::PrefillLayer,
            })
        );
    }

    #[test]
    fn classify_decode_checkpoints_across_protocol_phases() {
        assert_eq!(
            classify_checkpoint("decode.select_token"),
            Some(CheckpointTaxonomy {
                phase_id: PhaseId::OutputDecode,
                routine_id: RoutineId::SelectOutputToken,
            })
        );
        assert_eq!(
            classify_checkpoint("decode.layer_token.layer_5.position_17"),
            Some(CheckpointTaxonomy {
                phase_id: PhaseId::TransformerStateTransition,
                routine_id: RoutineId::DecodeTransition,
            })
        );
    }
}
