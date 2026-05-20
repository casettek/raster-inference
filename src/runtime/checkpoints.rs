use std::{fmt, str::FromStr};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PhaseId {
    InputEmbedding,
    TransformerStateTransition,
    OutputDecode,
}

impl PhaseId {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InputEmbedding => "input_embedding",
            Self::TransformerStateTransition => "transformer_state_transition",
            Self::OutputDecode => "output_decode",
        }
    }
}

impl fmt::Display for PhaseId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PhaseId {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "input_embedding" => Ok(Self::InputEmbedding),
            "transformer_state_transition" => Ok(Self::TransformerStateTransition),
            "output_decode" => Ok(Self::OutputDecode),
            _ => anyhow::bail!(
                "unknown phase id `{value}`; expected one of: input_embedding, transformer_state_transition, output_decode"
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RoutineId {
    PromptPrepare,
    InputEmbedding,
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
        "input.embedding" => Some(CheckpointTaxonomy {
            phase_id: PhaseId::InputEmbedding,
            routine_id: RoutineId::InputEmbedding,
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
    fn classify_input_embedding_checkpoint() {
        assert_eq!(
            classify_checkpoint("input.embedding"),
            Some(CheckpointTaxonomy {
                phase_id: PhaseId::InputEmbedding,
                routine_id: RoutineId::InputEmbedding,
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

    #[test]
    fn phase_id_round_trips_through_strings() {
        assert_eq!(
            "transformer_state_transition".parse::<PhaseId>().unwrap(),
            PhaseId::TransformerStateTransition
        );
        assert_eq!(PhaseId::OutputDecode.to_string(), "output_decode");
    }
}
