use std::{collections::HashMap, fmt, str::FromStr};

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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum RoutineId {
    PromptPrepare,
    InputEmbedding,
    PrefillPrepareAux,
    PrefillRange,
    PrefillRangeFinalize,
    PrefillFinalize,
    SelectOutputToken,
    DecodeLayerRange,
    DecodeTransitionFinalize,
    FinalizeOutput,
}

impl RoutineId {
    pub const ALL: [Self; 10] = [
        Self::PromptPrepare,
        Self::InputEmbedding,
        Self::PrefillPrepareAux,
        Self::PrefillRange,
        Self::PrefillRangeFinalize,
        Self::PrefillFinalize,
        Self::SelectOutputToken,
        Self::DecodeLayerRange,
        Self::DecodeTransitionFinalize,
        Self::FinalizeOutput,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::PromptPrepare => "prompt.prepare",
            Self::InputEmbedding => "input.embedding",
            Self::PrefillPrepareAux => "prefill.prepare_aux",
            Self::PrefillRange => "prefill.range",
            Self::PrefillRangeFinalize => "prefill.range_finalize",
            Self::PrefillFinalize => "prefill.finalize",
            Self::SelectOutputToken => "decode.select_token",
            Self::DecodeLayerRange => "decode.layer_range",
            Self::DecodeTransitionFinalize => "decode.transition_finalize",
            Self::FinalizeOutput => "output.finalize",
        }
    }
}

impl FromStr for RoutineId {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .iter()
            .copied()
            .find(|routine_id| routine_id.as_str() == value)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "unknown routine id `{value}`; expected one of: {}",
                    expected_routine_ids()
                )
            })
    }
}

impl fmt::Display for RoutineId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Which raster backend a detour spec targets.
///
/// The sim backend is the in-repo DSL path (`src/dsl/` + per-routine
/// `raster/` modules); the raster-core backend is the real `raster`
/// toolchain path (program crates under `crates/raster-programs/`). The
/// backend never enters committed checkpoint payloads — committed
/// checkpoints represent logical inference state, not execution backend.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DetourBackend {
    #[default]
    Sim,
    RasterCore,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct RasterDetourSpec {
    routine_id: RoutineId,
    occurrence: usize,
    #[serde(default)]
    backend: DetourBackend,
}

impl RasterDetourSpec {
    /// Parses a sim-backend detour spec (`detour --at` semantics).
    pub fn parse(value: &str) -> anyhow::Result<Self> {
        Self::parse_with_backend(value, DetourBackend::Sim)
    }

    /// Parses a raster-core-backend detour spec (`detour --raster-core-at`
    /// semantics). Identical `routine-id[:occurrence]` grammar as [`parse`].
    ///
    /// [`parse`]: Self::parse
    pub fn parse_raster_core(value: &str) -> anyhow::Result<Self> {
        Self::parse_with_backend(value, DetourBackend::RasterCore)
    }

    fn parse_with_backend(value: &str, backend: DetourBackend) -> anyhow::Result<Self> {
        let (routine_id, occurrence) = match value.rsplit_once(':') {
            Some((routine_id, occurrence)) => {
                if routine_id.is_empty() {
                    anyhow::bail!("raster detour routine id must not be empty");
                }
                let occurrence = occurrence.parse::<usize>().map_err(|_| {
                    anyhow::anyhow!("raster detour occurrence must be a positive integer")
                })?;
                (routine_id, occurrence)
            }
            None => (value, 1),
        };

        if routine_id.is_empty() {
            anyhow::bail!("raster detour routine id must not be empty");
        }
        if occurrence == 0 {
            anyhow::bail!("raster detour occurrence must be greater than zero");
        }

        Ok(Self {
            routine_id: routine_id.parse()?,
            occurrence,
            backend,
        })
    }

    pub fn routine_id(self) -> RoutineId {
        self.routine_id
    }

    pub fn occurrence(self) -> usize {
        self.occurrence
    }

    pub fn backend(self) -> DetourBackend {
        self.backend
    }
}

impl FromStr for RasterDetourSpec {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::parse(value)
    }
}

impl fmt::Display for RasterDetourSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.occurrence == 1 {
            f.write_str(self.routine_id.as_str())
        } else {
            write!(f, "{}:{}", self.routine_id.as_str(), self.occurrence)
        }
    }
}

#[derive(Debug, Clone)]
pub struct RasterDetourController {
    spec: Option<RasterDetourSpec>,
    seen: HashMap<RoutineId, usize>,
    matched: bool,
}

impl RasterDetourController {
    pub fn new(spec: Option<RasterDetourSpec>) -> Self {
        Self {
            spec,
            seen: HashMap::new(),
            matched: false,
        }
    }

    pub fn is_active(&self) -> bool {
        self.spec.is_some()
    }

    pub fn selected_spec(&self) -> Option<RasterDetourSpec> {
        self.spec
    }

    pub fn should_detour(&mut self, routine_id: RoutineId) -> bool {
        let Some(spec) = self.spec else {
            return false;
        };
        if spec.routine_id != routine_id {
            return false;
        }

        let seen = self.seen.entry(routine_id).or_insert(0);
        *seen += 1;
        let matched = *seen == spec.occurrence;
        if matched {
            self.matched = true;
        }
        matched
    }

    pub fn ensure_matched_if_active(&self) -> anyhow::Result<()> {
        if let Some(spec) = self.spec {
            if !self.matched {
                anyhow::bail!("selective raster detour target {} was not reached", spec);
            }
        }
        Ok(())
    }

    pub fn reject_if_selected_unsupported(&mut self, routine_id: RoutineId) -> anyhow::Result<()> {
        if self.should_detour(routine_id) {
            let spec = self
                .spec
                .expect("active raster detour controller should carry a spec");
            anyhow::bail!(
                "selective raster detour for {} is not implemented yet",
                spec
            );
        }
        Ok(())
    }
}

fn expected_routine_ids() -> String {
    RoutineId::ALL
        .iter()
        .map(|routine_id| routine_id.as_str())
        .collect::<Vec<_>>()
        .join(", ")
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
        "prefill.range" => Some(CheckpointTaxonomy {
            phase_id: PhaseId::TransformerStateTransition,
            routine_id: RoutineId::PrefillRange,
        }),
        "prefill.range_finalize" => Some(CheckpointTaxonomy {
            phase_id: PhaseId::TransformerStateTransition,
            routine_id: RoutineId::PrefillRangeFinalize,
        }),
        "prefill.finalize" => Some(CheckpointTaxonomy {
            phase_id: PhaseId::TransformerStateTransition,
            routine_id: RoutineId::PrefillFinalize,
        }),
        "decode.select_token" => Some(CheckpointTaxonomy {
            phase_id: PhaseId::OutputDecode,
            routine_id: RoutineId::SelectOutputToken,
        }),
        "decode.layer_range" => Some(CheckpointTaxonomy {
            phase_id: PhaseId::OutputDecode,
            routine_id: RoutineId::DecodeLayerRange,
        }),
        "decode.transition_finalize" => Some(CheckpointTaxonomy {
            phase_id: PhaseId::OutputDecode,
            routine_id: RoutineId::DecodeTransitionFinalize,
        }),
        "output.finalize" => Some(CheckpointTaxonomy {
            phase_id: PhaseId::OutputDecode,
            routine_id: RoutineId::FinalizeOutput,
        }),
        _ if checkpoint_name.starts_with("prefill.layer_token.") => Some(CheckpointTaxonomy {
            phase_id: PhaseId::TransformerStateTransition,
            routine_id: RoutineId::PrefillRangeFinalize,
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        classify_checkpoint, CheckpointTaxonomy, DetourBackend, PhaseId, RasterDetourController,
        RasterDetourSpec, RoutineId,
    };

    #[test]
    fn classify_prefill_layer_token_checkpoint() {
        assert_eq!(
            classify_checkpoint("prefill.layer_token.layer_3.token_9"),
            Some(CheckpointTaxonomy {
                phase_id: PhaseId::TransformerStateTransition,
                routine_id: RoutineId::PrefillRangeFinalize,
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
            classify_checkpoint("decode.layer_range"),
            Some(CheckpointTaxonomy {
                phase_id: PhaseId::OutputDecode,
                routine_id: RoutineId::DecodeLayerRange,
            })
        );
        assert_eq!(
            classify_checkpoint("decode.transition_finalize"),
            Some(CheckpointTaxonomy {
                phase_id: PhaseId::OutputDecode,
                routine_id: RoutineId::DecodeTransitionFinalize,
            })
        );
        assert_eq!(
            classify_checkpoint("decode.layer_token.layer_5.position_17"),
            None
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

    #[test]
    fn routine_id_parses_canonical_ids() {
        assert_eq!(
            "prefill.range".parse::<RoutineId>().unwrap(),
            RoutineId::PrefillRange
        );
        assert_eq!(
            "prefill.range_finalize".parse::<RoutineId>().unwrap(),
            RoutineId::PrefillRangeFinalize
        );
        assert_eq!(
            "decode.layer_range".parse::<RoutineId>().unwrap(),
            RoutineId::DecodeLayerRange
        );
        assert_eq!(
            "decode.transition_finalize".parse::<RoutineId>().unwrap(),
            RoutineId::DecodeTransitionFinalize
        );
    }

    #[test]
    fn routine_id_rejects_checkpoint_sub_ids() {
        let error = "prefill.layer_token.layer_0.token_0"
            .parse::<RoutineId>()
            .expect_err("sub-checkpoints should not parse as routine ids");

        assert!(error.to_string().contains("unknown routine id"));
        assert!(error.to_string().contains("prefill.range_finalize"));

        let error = "decode.layer_token.layer_0.position_0"
            .parse::<RoutineId>()
            .expect_err("decode sub-checkpoints should not parse as routine ids");

        assert!(error.to_string().contains("unknown routine id"));
        assert!(error.to_string().contains("decode.layer_range"));

        let error = "output.finalize.detokenize"
            .parse::<RoutineId>()
            .expect_err("output finalize sub-checkpoints should not parse as routine ids");

        assert!(error.to_string().contains("unknown routine id"));
        assert!(error.to_string().contains("output.finalize"));
    }

    #[test]
    fn raster_detour_spec_parses_default_occurrence() {
        let spec = RasterDetourSpec::parse("input.embedding").expect("spec should parse");

        assert_eq!(spec.routine_id(), RoutineId::InputEmbedding);
        assert_eq!(spec.occurrence(), 1);
        assert_eq!(spec.to_string(), "input.embedding");
    }

    #[test]
    fn raster_detour_spec_parses_occurrence_suffix() {
        let spec = RasterDetourSpec::parse("prefill.range:2").expect("spec should parse");

        assert_eq!(spec.routine_id(), RoutineId::PrefillRange);
        assert_eq!(spec.occurrence(), 2);
        assert_eq!(spec.to_string(), "prefill.range:2");
    }

    #[test]
    fn raster_detour_spec_parse_defaults_to_sim_backend() {
        let spec = RasterDetourSpec::parse("prefill.range:2").expect("spec should parse");
        assert_eq!(spec.backend(), DetourBackend::Sim);
    }

    #[test]
    fn raster_core_detour_spec_parses_every_routine_id() {
        for routine_id in RoutineId::ALL {
            let spec = RasterDetourSpec::parse_raster_core(routine_id.as_str())
                .expect("raster-core spec should parse for every routine id");
            assert_eq!(spec.routine_id(), routine_id);
            assert_eq!(spec.occurrence(), 1);
            assert_eq!(spec.backend(), DetourBackend::RasterCore);
        }
    }

    #[test]
    fn raster_core_detour_spec_parses_occurrence_suffix() {
        let spec =
            RasterDetourSpec::parse_raster_core("prefill.range:3").expect("spec should parse");
        assert_eq!(spec.routine_id(), RoutineId::PrefillRange);
        assert_eq!(spec.occurrence(), 3);
        assert_eq!(spec.backend(), DetourBackend::RasterCore);
        assert_eq!(spec.to_string(), "prefill.range:3");
    }

    #[test]
    fn raster_core_detour_spec_rejects_unknown_routine_and_bad_occurrence() {
        let unknown = RasterDetourSpec::parse_raster_core("not.a_routine")
            .expect_err("unknown routine id should fail");
        assert!(unknown.to_string().contains("unknown routine id"));

        let zero = RasterDetourSpec::parse_raster_core("prefill.range:0")
            .expect_err("zero occurrence should fail");
        assert!(zero.to_string().contains("greater than zero"));

        let non_numeric = RasterDetourSpec::parse_raster_core("prefill.range:abc")
            .expect_err("non-numeric occurrence should fail");
        assert!(non_numeric.to_string().contains("positive integer"));
    }

    #[test]
    fn raster_detour_spec_rejects_invalid_occurrence() {
        let zero =
            RasterDetourSpec::parse("prefill.range:0").expect_err("zero occurrence should fail");
        assert!(zero.to_string().contains("greater than zero"));

        let non_numeric = RasterDetourSpec::parse("prefill.range:abc")
            .expect_err("non-numeric occurrence should fail");
        assert!(non_numeric.to_string().contains("positive integer"));
    }

    #[test]
    fn raster_detour_controller_matches_selected_occurrence_once() {
        let spec = RasterDetourSpec::parse("prefill.range:2").expect("spec should parse");
        let mut controller = RasterDetourController::new(Some(spec));

        assert!(!controller.should_detour(RoutineId::PrefillRange));
        assert!(controller.should_detour(RoutineId::PrefillRange));
        assert!(!controller.should_detour(RoutineId::PrefillRange));
    }

    #[test]
    fn raster_detour_controller_ignores_other_routines() {
        let spec = RasterDetourSpec::parse("prefill.range").expect("spec should parse");
        let mut controller = RasterDetourController::new(Some(spec));

        assert!(!controller.should_detour(RoutineId::InputEmbedding));
        assert!(controller.should_detour(RoutineId::PrefillRange));
    }

    #[test]
    fn raster_detour_controller_reports_unmatched_active_target() {
        let spec = RasterDetourSpec::parse("prefill.range:2").expect("spec should parse");
        let mut controller = RasterDetourController::new(Some(spec));

        assert!(!controller.should_detour(RoutineId::PrefillRange));
        let error = controller
            .ensure_matched_if_active()
            .expect_err("active unmatched detour should fail");

        assert!(error
            .to_string()
            .contains("selective raster detour target prefill.range:2 was not reached"));
    }
}
