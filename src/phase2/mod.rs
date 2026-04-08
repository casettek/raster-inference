use anyhow::Result;

pub mod tiles;
pub mod types;

pub use tiles::embed_input_tokens;
pub use types::{EmbeddedTokenSequence, EmbeddingTable};

pub fn run_phase2(
    phase1_state: &crate::phase1::Phase1State,
    embedding_table: &EmbeddingTable,
) -> Result<EmbeddedTokenSequence> {
    embed_input_tokens(&phase1_state.prompt_token_ids, embedding_table)
}

#[cfg(test)]
mod tests {
    use super::{run_phase2, EmbeddingTable};
    use crate::phase1::Phase1State;

    #[test]
    fn run_phase2_embeds_phase1_prompt_tokens_in_order() {
        let phase1_state = Phase1State {
            prompt_token_ids: vec![1, 0],
            prompt_token_ids_sha256: "unused-for-phase2".to_string(),
        };
        let embedding_table = EmbeddingTable {
            rows: vec![vec![0.0, 0.5], vec![1.0, 1.5]],
            scale: 1.0,
        };

        let phase2_state =
            run_phase2(&phase1_state, &embedding_table).expect("phase 2 should succeed");

        assert_eq!(phase2_state.activations, vec![vec![1.0, 1.5], vec![0.0, 0.5]]);
    }
}
