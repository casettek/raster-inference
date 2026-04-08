use anyhow::{bail, Result};
use sha2::{Digest, Sha256};

use super::types::{EmbeddedTokenSequence, EmbeddingTable};

pub fn embed_input_tokens(
    token_ids: &[u32],
    embedding_table: &EmbeddingTable,
) -> Result<EmbeddedTokenSequence> {
    if token_ids.is_empty() {
        bail!("phase 2 embedding requires at least one token id");
    }

    if embedding_table.rows.is_empty() {
        bail!("phase 2 embedding requires a non-empty embedding table");
    }

    let hidden_size = embedding_table.rows[0].len();
    if hidden_size == 0 {
        bail!("phase 2 embedding rows must have non-zero width");
    }

    if let Some((row_idx, row)) = embedding_table
        .rows
        .iter()
        .enumerate()
        .find(|(_, row)| row.len() != hidden_size)
    {
        bail!(
            "phase 2 embedding table row {row_idx} has width {}, expected {hidden_size}",
            row.len()
        );
    }

    let mut activations = Vec::with_capacity(token_ids.len());
    for token_id in token_ids {
        let row_idx = usize::try_from(*token_id).expect("u32 should fit into usize");
        let row = embedding_table.rows.get(row_idx).ok_or_else(|| {
            anyhow::anyhow!("token id {token_id} is out of bounds for embedding table")
        })?;
        let mut activation = row.clone();
        if embedding_table.scale != 1.0 {
            for value in &mut activation {
                *value *= embedding_table.scale;
            }
        }
        activations.push(activation);
    }

    let activations_sha256 = build_phase2_commitment(&activations);

    Ok(EmbeddedTokenSequence {
        activations,
        activations_sha256,
    })
}

fn build_phase2_commitment(activations: &[Vec<f32>]) -> String {
    let mut hasher = Sha256::new();
    for row in activations {
        for value in row {
            hasher.update(value.to_le_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::embed_input_tokens;
    use crate::phase2::types::EmbeddingTable;

    #[test]
    fn embed_input_tokens_looks_up_rows_in_order() {
        let embedding_table = EmbeddingTable {
            rows: vec![vec![0.0, 0.5], vec![1.0, 1.5], vec![2.0, 2.5]],
            scale: 1.0,
        };

        let embedded =
            embed_input_tokens(&[2, 0], &embedding_table).expect("embedding should succeed");

        assert_eq!(embedded.activations, vec![vec![2.0, 2.5], vec![0.0, 0.5]]);
        assert_eq!(
            embedded.activations_sha256,
            "ba27ccacfb427e2f44f9a6d875abe24e064893a5ea6a76d8f6c00a29ec10be6f"
        );
    }

    #[test]
    fn embed_input_tokens_rejects_out_of_bounds_token_ids() {
        let embedding_table = EmbeddingTable {
            rows: vec![vec![0.0, 0.5]],
            scale: 1.0,
        };

        let error = embed_input_tokens(&[1], &embedding_table)
            .expect_err("out of bounds token should fail");

        assert!(error.to_string().contains("out of bounds"));
    }

    #[test]
    fn embed_input_tokens_rejects_ragged_embedding_tables() {
        let embedding_table = EmbeddingTable {
            rows: vec![vec![0.0, 0.5], vec![1.0]],
            scale: 1.0,
        };

        let error =
            embed_input_tokens(&[0], &embedding_table).expect_err("ragged table should fail");

        assert!(error.to_string().contains("expected 2"));
    }

    #[test]
    fn embed_input_tokens_applies_embedding_scale() {
        let embedding_table = EmbeddingTable {
            rows: vec![vec![1.0, 2.0]],
            scale: 3.0,
        };

        let embedded =
            embed_input_tokens(&[0], &embedding_table).expect("embedding should succeed");

        assert_eq!(embedded.activations, vec![vec![3.0, 6.0]]);
        assert_eq!(
            embedded.activations_sha256,
            "209a39e983bfd5b06df628da8981625bd58c1342e1543c3641d9873380b9d310"
        );
    }
}
