//! Routine sequence for `input.embedding`.

use alloc::format;
use alloc::vec::Vec;
use raster::prelude::*;

use crate::embedding::*;
use crate::prompt_tokens::*;
use crate::types::{
    chunk_count, validate_loop_driver, EmbeddingSource, InputEmbeddingActivationDraft,
    InputEmbeddingActivationDraftDraftExt, InputEmbeddingConfig, InputEmbeddingCopyState,
    InputEmbeddingCounts, InputEmbeddingLoopDrivers, InputEmbeddingOutput, PromptTokenSource,
};

/// Prompt token ids + Gemma embedding rows -> embedded prompt activations.
#[sequence]
pub fn embed_input_tokens(
    prompt_token_ids: PromptTokenSource,
    embedding: EmbeddingSource,
    loop_drivers: InputEmbeddingLoopDrivers,
    config: InputEmbeddingConfig,
) -> Result<InputEmbeddingOutput> {
    let counts = call!(
        init_input_embedding_counts,
        prompt_token_ids.clone(),
        embedding.clone(),
        config
    )?;
    let token_chunks = select!(Vec<u32>, loop_drivers.token_ordinals);
    let prompt_token_count = select!(u32, counts.clone().prompt_token_count);
    let hidden_size = select!(u32, counts.clone().hidden_size);
    let vocab_size = select!(u32, counts.clone().vocab_size);
    let tokens_per_tile = select!(u32, counts.clone().tokens_per_tile);

    let embedded = call_recur!(
        tile = embed_one_token_chunk,
        input = token_chunks,
        state = InputEmbeddingCopyState::initial(),
        output = new!(InputEmbeddingActivationDraft),
        args = (
            prompt_token_ids,
            embedding,
            prompt_token_count,
            hidden_size,
            vocab_size,
            tokens_per_tile
        )
    );

    Ok(call!(finalize_input_embedding, embedded, counts)?)
}

#[tile]
pub fn init_input_embedding_counts(
    prompt_token_ids: PromptTokenSource,
    embedding: EmbeddingSource,
    config: InputEmbeddingConfig,
) -> Result<InputEmbeddingCounts> {
    if config.tokens_per_tile == 0 {
        return Err("raster input embedding tokens per tile must be greater than zero".to_string());
    }

    let prompt_token_count = read_prompt_token_count(&prompt_token_ids);
    if prompt_token_count == 0 {
        return Err("transformer embedding requires at least one token id".to_string());
    }
    let prompt_token_ids_sha256 = read_prompt_token_ids_sha256(&prompt_token_ids);
    if prompt_token_ids_sha256 != config.prompt_token_ids_sha256 {
        return Err(format!(
            "raster input embedding prompt-token commitment {} does not match expected {}",
            prompt_token_ids_sha256, config.prompt_token_ids_sha256
        ));
    }

    let metadata = read_embedding_metadata(&embedding);
    if metadata.hidden_size == 0 {
        return Err("Gemma input embedding source must have non-zero hidden size".to_string());
    }
    if metadata.vocab_size == 0 {
        return Err("Gemma input embedding source requires at least one row".to_string());
    }

    Ok(InputEmbeddingCounts {
        prompt_token_count,
        hidden_size: metadata.hidden_size,
        vocab_size: metadata.vocab_size,
        tokens_per_tile: config.tokens_per_tile,
        source_id: metadata.source_id,
        prompt_token_ids_sha256,
        prompt_token_ids_root: config.prompt_token_ids_root,
        embedding_source_root: config.embedding_source_root,
    })
}

#[tile(kind = recur)]
pub fn embed_one_token_chunk(
    input: RecurInput<u32>,
    state: RecurState<InputEmbeddingCopyState>,
    output: RecurOutput<InputEmbeddingActivationDraft>,
    prompt_token_ids: PromptTokenSource,
    embedding: EmbeddingSource,
    prompt_token_count: u32,
    hidden_size: u32,
    vocab_size: u32,
    tokens_per_tile: u32,
) -> RecurControl<(
    RecurState<InputEmbeddingCopyState>,
    RecurOutput<InputEmbeddingActivationDraft>,
)> {
    let chunk_idx = *input.value();
    let mut state = state;
    let mut output = output;
    if let Err(error) = validate_loop_driver(
        "token",
        input.index(),
        input.len(),
        chunk_idx,
        prompt_token_count,
        tokens_per_tile,
    ) {
        output.errors().push(error);
        return RecurControl::Break((state, output));
    }

    let end = chunk_idx
        .saturating_add(1)
        .saturating_mul(tokens_per_tile)
        .min(prompt_token_count);
    for token_idx in state.next_token_idx..end {
        let token_id = read_prompt_token_id(&prompt_token_ids, token_idx);
        if token_id >= vocab_size {
            output.errors().push(format!(
                "Gemma input embedding token id {token_id} is out of range for {vocab_size} rows"
            ));
            return RecurControl::Break((state, output));
        }
        let row = match read_embedding_row(&embedding, token_id) {
            Ok(row) => row,
            Err(error) => {
                output.errors().push(error);
                return RecurControl::Break((state, output));
            }
        };
        if row.len() != hidden_size as usize {
            output.errors().push(format!(
                "input embedding row {token_id} has width {}, expected {hidden_size}",
                row.len()
            ));
            return RecurControl::Break((state, output));
        }
        output.rows().push(row);
    }
    state.next_token_idx = state.next_token_idx.max(end);
    RecurControl::Continue((state, output))
}

#[tile]
pub fn finalize_input_embedding(
    embedded: InputEmbeddingActivationDraft,
    counts: InputEmbeddingCounts,
) -> Result<InputEmbeddingOutput> {
    if let Some(error) = embedded.errors.into_iter().next() {
        return Err(error);
    }
    if embedded.rows.len() != counts.prompt_token_count as usize {
        return Err(format!(
            "raster input embedding loop produced {} rows, expected {}",
            embedded.rows.len(),
            counts.prompt_token_count
        ));
    }
    if chunk_count(counts.prompt_token_count, counts.tokens_per_tile) == 0 {
        return Err("raster input embedding cannot finalize empty token loop".to_string());
    }
    Ok(InputEmbeddingOutput {
        source_id: counts.source_id,
        prompt_token_ids_sha256: counts.prompt_token_ids_sha256,
        prompt_token_ids_root: counts.prompt_token_ids_root,
        embedding_source_root: counts.embedding_source_root,
        prompt_token_count: counts.prompt_token_count,
        hidden_size: counts.hidden_size,
        activation_rows: embedded.rows,
    })
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use alloc::vec;

    use raster::materialize_auth_result;

    use super::*;
    use crate::types::{
        pack_embedding_row_hex, GemmaInputEmbeddingMetadata, GemmaInputEmbeddingTable,
        InputEmbeddingPromptTokenIds,
    };

    fn loop_drivers(token_count: u32, tokens_per_tile: u32) -> InputEmbeddingLoopDrivers {
        InputEmbeddingLoopDrivers {
            token_ordinals: (0..chunk_count(token_count, tokens_per_tile)).collect(),
        }
    }

    fn embed_with_inputs(
        prompt_token_ids: Vec<u32>,
        token_ids_sha256: &str,
        expected_token_ids_sha256: &str,
        embedding_rows: Vec<Vec<i32>>,
        tokens_per_tile: u32,
    ) -> core::result::Result<InputEmbeddingOutput, String> {
        let _guard = raster::__private::SequenceScopeGuard::enter("input_embedding_routine_tests");
        let token_count = prompt_token_ids.len() as u32;
        materialize_auth_result::<InputEmbeddingOutput, _>(
            __raster_sequence_auth_embed_input_tokens(
                internal!(
                    PromptTokenSource,
                    raster::store_internal_value(&PromptTokenSource::internal(
                        raster::store_internal_value(&InputEmbeddingPromptTokenIds {
                            token_count,
                            token_ids: prompt_token_ids,
                            token_ids_sha256: token_ids_sha256.to_string(),
                        })
                        .expect("store prompt ids")
                    ))
                    .expect("store prompt source")
                ),
                internal!(
                    EmbeddingSource,
                    raster::store_internal_value(&EmbeddingSource::internal(
                        raster::store_internal_value(&GemmaInputEmbeddingTable {
                            metadata: GemmaInputEmbeddingMetadata {
                                source_id: "embedding-fixture".to_string(),
                                vocab_size: embedding_rows.len() as u32,
                                hidden_size: embedding_rows.first().map(Vec::len).unwrap_or(0)
                                    as u32,
                                scale_bits: 0,
                            },
                            rows: embedding_rows
                                .iter()
                                .map(|row| pack_embedding_row_hex(row))
                                .collect(),
                        })
                        .expect("store embedding")
                    ))
                    .expect("store embedding source")
                ),
                internal!(
                    InputEmbeddingLoopDrivers,
                    raster::store_internal_value(&loop_drivers(token_count, tokens_per_tile))
                        .expect("store drivers")
                ),
                internal!(
                    InputEmbeddingConfig,
                    raster::store_internal_value(&InputEmbeddingConfig {
                        tokens_per_tile,
                        prompt_token_ids_sha256: expected_token_ids_sha256.to_string(),
                        prompt_token_ids_root: "prompt-root".to_string(),
                        embedding_source_root: "embedding-root".to_string(),
                    })
                    .expect("store config")
                ),
            ),
        )
    }

    #[test]
    fn embeds_prompt_tokens_from_storage() {
        let output = embed_with_inputs(
            vec![1, 0],
            "token-sha",
            "token-sha",
            vec![vec![1, 2], vec![3, 4]],
            1,
        )
        .expect("embed");
        assert_eq!(output.activation_rows, vec![vec![3, 4], vec![1, 2]]);
        assert_eq!(output.prompt_token_count, 2);
        assert_eq!(output.hidden_size, 2);
    }

    #[test]
    fn chunk_size_does_not_change_output() {
        let rows = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
        let single = embed_with_inputs(vec![2, 1, 0], "token-sha", "token-sha", rows.clone(), 1)
            .expect("single");
        let multi =
            embed_with_inputs(vec![2, 1, 0], "token-sha", "token-sha", rows, 8).expect("multi");
        assert_eq!(single, multi);
    }

    #[test]
    fn rejects_commitment_mismatch() {
        let error = embed_with_inputs(vec![0], "actual", "expected", vec![vec![1]], 1)
            .map(|_| ())
            .expect_err("commitment mismatch");
        assert!(error.contains("prompt-token commitment"));
    }

    #[test]
    fn rejects_out_of_range_token_id() {
        let error = embed_with_inputs(vec![9], "token-sha", "token-sha", vec![vec![1]], 1)
            .map(|_| ())
            .expect_err("out of range");
        assert!(error.contains("out of range"));
    }

    #[test]
    fn rejects_zero_chunk_size() {
        let error = embed_with_inputs(vec![0], "token-sha", "token-sha", vec![vec![1]], 0)
            .map(|_| ())
            .expect_err("zero chunk size");
        assert!(error.contains("greater than zero"));
    }
}
