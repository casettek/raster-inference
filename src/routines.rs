pub mod decode_select_token;
pub mod decode_transition;
pub mod input_embedding;
pub mod output_finalize;
pub mod prefill_finalize;
pub mod prefill_layer;
pub mod prefill_prepare_aux;
pub mod prompt_prepare;

#[cfg(test)]
mod tests {
    const HOSTS: &[(&str, &str)] = &[
        (
            "decode_select_token",
            include_str!("routines/decode_select_token.rs"),
        ),
        (
            "decode_transition",
            include_str!("routines/decode_transition.rs"),
        ),
        (
            "input_embedding",
            include_str!("routines/input_embedding.rs"),
        ),
        (
            "output_finalize",
            include_str!("routines/output_finalize.rs"),
        ),
        (
            "prefill_finalize",
            include_str!("routines/prefill_finalize.rs"),
        ),
        ("prefill_layer", include_str!("routines/prefill_layer.rs")),
        (
            "prefill_prepare_aux",
            include_str!("routines/prefill_prepare_aux.rs"),
        ),
        ("prompt_prepare", include_str!("routines/prompt_prepare.rs")),
    ];

    #[test]
    fn raster_hosts_expose_one_public_run_raster_entrypoint() {
        for (host, contents) in HOSTS {
            assert_eq!(
                contents.matches("pub fn run_raster(").count(),
                1,
                "{host} should expose exactly one public run_raster entrypoint"
            );

            for forbidden in [
                "pub fn run_raster_refs",
                "pub fn run_raster_with",
                "pub fn run_raster_output",
                "pub fn run_raster_pipeline",
                "pub fn run_raster_state",
                "pub fn run_raster_from",
                "pub fn format_native_prompt_as_raster_checkpoint(",
                "pub fn format_native_input_embedding_as_raster_checkpoint(",
                "pub fn format_native_prefill_prepare_aux_as_raster_checkpoint(",
                "pub fn materialize_input_embedding_refs(",
                "pub fn materialize_prefill_ple_input_refs(",
                "pub fn materialize_decode_state_from_raster_state(",
                "pub fn materialize_raster_output_refs(",
                "pub fn materialize_prefill_layer_output_refs_from_roots(",
            ] {
                assert!(
                    !contents.contains(forbidden),
                    "{host} still exposes forbidden raster wrapper pattern {forbidden}"
                );
            }
        }
    }
}
