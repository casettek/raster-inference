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
            include_str!("decode_select_token/mod.rs"),
        ),
        (
            "decode_transition",
            include_str!("decode_transition/mod.rs"),
        ),
        ("input_embedding", include_str!("input_embedding/mod.rs")),
        ("output_finalize", include_str!("output_finalize/mod.rs")),
        ("prefill_finalize", include_str!("prefill_finalize/mod.rs")),
        ("prefill_layer", include_str!("prefill_layer/mod.rs")),
        (
            "prefill_prepare_aux",
            include_str!("prefill_prepare_aux/mod.rs"),
        ),
        ("prompt_prepare", include_str!("prompt_prepare/mod.rs")),
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
