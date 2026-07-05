pub mod decode_layer_range;
pub mod decode_select_token;
pub mod decode_transition_finalize;
pub mod input_embedding;
pub mod output_finalize;
pub mod prefill_finalize;
pub mod prefill_prepare_aux;
pub mod prefill_range;
pub mod prefill_range_finalize;
pub mod prompt_prepare;

#[cfg(test)]
mod tests {
    const HOSTS: &[(&str, &str)] = &[
        (
            "decode_select_token",
            include_str!("decode_select_token/mod.rs"),
        ),
        (
            "decode_layer_range",
            include_str!("decode_layer_range/mod.rs"),
        ),
        (
            "decode_transition_finalize",
            include_str!("decode_transition_finalize/mod.rs"),
        ),
        ("input_embedding", include_str!("input_embedding/mod.rs")),
        ("output_finalize", include_str!("output_finalize/mod.rs")),
        ("prefill_finalize", include_str!("prefill_finalize/mod.rs")),
        (
            "prefill_prepare_aux",
            include_str!("prefill_prepare_aux/mod.rs"),
        ),
        ("prefill_range", include_str!("prefill_range/mod.rs")),
        ("prompt_prepare", include_str!("prompt_prepare/mod.rs")),
    ];

    /// Raster-core host adapters, one per routine
    /// (`src/routines/<routine>/raster_core/mod.rs`).
    const RASTER_CORE_HOSTS: &[(&str, &str)] = &[
        (
            "decode_select_token",
            include_str!("decode_select_token/raster_core/mod.rs"),
        ),
        (
            "decode_layer_range",
            include_str!("decode_layer_range/raster_core/mod.rs"),
        ),
        (
            "decode_transition_finalize",
            include_str!("decode_transition_finalize/raster_core/mod.rs"),
        ),
        (
            "input_embedding",
            include_str!("input_embedding/raster_core/mod.rs"),
        ),
        (
            "output_finalize",
            include_str!("output_finalize/raster_core/mod.rs"),
        ),
        (
            "prefill_finalize",
            include_str!("prefill_finalize/raster_core/mod.rs"),
        ),
        (
            "prefill_prepare_aux",
            include_str!("prefill_prepare_aux/raster_core/mod.rs"),
        ),
        (
            "prefill_range",
            include_str!("prefill_range/raster_core/mod.rs"),
        ),
        (
            "prefill_range_finalize",
            include_str!("prefill_range_finalize/raster_core/mod.rs"),
        ),
        (
            "prompt_prepare",
            include_str!("prompt_prepare/raster_core/mod.rs"),
        ),
    ];

    /// Routines whose raster-core migration has landed (WS3/WS4). A migrated
    /// host must expose exactly one `pub fn run_raster_core(` entrypoint in
    /// its `raster_core/` module; an unmigrated host must expose none. Flip a
    /// routine into this list in the same change that lands its migration.
    const RASTER_CORE_MIGRATED: &[&str] = &[];

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

    /// The raster-core counterpart of the host-contract guard. Activates per
    /// routine as WS3 migration lands: a migrated raster-core host exposes
    /// exactly one `pub fn run_raster_core(` entrypoint, an unmigrated host
    /// exposes none, and the forbidden-wrapper-pattern list applies to the
    /// new namespace so entrypoint proliferation cannot restart there.
    #[test]
    fn raster_core_hosts_honor_the_migration_contract() {
        for (host, contents) in RASTER_CORE_HOSTS {
            let migrated = RASTER_CORE_MIGRATED.contains(host);
            let expected = usize::from(migrated);
            assert_eq!(
                contents.matches("pub fn run_raster_core(").count(),
                expected,
                "{host} raster_core host should expose exactly {expected} public \
                 run_raster_core entrypoint(s) (migrated: {migrated})"
            );

            if !migrated {
                assert!(
                    !contents.contains("pub fn"),
                    "unmigrated {host} raster_core host must not expose public functions"
                );
            }

            for forbidden in [
                "pub fn run_raster_core_refs",
                "pub fn run_raster_core_with",
                "pub fn run_raster_core_output",
                "pub fn run_raster_core_pipeline",
                "pub fn run_raster_core_state",
                "pub fn run_raster_core_from",
                "pub fn format_native_",
                "pub fn materialize_",
            ] {
                assert!(
                    !contents.contains(forbidden),
                    "{host} raster_core host exposes forbidden wrapper pattern {forbidden}"
                );
            }
        }
    }
}
