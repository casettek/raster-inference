use std::path::PathBuf;

use raster_inference::shared::model::common::{
    AttentionKind, FfnKind, ModelFamily, NormKind, ProjectionKind, WeightMatrixView,
};
use raster_inference::shared::model::gemma::adapter::GemmaModelBundle;
use raster_inference::shared::model::runtime::LoadedModel;
use raster_inference::shared::numerics::det_num::f32_to_act;
use raster_inference::{
    load_chat_template, load_tokenizer_from_path,
    load_transformer_state_model_from_det_num_wgt_path, ModelSpec,
};

#[test]
fn gemma_bundle_exposes_common_decoder_view() {
    let model_dir = tiny_gemma_dir();
    let tokenizer_path = model_dir.join("tokenizer.json");
    let transformer_model = load_transformer_state_model_from_det_num_wgt_path(&model_dir)
        .expect("tiny-gemma-dev deterministic weights should load");
    let loaded_model = LoadedModel::Gemma(GemmaModelBundle::new(
        ModelSpec {
            model_id: "tiny-gemma-dev".to_string(),
            tokenizer_path: tokenizer_path.clone(),
            chat_template: load_chat_template(model_dir.join("chat_template.jinja"))
                .expect("chat template should load"),
            bos_token: None,
            eos_token: None,
            unk_token: None,
        },
        load_tokenizer_from_path(&tokenizer_path).expect("tokenizer should load"),
        transformer_model,
        None,
    ));

    let spec = loaded_model.architecture_spec();
    assert_eq!(spec.family, ModelFamily::Gemma);
    assert_eq!(spec.architecture_id, "gemma4");
    assert_eq!(spec.num_layers, 4);
    assert_eq!(spec.hidden_size, 4);
    assert_eq!(spec.intermediate_size, Some(8));
    assert_eq!(spec.vocab_size, 280);
    assert_eq!(spec.num_attention_heads, 2);
    assert_eq!(spec.num_kv_heads, 1);
    assert_eq!(spec.head_dim, 2);
    assert_eq!(loaded_model.transformer_layer_count(), spec.num_layers);
    assert_eq!(spec.attention.kind, AttentionKind::Sliding);
    assert_eq!(spec.attention.sliding_window, Some(2));
    assert!(spec.attention.has_mixed_attention);
    assert!(matches!(spec.norm, NormKind::RmsNorm { .. }));

    let view = loaded_model.decoder_view();
    assert_eq!(view.spec, spec);
    assert_eq!(view.layers.len(), spec.num_layers);
    assert!(matches!(
        view.embeddings.weights,
        Some(WeightMatrixView::DetNumSlice(_))
    ));
    assert_eq!(
        view.embeddings.weights.unwrap().shape().rows,
        spec.vocab_size
    );
    assert_eq!(
        view.embeddings.weights.unwrap().shape().cols,
        spec.hidden_size
    );
    assert_eq!(
        view.final_norm
            .det_weights
            .expect("canonical final norm")
            .len(),
        4
    );
    assert_eq!(view.lm_head.kind, ProjectionKind::UntiedLmHead);
    assert!(matches!(
        view.lm_head.weights,
        Some(WeightMatrixView::DetNumMatrix(_))
    ));
    assert_eq!(view.final_logit_softcapping, Some(f32_to_act(7.5)));

    let first = view.layers.first().expect("first layer");
    assert_eq!(first.attention.kind, AttentionKind::Sliding);
    assert_eq!(first.attention.sliding_window, Some(2));
    assert_eq!(first.attention.num_heads, 2);
    assert_eq!(first.attention.num_kv_heads, 1);
    assert_eq!(first.attention.head_dim, 2);
    assert_eq!(first.attention.rope.partial_rotary_dim, 2);
    assert!(matches!(first.ffn, FfnKind::Dense(_)));
    assert!(first.ple.is_some());

    let second = view.layers.get(1).expect("second layer");
    assert_eq!(second.attention.kind, AttentionKind::Full);
    assert_eq!(second.attention.sliding_window, None);
    assert_eq!(second.attention.rope.base, Some(12345.0));
    assert_eq!(second.attention.rope.partial_rotary_dim, 2);
}

fn tiny_gemma_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/tiny-gemma-dev")
}
