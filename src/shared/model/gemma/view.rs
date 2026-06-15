use crate::shared::model::common::{
    ArchitectureSpec, AttentionKind, AttentionSpec, AttentionView, DecoderLayerView,
    DecoderModelView, DenseFfnView, EmbeddingView, FfnArchitecture, FfnKind, MatrixShape,
    ModelFamily, NormKind, NormView, PleGlobalView, PleLayerView, ProjectionKind, ProjectionView,
    RopeSpec, WeightMatrixView,
};
use crate::shared::model::gemma::transformer::{
    Gemma4AttentionKind, Gemma4LayerMatrixSource, Gemma4LayerWeights, Gemma4LogitsProjection,
    Gemma4PleGlobalWeights, Gemma4PleMatrixSource, Gemma4TransformerModel,
    GemmaEmbeddingTensorSource, GemmaTensorSliceSource,
};

impl Gemma4TransformerModel {
    pub fn decoder_view(&self) -> DecoderModelView<'_> {
        let spec = self.architecture_spec();
        let layers = self
            .layers
            .iter()
            .enumerate()
            .map(|(layer_idx, layer)| decoder_layer_view(layer_idx, layer))
            .collect();

        DecoderModelView {
            spec,
            embeddings: embedding_view(self.embedding_source.as_ref()),
            layers,
            ple: self
                .ple_global
                .as_ref()
                .map(|ple| ple_global_view(ple, self.rms_norm_eps_det)),
            final_norm: NormView {
                weights: &self.final_norm_weight,
                det_weights: self.final_norm_weight_det.as_deref(),
            },
            lm_head: projection_view(&self.logits_projection, self.embedding_source.as_ref()),
            rms_norm_eps: self.rms_norm_eps_det,
            has_final_logit_softcapping: self.final_logit_softcapping.is_some(),
            final_logit_softcapping: if self.final_logit_softcapping.is_some() {
                self.final_logit_softcapping_det
            } else {
                None
            },
        }
    }

    pub fn architecture_spec(&self) -> ArchitectureSpec {
        let first_layer = self.layers.first();
        let hidden_size = first_layer
            .map(|layer| layer.hidden_size)
            .or_else(|| {
                self.embedding_source
                    .as_ref()
                    .map(GemmaEmbeddingTensorSource::hidden_size)
            })
            .unwrap_or(0);
        let intermediate_size = first_layer.map(|layer| matrix_view(&layer.gate_proj).shape().rows);
        let embedding_vocab_size = self
            .embedding_source
            .as_ref()
            .and_then(|embedding| embedding_matrix_view(embedding).map(|view| view.shape().rows))
            .filter(|rows| *rows > 0);
        let vocab_size = embedding_vocab_size
            .or_else(|| projection_vocab_size(&self.logits_projection))
            .unwrap_or(0);
        let mut attention = first_layer.map(attention_spec).unwrap_or(AttentionSpec {
            kind: AttentionKind::Full,
            sliding_window: None,
            cache_sliding_window: None,
            attention_k_eq_v: false,
            has_mixed_attention: false,
        });
        attention.has_mixed_attention = first_layer.is_some_and(|first| {
            self.layers
                .iter()
                .any(|layer| layer.attention_kind != first.attention_kind)
        });
        let rope = first_layer.map(rope_spec).unwrap_or(RopeSpec {
            base: None,
            partial_rotary_dim: 0,
            freq_base_dim: 0,
        });

        ArchitectureSpec {
            family: ModelFamily::Gemma,
            architecture_id: "gemma4".to_string(),
            num_layers: self.layers.len(),
            hidden_size,
            intermediate_size,
            vocab_size,
            num_attention_heads: first_layer.map(|layer| layer.num_heads).unwrap_or(0),
            num_kv_heads: first_layer.map(|layer| layer.num_kv_heads).unwrap_or(0),
            head_dim: first_layer.map(|layer| layer.head_dim).unwrap_or(0),
            norm: NormKind::RmsNorm {
                epsilon: self.rms_norm_eps,
            },
            rope,
            attention,
            ffn: FfnArchitecture::Dense,
        }
    }
}

fn decoder_layer_view(layer_idx: usize, layer: &Gemma4LayerWeights) -> DecoderLayerView<'_> {
    DecoderLayerView {
        layer_idx,
        hidden_size: layer.hidden_size,
        input_norm: norm_view(
            &layer.input_layernorm_weight,
            layer.input_layernorm_weight_det.as_deref(),
        ),
        attention: attention_view(layer),
        post_attention_norm: norm_view(
            &layer.post_attention_layernorm_weight,
            layer.post_attention_layernorm_weight_det.as_deref(),
        ),
        pre_feedforward_norm: norm_view(
            &layer.pre_feedforward_layernorm_weight,
            layer.pre_feedforward_layernorm_weight_det.as_deref(),
        ),
        post_feedforward_norm: norm_view(
            &layer.post_feedforward_layernorm_weight,
            layer.post_feedforward_layernorm_weight_det.as_deref(),
        ),
        ffn: FfnKind::Dense(DenseFfnView {
            gate_proj: matrix_view(&layer.gate_proj),
            up_proj: matrix_view(&layer.up_proj),
            down_proj: matrix_view(&layer.down_proj),
        }),
        ple: layer.ple.as_ref().map(|ple| PleLayerView {
            input_gate: matrix_view(&ple.input_gate),
            layer_projection: matrix_view(&ple.layer_projection),
            post_input_norm: norm_view(
                &ple.post_input_norm_weight,
                ple.post_input_norm_weight_det.as_deref(),
            ),
        }),
        rms_norm_eps: layer.rms_norm_eps_det,
        rope_base: layer.rope_base_det,
        has_layer_scalar: layer.layer_scalar.is_some(),
        layer_scalar: if layer.layer_scalar.is_some() {
            layer.layer_scalar_det
        } else {
            None
        },
    }
}

fn attention_view(layer: &Gemma4LayerWeights) -> AttentionView<'_> {
    AttentionView {
        kind: attention_kind(layer.attention_kind),
        num_heads: layer.num_heads,
        num_kv_heads: layer.num_kv_heads,
        head_dim: layer.head_dim,
        sliding_window: layer.sliding_window,
        cache_sliding_window: layer.cache_sliding_window,
        rope: rope_spec(layer),
        kv_shared_layer_index: layer.kv_shared_layer_index,
        attention_k_eq_v: layer.attention_k_eq_v,
        q_proj: matrix_view(&layer.q_proj),
        k_proj: matrix_view(&layer.k_proj),
        v_proj: layer.v_proj.as_ref().map(matrix_view),
        o_proj: matrix_view(&layer.o_proj),
        q_norm: norm_view(&layer.q_norm_weight, layer.q_norm_weight_det.as_deref()),
        k_norm: norm_view(&layer.k_norm_weight, layer.k_norm_weight_det.as_deref()),
    }
}

fn ple_global_view(
    ple_global: &Gemma4PleGlobalWeights,
    rms_norm_eps: Option<crate::shared::numerics::det_num::Acc>,
) -> PleGlobalView<'_> {
    PleGlobalView {
        token_embeddings: ple_global
            .token_embeddings
            .iter()
            .map(ple_matrix_view)
            .collect(),
        model_projections: ple_global
            .model_projections
            .iter()
            .map(ple_matrix_view)
            .collect(),
        projection_norm: norm_view(
            &ple_global.projection_norm_weight,
            ple_global.projection_norm_weight_det.as_deref(),
        ),
        embedding_scale: ple_global.embedding_scale_det,
        projection_scalar: ple_global.projection_scalar_det,
        input_scale: ple_global.input_scale_det,
        rms_norm_eps,
    }
}

fn embedding_view(source: Option<&GemmaEmbeddingTensorSource>) -> EmbeddingView<'_> {
    EmbeddingView {
        weights: source.and_then(embedding_matrix_view),
        scale: source.map(GemmaEmbeddingTensorSource::scale).unwrap_or(1.0),
    }
}

fn projection_view<'a>(
    projection: &'a Gemma4LogitsProjection,
    embedding_source: Option<&'a GemmaEmbeddingTensorSource>,
) -> ProjectionView<'a> {
    match projection {
        Gemma4LogitsProjection::UntiedLmHead { weight, det_weight } => ProjectionView {
            kind: ProjectionKind::UntiedLmHead,
            weights: det_weight
                .as_ref()
                .map(WeightMatrixView::DetNumMatrix)
                .or(Some(WeightMatrixView::Materialized(weight))),
        },
        Gemma4LogitsProjection::TiedEmbedding(weight) => ProjectionView {
            kind: ProjectionKind::TiedEmbedding,
            weights: embedding_source
                .and_then(embedding_matrix_view)
                .or(Some(WeightMatrixView::Materialized(weight))),
        },
    }
}

fn projection_vocab_size(projection: &Gemma4LogitsProjection) -> Option<usize> {
    match projection {
        Gemma4LogitsProjection::UntiedLmHead { weight, det_weight } => det_weight
            .as_ref()
            .map(|matrix| matrix.rows)
            .or(Some(weight.rows)),
        Gemma4LogitsProjection::TiedEmbedding(weight) => Some(weight.rows),
    }
}

fn attention_spec(layer: &Gemma4LayerWeights) -> AttentionSpec {
    AttentionSpec {
        kind: attention_kind(layer.attention_kind),
        sliding_window: layer.sliding_window,
        cache_sliding_window: layer.cache_sliding_window,
        attention_k_eq_v: layer.attention_k_eq_v,
        has_mixed_attention: false,
    }
}

fn attention_kind(kind: Gemma4AttentionKind) -> AttentionKind {
    match kind {
        Gemma4AttentionKind::Full => AttentionKind::Full,
        Gemma4AttentionKind::Sliding => AttentionKind::Sliding,
    }
}

fn rope_spec(layer: &Gemma4LayerWeights) -> RopeSpec {
    RopeSpec {
        base: Some(layer.rope_base),
        partial_rotary_dim: layer.partial_rotary_dim,
        freq_base_dim: layer.rope_freq_base_dim,
    }
}

fn norm_view<'a>(
    weights: &'a [f32],
    det_weights: Option<&'a [crate::shared::numerics::det_num::Wgt]>,
) -> NormView<'a> {
    NormView {
        weights,
        det_weights,
    }
}

fn matrix_view(source: &Gemma4LayerMatrixSource) -> WeightMatrixView<'_> {
    match source {
        Gemma4LayerMatrixSource::Materialized(matrix) => {
            WeightMatrixView::Materialized(matrix.as_ref())
        }
        Gemma4LayerMatrixSource::Lazy { source, .. } => WeightMatrixView::ShapeOnly(shape(source)),
        Gemma4LayerMatrixSource::DetNumLazy { source, .. } => WeightMatrixView::DetNumSlice(source),
    }
}

fn ple_matrix_view(source: &Gemma4PleMatrixSource) -> WeightMatrixView<'_> {
    match source {
        Gemma4PleMatrixSource::Materialized(matrix) => WeightMatrixView::Materialized(matrix),
        Gemma4PleMatrixSource::Lazy(source) => WeightMatrixView::ShapeOnly(shape(source)),
        Gemma4PleMatrixSource::DetNumLazy(source) => WeightMatrixView::DetNumSlice(source),
    }
}

fn embedding_matrix_view(source: &GemmaEmbeddingTensorSource) -> Option<WeightMatrixView<'_>> {
    Some(match source {
        GemmaEmbeddingTensorSource::Single { hidden_size, .. }
        | GemmaEmbeddingTensorSource::Indexed { hidden_size, .. } => {
            WeightMatrixView::ShapeOnly(MatrixShape {
                rows: 0,
                cols: *hidden_size,
            })
        }
        GemmaEmbeddingTensorSource::Deterministic { source, .. } => {
            WeightMatrixView::DetNumSlice(source)
        }
    })
}

fn shape(source: &GemmaTensorSliceSource) -> MatrixShape {
    MatrixShape {
        rows: source.row_count,
        cols: source.col_count,
    }
}
