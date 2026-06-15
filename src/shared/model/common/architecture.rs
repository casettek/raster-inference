#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    Gemma,
    Qwen,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArchitectureSpec {
    pub family: ModelFamily,
    pub architecture_id: String,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: Option<usize>,
    pub vocab_size: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub norm: NormKind,
    pub rope: RopeSpec,
    pub attention: AttentionSpec,
    pub ffn: FfnArchitecture,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NormKind {
    RmsNorm { epsilon: f32 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeSpec {
    pub base: Option<f32>,
    pub partial_rotary_dim: usize,
    pub freq_base_dim: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionKind {
    Full,
    Sliding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionSpec {
    pub kind: AttentionKind,
    pub sliding_window: Option<usize>,
    pub cache_sliding_window: Option<usize>,
    pub attention_k_eq_v: bool,
    pub has_mixed_attention: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnArchitecture {
    Dense,
    Moe {
        num_experts: usize,
        top_k: usize,
        has_shared_expert: bool,
    },
}
