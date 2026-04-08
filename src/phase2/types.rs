use serde::{Deserialize, Serialize};

fn default_embedding_scale() -> f32 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingTable {
    pub rows: Vec<Vec<f32>>,
    #[serde(default = "default_embedding_scale")]
    pub scale: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbeddedTokenSequence {
    #[serde(skip_serializing, default)]
    pub activations: Vec<Vec<f32>>,
    pub activations_sha256: String,
}
