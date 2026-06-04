pub use crate::decode_layer_range::raster::auth_source::*;

pub type AuthenticatedGemmaDecodeTransitionSource =
    crate::decode_layer_range::raster::auth_source::AuthenticatedGemmaDecodeLayerRangeSource;
pub type RasterDecodeTransitionSource<'a> =
    crate::decode_layer_range::raster::auth_source::RasterDecodeLayerRangeSource<'a>;
