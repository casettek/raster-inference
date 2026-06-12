pub use crate::routines::decode_layer_range::raster::auth_source::*;

pub type AuthenticatedGemmaDecodeTransitionSource =
    crate::routines::decode_layer_range::raster::auth_source::AuthenticatedGemmaDecodeLayerRangeSource;
pub type RasterDecodeTransitionSource<'a> =
    crate::routines::decode_layer_range::raster::auth_source::RasterDecodeLayerRangeSource<'a>;
