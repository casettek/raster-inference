pub use crate::routines::decode_layer_range::raster::auth_source::*;

pub type AuthenticatedDecoderDecodeTransitionSource =
    crate::routines::decode_layer_range::raster::auth_source::AuthenticatedDecoderDecodeLayerRangeSource;
pub type RasterDecodeTransitionSource<'a> =
    crate::routines::decode_layer_range::raster::auth_source::RasterDecodeLayerRangeSource<'a>;
