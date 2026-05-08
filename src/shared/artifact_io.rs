use anyhow::Result;

use crate::shared::raster_artifact_store::{
    self, RasterArtifactBuilderRef, RasterArtifactId, RasterArtifactMetadata, RasterArtifactRead,
    RasterArtifactRef,
};

pub trait AuthRead<Request> {
    type Output;

    fn auth_read(&self, request: Request) -> Result<Self::Output>;
}

#[derive(Debug, Clone, Copy)]
pub struct ArtifactIo;

impl ArtifactIo {
    pub fn auth_read<Source, Request>(
        source: &Source,
        request: Request,
    ) -> Result<<Source as AuthRead<Request>>::Output>
    where
        Source: AuthRead<Request> + ?Sized,
    {
        source.auth_read(request)
    }

    pub fn reset_store() {
        raster_artifact_store::reset_artifact_store();
    }

    pub fn start_builder(
        id: RasterArtifactId,
        metadata: RasterArtifactMetadata,
    ) -> Result<RasterArtifactBuilderRef> {
        raster_artifact_store::start_builder(id, metadata)
    }

    pub fn insert_artifact(
        id: RasterArtifactId,
        metadata: RasterArtifactMetadata,
        leaves: Vec<Vec<u8>>,
    ) -> Result<RasterArtifactRef> {
        raster_artifact_store::insert_artifact(id, metadata, leaves)
    }

    pub fn append_leaf(
        builder_ref: &mut RasterArtifactBuilderRef,
        leaf_idx: usize,
        payload: Vec<u8>,
    ) -> Result<()> {
        raster_artifact_store::append_leaf(builder_ref, leaf_idx, payload)
    }

    pub fn append_leaf_by_builder_root(
        builder_root: &str,
        leaf_idx: usize,
        payload: Vec<u8>,
    ) -> Result<String> {
        raster_artifact_store::append_leaf_by_builder_root(builder_root, leaf_idx, payload)
    }

    pub fn finalize_builder_by_root(builder_root: &str) -> Result<RasterArtifactRef> {
        raster_artifact_store::finalize_builder_by_root(builder_root)
    }

    pub fn finalize_builder(builder_ref: RasterArtifactBuilderRef) -> Result<RasterArtifactRef> {
        raster_artifact_store::finalize_builder(builder_ref)
    }

    pub fn read_leaf(
        artifact_ref: &RasterArtifactRef,
        leaf_idx: usize,
    ) -> Result<RasterArtifactRead> {
        raster_artifact_store::read_leaf(artifact_ref, leaf_idx)
    }

    pub fn artifact_ref_for_root(root: &str) -> Result<RasterArtifactRef> {
        raster_artifact_store::artifact_ref_for_root(root)
    }

    pub fn verify_artifact_read(
        artifact_ref: &RasterArtifactRef,
        read: &RasterArtifactRead,
    ) -> Result<()> {
        raster_artifact_store::verify_artifact_read(artifact_ref, read)
    }
}

pub fn auth_read<Source, Request>(
    source: &Source,
    request: Request,
) -> Result<<Source as AuthRead<Request>>::Output>
where
    Source: AuthRead<Request> + ?Sized,
{
    ArtifactIo::auth_read(source, request)
}
