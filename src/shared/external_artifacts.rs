use anyhow::Result;

use crate::shared::artifact_io::{ArtifactIo, AuthRead};

pub trait ExternalArtifact<Request>: AuthRead<Request> {}

impl<Source, Request> ExternalArtifact<Request> for Source where Source: AuthRead<Request> {}

pub fn read<Source, Request>(
    source: &Source,
    request: Request,
) -> Result<<Source as AuthRead<Request>>::Output>
where
    Source: ExternalArtifact<Request> + ?Sized,
{
    ArtifactIo::auth_read(source, request)
}
