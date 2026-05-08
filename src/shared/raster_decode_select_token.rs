use anyhow::{anyhow, bail, Result};

use crate::shared::artifact_io::AuthRead;
use crate::shared::transformer::InternalLogits;

#[derive(Debug, Clone)]
pub struct AuthenticatedDecodeSelectLogitsSource {
    identifier: String,
    metadata: DecodeSelectLogitsMetadata,
    logit_bits: Vec<i32>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct DecodeSelectLogitsMetadata {
    pub source_id: String,
    pub logit_count: usize,
    pub det_logits_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeSelectLogitsMetadataRequest;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeSelectLogitRequest {
    pub token_idx: usize,
}

impl AuthenticatedDecodeSelectLogitsSource {
    pub(crate) fn from_internal_logits(
        identifier: impl Into<String>,
        logits: &InternalLogits,
    ) -> Result<Self> {
        let identifier = validate_identifier(identifier.into())?;
        let det_logits = logits.det_values().ok_or_else(|| {
            anyhow!("raster decode select token requires canonical deterministic logits")
        })?;
        if det_logits.is_empty() {
            bail!("raster decode select token requires at least one canonical logit");
        }

        let det_logits_sha256 =
            crate::shared::transformer_kernels::build_det_vector_commitment(det_logits);
        let logit_bits = det_logits
            .iter()
            .map(|logit| logit.to_bits())
            .collect::<Vec<_>>();

        Ok(Self {
            metadata: DecodeSelectLogitsMetadata {
                source_id: identifier.clone(),
                logit_count: logit_bits.len(),
                det_logits_sha256,
            },
            identifier,
            logit_bits,
        })
    }

    pub fn identifier(&self) -> &str {
        &self.identifier
    }
}

impl AuthRead<DecodeSelectLogitsMetadataRequest> for AuthenticatedDecodeSelectLogitsSource {
    type Output = DecodeSelectLogitsMetadata;

    fn auth_read(&self, _request: DecodeSelectLogitsMetadataRequest) -> Result<Self::Output> {
        Ok(self.metadata.clone())
    }
}

impl AuthRead<DecodeSelectLogitRequest> for AuthenticatedDecodeSelectLogitsSource {
    type Output = i32;

    fn auth_read(&self, request: DecodeSelectLogitRequest) -> Result<Self::Output> {
        self.logit_bits
            .get(request.token_idx)
            .copied()
            .ok_or_else(|| {
                anyhow!(
                    "raster decode select logit {} is out of range for {} logits",
                    request.token_idx,
                    self.logit_bits.len()
                )
            })
    }
}

fn validate_identifier(identifier: String) -> Result<String> {
    if identifier.is_empty() {
        bail!("raster decode select logits source identifier must not be empty");
    }
    Ok(identifier)
}

#[cfg(test)]
mod tests {
    use super::{
        AuthenticatedDecodeSelectLogitsSource, DecodeSelectLogitRequest,
        DecodeSelectLogitsMetadataRequest,
    };
    use crate::shared::artifact_io::AuthRead;
    use crate::shared::{det_num::Act, transformer::InternalLogits};

    #[test]
    fn source_metadata_records_count_and_commitment() {
        let logits = InternalLogits::from_det_values(vec![Act::from_bits(3), Act::from_bits(5)]);
        let source =
            AuthenticatedDecodeSelectLogitsSource::from_internal_logits("decode-0", &logits)
                .expect("source should build");

        let metadata = source
            .auth_read(DecodeSelectLogitsMetadataRequest)
            .expect("metadata should read");

        assert_eq!(metadata.source_id, "decode-0");
        assert_eq!(metadata.logit_count, 2);
        assert_eq!(
            metadata.det_logits_sha256,
            crate::shared::transformer_kernels::build_det_vector_commitment(
                logits.det_values().expect("canonical logits")
            )
        );
    }

    #[test]
    fn source_reads_canonical_logit_bits() {
        let logits = InternalLogits::from_det_values(vec![Act::from_bits(-2), Act::from_bits(9)]);
        let source =
            AuthenticatedDecodeSelectLogitsSource::from_internal_logits("decode-0", &logits)
                .expect("source should build");

        assert_eq!(
            source
                .auth_read(DecodeSelectLogitRequest { token_idx: 0 })
                .expect("first logit should read"),
            -2
        );
        assert_eq!(
            source
                .auth_read(DecodeSelectLogitRequest { token_idx: 1 })
                .expect("second logit should read"),
            9
        );
    }

    #[test]
    fn source_rejects_f32_only_logits() {
        let logits = InternalLogits::from_values(vec![1.0, 2.0]);

        let error =
            AuthenticatedDecodeSelectLogitsSource::from_internal_logits("decode-0", &logits)
                .expect_err("f32-only logits should fail");

        assert!(error.to_string().contains("canonical deterministic logits"));
    }

    #[test]
    fn source_rejects_empty_canonical_logits() {
        let logits = InternalLogits::from_det_values(vec![]);

        let error =
            AuthenticatedDecodeSelectLogitsSource::from_internal_logits("decode-0", &logits)
                .expect_err("empty logits should fail");

        assert!(error.to_string().contains("at least one canonical logit"));
    }

    #[test]
    fn source_rejects_out_of_range_reads() {
        let logits = InternalLogits::from_det_values(vec![Act::from_bits(1)]);
        let source =
            AuthenticatedDecodeSelectLogitsSource::from_internal_logits("decode-0", &logits)
                .expect("source should build");

        let error = source
            .auth_read(DecodeSelectLogitRequest { token_idx: 1 })
            .expect_err("out-of-range read should fail");

        assert!(error.to_string().contains("out of range"));
    }
}
