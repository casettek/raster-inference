use super::{
    AuthenticatedGemmaPrefillFinalizeSource, GemmaPrefillFinalizeMetadataRequest,
    GemmaPrefillFinalizeNormWeightsRequest, GemmaPrefillFinalizeProjectionKind,
    GemmaPrefillFinalizeProjectionRowRequest, GemmaPrefillFinalizeScalarsRequest,
};
use crate::shared::model::transformer::{
    DetNumMatrix, DetNumTensorSliceSource, Gemma4LogitsProjection, Gemma4ModelProvenance,
    Gemma4TransformerModel, GemmaEmbeddingTensorSource, MatrixF32,
};
use crate::shared::numerics::det_num::{Acc, Wgt};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[test]
fn untied_source_reads_metadata_scalars_norm_and_projection_rows() {
    let model = untied_model();
    let source = AuthenticatedGemmaPrefillFinalizeSource::from_model("finalize", &model)
        .expect("source should build");

    let metadata = crate::auth_read!(&source, GemmaPrefillFinalizeMetadataRequest)
        .expect("metadata should read");
    assert_eq!(source.identifier(), "finalize");
    assert_eq!(metadata.source_id, "finalize");
    assert_eq!(metadata.hidden_width, 2);
    assert_eq!(metadata.projection_rows, 2);
    assert_eq!(metadata.projection_cols, 2);
    assert_eq!(
        metadata.projection_kind,
        GemmaPrefillFinalizeProjectionKind::UntiedLmHead
    );
    assert!(!metadata.has_final_logit_softcapping);

    let norm = crate::auth_read!(&source, GemmaPrefillFinalizeNormWeightsRequest)
        .expect("norm should read");
    assert_eq!(norm, vec![Wgt::from_num(1.0), Wgt::from_num(0.5)]);

    let scalars = crate::auth_read!(&source, GemmaPrefillFinalizeScalarsRequest)
        .expect("scalars should read");
    assert_eq!(scalars.rms_norm_eps, Acc::from_num(0.001));
    assert_eq!(scalars.final_logit_softcapping, None);

    let row = crate::auth_read!(
        &source,
        GemmaPrefillFinalizeProjectionRowRequest { row_idx: 1 },
    )
    .expect("projection row should read");
    assert_eq!(row, vec![Wgt::from_num(0.0), Wgt::from_num(1.0)]);
}

#[test]
fn committed_source_root_reads_metadata_scalars_norm_and_projection_rows() {
    let model = untied_model();
    let source = AuthenticatedGemmaPrefillFinalizeSource::from_model("committed", &model)
        .expect("source should build");
    let committed = source.committed_source().expect("source should commit");
    let root = committed.root().to_string();

    let direct_metadata = crate::auth_read!(&source, GemmaPrefillFinalizeMetadataRequest)
        .expect("direct metadata should read");
    let committed_metadata = crate::auth_read!(root.as_str(), GemmaPrefillFinalizeMetadataRequest)
        .expect("committed metadata should read");
    assert_eq!(committed_metadata, direct_metadata);
    assert_eq!(
        crate::auth_read!(&committed, GemmaPrefillFinalizeNormWeightsRequest)
            .expect("committed norm should read"),
        vec![Wgt::from_num(1.0), Wgt::from_num(0.5)]
    );
    assert_eq!(
        crate::auth_read!(
            root.as_str(),
            GemmaPrefillFinalizeProjectionRowRequest { row_idx: 1 },
        )
        .expect("committed projection row should read"),
        vec![Wgt::from_num(0.0), Wgt::from_num(1.0)]
    );
    assert!(crate::auth_read!(
        "missing-prefill-finalize-root",
        GemmaPrefillFinalizeScalarsRequest
    )
    .is_err());
}

#[test]
fn committed_source_rejects_same_id_with_different_data() {
    let source = AuthenticatedGemmaPrefillFinalizeSource::from_model("conflict", &untied_model())
        .expect("source should build");
    source
        .committed_source_ref()
        .expect("first source should commit");
    let mut model = untied_model();
    model.final_norm_weight_det = Some(vec![Wgt::from_num(2.0), Wgt::from_num(0.5)]);
    let conflicting = AuthenticatedGemmaPrefillFinalizeSource::from_model("conflict", &model)
        .expect("conflicting source should build");

    assert!(conflicting.committed_source_ref().is_err());
}

#[test]
fn tied_source_reads_embedding_rows_as_projection_rows() {
    let (_path, model) = tied_model().expect("tied model fixture should build");
    let source = AuthenticatedGemmaPrefillFinalizeSource::from_model("tied", &model)
        .expect("source should build");

    let metadata = crate::auth_read!(&source, GemmaPrefillFinalizeMetadataRequest)
        .expect("metadata should read");
    assert_eq!(
        metadata.projection_kind,
        GemmaPrefillFinalizeProjectionKind::TiedEmbedding
    );
    assert_eq!(metadata.projection_rows, 2);

    let row = crate::auth_read!(
        &source,
        GemmaPrefillFinalizeProjectionRowRequest { row_idx: 0 },
    )
    .expect("projection row should read");
    assert_eq!(row, vec![Wgt::from_num(0.25), Wgt::from_num(-0.25)]);
}

#[test]
fn tied_source_honors_nonzero_embedding_data_offset() {
    let (path, source) = write_prefixed_det_matrix(
        vec![0xaa; 13],
        vec![
            vec![Wgt::from_num(0.25), Wgt::from_num(-0.25)],
            vec![Wgt::from_num(0.5), Wgt::from_num(1.0)],
        ],
    )
    .expect("prefixed tied fixture should build");
    let embedding_source = GemmaEmbeddingTensorSource::Deterministic {
        source,
        scale: 1.0,
        det_cache: Arc::new(Mutex::new(None)),
    };
    let model = base_model(
        Gemma4LogitsProjection::TiedEmbedding(matrix_f32(2, 2)),
        Some(embedding_source),
    );
    let source = AuthenticatedGemmaPrefillFinalizeSource::from_model("offset", &model)
        .expect("source should build");

    let row = crate::auth_read!(
        &source,
        GemmaPrefillFinalizeProjectionRowRequest { row_idx: 1 },
    )
    .expect("projection row should read");

    assert!(path.exists());
    assert_eq!(row, vec![Wgt::from_num(0.5), Wgt::from_num(1.0)]);
}

#[test]
fn construction_rejects_fp32_model_provenance() {
    let mut model = untied_model();
    model.provenance = Gemma4ModelProvenance::Fp32;

    let error = AuthenticatedGemmaPrefillFinalizeSource::from_model("fp32", &model)
        .expect_err("fp32 model should fail");

    assert!(error.to_string().contains(".detwgt artifact"));
}

#[test]
fn construction_rejects_missing_softcap_scalar() {
    let mut model = untied_model();
    model.final_logit_softcapping = Some(1.0);
    model.final_logit_softcapping_det = None;

    let error = AuthenticatedGemmaPrefillFinalizeSource::from_model("softcap", &model)
        .expect_err("missing softcap scalar should fail");

    assert!(error.to_string().contains("canonical final logit softcap"));
}

#[test]
fn construction_rejects_missing_projection_backing() {
    let mut model = untied_model();
    model.logits_projection = Gemma4LogitsProjection::UntiedLmHead {
        weight: matrix_f32(2, 2),
        det_weight: None,
    };

    let error = AuthenticatedGemmaPrefillFinalizeSource::from_model("projection", &model)
        .expect_err("missing projection should fail");

    assert!(error.to_string().contains("canonical lm_head det_weight"));
}

fn untied_model() -> Gemma4TransformerModel {
    base_model(
        Gemma4LogitsProjection::UntiedLmHead {
            weight: matrix_f32(2, 2),
            det_weight: Some(Arc::new(DetNumMatrix {
                rows: 2,
                cols: 2,
                values: vec![
                    Wgt::from_num(1.0).to_bits(),
                    Wgt::from_num(0.0).to_bits(),
                    Wgt::from_num(0.0).to_bits(),
                    Wgt::from_num(1.0).to_bits(),
                ].into(),
            })),
        },
        None,
    )
}

fn tied_model() -> Result<(PathBuf, Gemma4TransformerModel)> {
    let (path, source) = write_det_matrix(vec![
        vec![Wgt::from_num(0.25), Wgt::from_num(-0.25)],
        vec![Wgt::from_num(0.5), Wgt::from_num(1.0)],
    ])?;
    let embedding_source = GemmaEmbeddingTensorSource::Deterministic {
        source,
        scale: 1.0,
        det_cache: Arc::new(Mutex::new(None)),
    };
    Ok((
        path,
        base_model(
            Gemma4LogitsProjection::TiedEmbedding(matrix_f32(2, 2)),
            Some(embedding_source),
        ),
    ))
}

fn base_model(
    logits_projection: Gemma4LogitsProjection,
    embedding_source: Option<GemmaEmbeddingTensorSource>,
) -> Gemma4TransformerModel {
    Gemma4TransformerModel {
        provenance: Gemma4ModelProvenance::DetNumWgt,
        embedding_table: None,
        embedding_source,
        layers: vec![],
        ple_global: None,
        final_norm_weight: vec![1.0, 0.5],
        final_norm_weight_det: Some(vec![Wgt::from_num(1.0), Wgt::from_num(0.5)]),
        logits_projection,
        final_logit_softcapping: None,
        final_logit_softcapping_det: None,
        rms_norm_eps: 0.001,
        rms_norm_eps_det: Some(Acc::from_num(0.001)),
    }
}

fn matrix_f32(rows: usize, cols: usize) -> MatrixF32 {
    MatrixF32 {
        rows,
        cols,
        values: vec![0.0; rows * cols],
    }
}

fn write_det_matrix(rows: Vec<Vec<Wgt>>) -> Result<(PathBuf, DetNumTensorSliceSource)> {
    write_prefixed_det_matrix(Vec::new(), rows)
}

fn write_prefixed_det_matrix(
    prefix: Vec<u8>,
    rows: Vec<Vec<Wgt>>,
) -> Result<(PathBuf, DetNumTensorSliceSource)> {
    let path = std::env::temp_dir().join(format!(
        "raster-prefill-finalize-source-{}-{}.detwgt",
        std::process::id(),
        crate::trace::sha256_hex(&format!("{:?}{:?}", prefix, rows))
    ));
    let mut bytes = prefix;
    let data_offset = bytes.len();
    for row in &rows {
        for value in row {
            bytes.extend(value.to_bits().to_le_bytes());
        }
    }
    std::fs::write(&path, bytes)
        .with_context(|| format!("failed to write fixture weights {}", path.display()))?;
    let source = det_source(&path, rows.len(), rows[0].len(), data_offset);
    Ok((path, source))
}

fn det_source(
    path: &Path,
    rows: usize,
    cols: usize,
    data_offset: usize,
) -> DetNumTensorSliceSource {
    DetNumTensorSliceSource {
        weights_path: path.to_path_buf(),
        total_rows: rows,
        total_cols: cols,
        data_offset,
        element_width: crate::shared::numerics::det_num::DetWgtElementWidth::I32,
        row_offset: 0,
        row_count: rows,
        col_offset: 0,
        col_count: cols,
    }
}
