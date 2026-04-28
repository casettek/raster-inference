use anyhow::{anyhow, bail, Result};

use crate::auth_read;
use crate::raster_authoring::AuthRead;
use crate::shared::det_num::{
    act_to_f32, add_sat, mac_bits, requantize, rms_norm as det_rms_norm, scale_act, Acc, Act, Wgt,
};
use crate::shared::raster_prefill_ple::GemmaPleModelProjectionRowRequest;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterActivationRow {
    act_bits: Vec<i32>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RasterActivationSequence {
    rows: Vec<RasterActivationRow>,
}

impl RasterActivationRow {
    pub fn from_acts(acts: Vec<Act>) -> Self {
        Self {
            act_bits: acts.into_iter().map(|value| value.to_bits()).collect(),
        }
    }

    pub fn from_act_bits(act_bits: Vec<i32>) -> Self {
        Self { act_bits }
    }

    pub fn acts(&self) -> Vec<Act> {
        self.act_bits.iter().copied().map(Act::from_bits).collect()
    }

    pub fn act_bits(&self) -> &[i32] {
        &self.act_bits
    }

    pub fn width(&self) -> usize {
        self.act_bits.len()
    }

    pub fn to_f32_values(&self) -> Vec<f32> {
        self.acts().into_iter().map(act_to_f32).collect()
    }
}

impl RasterActivationSequence {
    pub fn from_rows(rows: Vec<RasterActivationRow>) -> Self {
        Self { rows }
    }

    pub fn from_acts(rows: Vec<Vec<Act>>) -> Self {
        Self {
            rows: rows
                .into_iter()
                .map(RasterActivationRow::from_acts)
                .collect(),
        }
    }

    pub fn from_act_bits(rows: Vec<Vec<i32>>) -> Self {
        Self {
            rows: rows
                .into_iter()
                .map(RasterActivationRow::from_act_bits)
                .collect(),
        }
    }

    pub fn rows(&self) -> &[RasterActivationRow] {
        &self.rows
    }

    pub fn into_rows(self) -> Vec<RasterActivationRow> {
        self.rows
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn width(&self) -> Result<usize> {
        sequence_width(self)
    }

    pub fn to_f32_values(&self) -> Vec<Vec<f32>> {
        self.rows
            .iter()
            .map(RasterActivationRow::to_f32_values)
            .collect()
    }
}

pub fn scale_sequence(
    input: &RasterActivationSequence,
    scalar: Option<Act>,
) -> Result<RasterActivationSequence> {
    let scalar = scalar
        .ok_or_else(|| anyhow!("deterministic sequence scaling requires canonical Act scalar"))?;
    validate_non_empty_sequence(input, "deterministic sequence scaling")?;

    Ok(RasterActivationSequence::from_rows(
        input
            .rows()
            .iter()
            .map(|row| {
                RasterActivationRow::from_acts(
                    row.acts()
                        .into_iter()
                        .map(|value| scale_act(value, scalar))
                        .collect(),
                )
            })
            .collect(),
    ))
}

pub fn add_sequences(
    lhs: &RasterActivationSequence,
    rhs: &RasterActivationSequence,
) -> Result<RasterActivationSequence> {
    let width = sequence_width(lhs)?;
    validate_sequence_width(rhs, width, "right sequence")?;
    if lhs.len() != rhs.len() {
        bail!("sequence length mismatch: {} vs {}", lhs.len(), rhs.len());
    }

    Ok(RasterActivationSequence::from_rows(
        lhs.rows()
            .iter()
            .zip(rhs.rows())
            .map(|(lhs_row, rhs_row)| {
                RasterActivationRow::from_acts(
                    lhs_row
                        .acts()
                        .into_iter()
                        .zip(rhs_row.acts())
                        .map(|(lhs_value, rhs_value)| add_sat(lhs_value, rhs_value))
                        .collect(),
                )
            })
            .collect(),
    ))
}

pub fn project_sequence(
    input: &RasterActivationSequence,
    projection_rows: &[Vec<Wgt>],
) -> Result<RasterActivationSequence> {
    validate_projection_rows(projection_rows)?;
    validate_sequence_width(
        input,
        projection_rows[0].len(),
        "deterministic linear input",
    )?;

    Ok(RasterActivationSequence::from_rows(
        input
            .rows()
            .iter()
            .map(|input_row| project_row(input_row, projection_rows))
            .collect::<Result<Vec<_>>>()?,
    ))
}

pub fn project_sequence_with_source<S>(
    input: &RasterActivationSequence,
    source: &S,
    layer_idx: usize,
    projection_rows: usize,
) -> Result<RasterActivationSequence>
where
    S: AuthRead<GemmaPleModelProjectionRowRequest, Output = Vec<Wgt>>,
{
    if projection_rows == 0 {
        bail!("deterministic linear projection requires at least one projection row");
    }

    let mut rows = Vec::with_capacity(projection_rows);
    for row_idx in 0..projection_rows {
        rows.push(auth_read!(
            source,
            GemmaPleModelProjectionRowRequest { layer_idx, row_idx }
        )?);
    }
    project_sequence(input, &rows)
}

pub fn rms_norm_sequence(
    input: &RasterActivationSequence,
    norm_weights: Option<&[Wgt]>,
    eps: Option<Acc>,
) -> Result<RasterActivationSequence> {
    let norm_weights = norm_weights
        .ok_or_else(|| anyhow!("deterministic RMSNorm requires canonical norm weights"))?;
    let eps = eps.ok_or_else(|| anyhow!("deterministic RMSNorm requires canonical Acc epsilon"))?;
    validate_sequence_width(input, norm_weights.len(), "deterministic RMSNorm input")?;

    Ok(RasterActivationSequence::from_rows(
        input
            .rows()
            .iter()
            .map(|row| RasterActivationRow::from_acts(det_rms_norm(&row.acts(), norm_weights, eps)))
            .collect(),
    ))
}

fn project_row(
    input: &RasterActivationRow,
    projection_rows: &[Vec<Wgt>],
) -> Result<RasterActivationRow> {
    let input_acts = input.acts();
    let width = projection_rows[0].len();
    if input_acts.len() != width {
        bail!(
            "deterministic linear input width mismatch: {} vs {}",
            input_acts.len(),
            width
        );
    }

    let mut output = Vec::with_capacity(projection_rows.len());
    for row in projection_rows {
        let mut acc_bits = 0_i64;
        for (act, weight) in input_acts.iter().zip(row) {
            acc_bits = mac_bits(acc_bits, act.to_bits(), weight.to_bits());
        }
        output.push(requantize(Acc::from_bits(acc_bits)));
    }
    Ok(RasterActivationRow::from_acts(output))
}

fn validate_non_empty_sequence(input: &RasterActivationSequence, label: &str) -> Result<()> {
    if input.is_empty() {
        bail!("{label} requires at least one activation row");
    }
    Ok(())
}

fn sequence_width(input: &RasterActivationSequence) -> Result<usize> {
    validate_non_empty_sequence(input, "deterministic sequence operation")?;
    let width = input.rows()[0].width();
    if width == 0 {
        bail!("deterministic sequence operation requires non-empty activation rows");
    }
    validate_sequence_width(input, width, "activation sequence")?;
    Ok(width)
}

fn validate_sequence_width(
    input: &RasterActivationSequence,
    expected_width: usize,
    label: &str,
) -> Result<()> {
    validate_non_empty_sequence(input, label)?;
    if expected_width == 0 {
        bail!("{label} requires a non-zero width");
    }
    if let Some((row_idx, row)) = input
        .rows()
        .iter()
        .enumerate()
        .find(|(_, row)| row.width() != expected_width)
    {
        bail!(
            "{label} row {row_idx} has width {}, expected {expected_width}",
            row.width()
        );
    }
    Ok(())
}

fn validate_projection_rows(projection_rows: &[Vec<Wgt>]) -> Result<()> {
    let Some(first_row) = projection_rows.first() else {
        bail!("deterministic linear projection requires at least one projection row");
    };
    let width = first_row.len();
    if width == 0 {
        bail!("deterministic linear projection rows must have non-zero width");
    }
    if let Some((row_idx, row)) = projection_rows
        .iter()
        .enumerate()
        .find(|(_, row)| row.len() != width)
    {
        bail!(
            "deterministic linear projection row {row_idx} has width {}, expected {width}",
            row.len()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        add_sequences, project_sequence, project_sequence_with_source, rms_norm_sequence,
        scale_sequence, RasterActivationSequence,
    };
    use crate::raster_authoring::AuthRead;
    use crate::shared::det_num::{add_sat, rms_norm, scale_act, Acc, Act, Wgt};
    use crate::shared::raster_prefill_ple::GemmaPleModelProjectionRowRequest;
    use anyhow::{anyhow, Result};
    use std::collections::HashMap;

    #[test]
    fn scaling_sequence_uses_det_num_act_scalar_semantics() {
        let input = RasterActivationSequence::from_acts(vec![
            vec![Act::from_num(1.5), Act::from_num(-2.0)],
            vec![Act::from_num(0.25), Act::from_num(4.0)],
        ]);

        let scaled = scale_sequence(&input, Some(Act::from_num(0.5))).expect("scale should work");

        assert_eq!(
            scaled_bits(&scaled),
            vec![
                vec![
                    scale_act(Act::from_num(1.5), Act::from_num(0.5)).to_bits(),
                    scale_act(Act::from_num(-2.0), Act::from_num(0.5)).to_bits(),
                ],
                vec![
                    scale_act(Act::from_num(0.25), Act::from_num(0.5)).to_bits(),
                    scale_act(Act::from_num(4.0), Act::from_num(0.5)).to_bits(),
                ],
            ]
        );
    }

    #[test]
    fn adding_sequences_preserves_shape_and_saturates() {
        let lhs = RasterActivationSequence::from_act_bits(vec![
            vec![i32::MAX, Act::from_num(0.25).to_bits()],
            vec![Act::from_num(-0.5).to_bits(), Act::from_num(1.0).to_bits()],
        ]);
        let rhs = RasterActivationSequence::from_act_bits(vec![
            vec![1, Act::from_num(0.25).to_bits()],
            vec![Act::from_num(1.0).to_bits(), Act::from_num(-0.25).to_bits()],
        ]);

        let added = add_sequences(&lhs, &rhs).expect("add should work");

        assert_eq!(
            scaled_bits(&added),
            vec![
                vec![
                    i32::MAX,
                    add_sat(Act::from_num(0.25), Act::from_num(0.25)).to_bits(),
                ],
                vec![
                    add_sat(Act::from_num(-0.5), Act::from_num(1.0)).to_bits(),
                    add_sat(Act::from_num(1.0), Act::from_num(-0.25)).to_bits(),
                ],
            ]
        );
        assert_eq!(added.len(), 2);
        assert_eq!(added.width().expect("width"), 2);
    }

    #[test]
    fn projection_sequence_matches_hand_computed_det_num_fixture() {
        let input = RasterActivationSequence::from_acts(vec![
            vec![Act::from_num(1.0), Act::from_num(0.5)],
            vec![Act::from_num(-1.0), Act::from_num(2.0)],
        ]);
        let projection_rows = vec![
            vec![Wgt::from_num(0.5), Wgt::from_num(1.0)],
            vec![Wgt::from_num(-1.0), Wgt::from_num(0.25)],
            vec![Wgt::from_num(0.0), Wgt::from_num(2.0)],
        ];

        let projected = project_sequence(&input, &projection_rows).expect("project should work");

        assert_eq!(
            scaled_bits(&projected),
            vec![
                vec![
                    Act::from_num(1.0).to_bits(),
                    Act::from_num(-0.875).to_bits(),
                    Act::from_num(1.0).to_bits(),
                ],
                vec![
                    Act::from_num(1.5).to_bits(),
                    Act::from_num(1.5).to_bits(),
                    Act::from_num(4.0).to_bits(),
                ],
            ]
        );
    }

    #[test]
    fn projection_can_read_rows_from_authenticated_source() {
        let source = ProjectionSource::new(HashMap::from([
            ((0, 0), vec![Wgt::from_num(0.5), Wgt::from_num(1.0)]),
            ((0, 1), vec![Wgt::from_num(-1.0), Wgt::from_num(0.25)]),
        ]));
        let input =
            RasterActivationSequence::from_acts(vec![vec![Act::from_num(1.0), Act::from_num(0.5)]]);

        let projected =
            project_sequence_with_source(&input, &source, 0, 2).expect("project should work");

        assert_eq!(
            scaled_bits(&projected),
            vec![vec![
                Act::from_num(1.0).to_bits(),
                Act::from_num(-0.875).to_bits(),
            ]]
        );
    }

    #[test]
    fn rms_norm_sequence_matches_det_num_fixture() {
        let input = RasterActivationSequence::from_acts(vec![vec![
            Act::from_bits(65_536),
            Act::from_bits(0),
        ]]);
        let weights = vec![Wgt::from_bits(32_768), Wgt::from_bits(65_536)];

        let normalized =
            rms_norm_sequence(&input, Some(&weights), Some(Acc::from_bits(0))).expect("norm");

        assert_eq!(
            scaled_bits(&normalized),
            vec![rms_norm(
                &[Act::from_bits(65_536), Act::from_bits(0)],
                &weights,
                Acc::from_bits(0),
            )
            .into_iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()]
        );
    }

    #[test]
    fn empty_sequence_fails_clearly() {
        let input = RasterActivationSequence::from_acts(Vec::new());

        let error = scale_sequence(&input, Some(Act::from_num(1.0)))
            .expect_err("empty sequence should fail");

        assert!(error
            .to_string()
            .contains("requires at least one activation row"));
    }

    #[test]
    fn width_mismatch_fails_clearly() {
        let lhs = RasterActivationSequence::from_acts(vec![vec![Act::from_num(1.0)]]);
        let rhs =
            RasterActivationSequence::from_acts(vec![vec![Act::from_num(1.0), Act::from_num(2.0)]]);

        let error = add_sequences(&lhs, &rhs).expect_err("width mismatch should fail");

        assert!(error
            .to_string()
            .contains("right sequence row 0 has width 2"));
    }

    #[test]
    fn missing_canonical_inputs_fail_closed() {
        let input = RasterActivationSequence::from_acts(vec![vec![Act::from_num(1.0)]]);

        let scale_error = scale_sequence(&input, None).expect_err("missing scalar should fail");
        assert!(scale_error
            .to_string()
            .contains("requires canonical Act scalar"));

        let norm_error =
            rms_norm_sequence(&input, None, Some(Acc::from_num(0.0))).expect_err("missing norm");
        assert!(norm_error
            .to_string()
            .contains("requires canonical norm weights"));

        let eps_error =
            rms_norm_sequence(&input, Some(&[Wgt::from_num(1.0)]), None).expect_err("missing eps");
        assert!(eps_error
            .to_string()
            .contains("requires canonical Acc epsilon"));
    }

    #[test]
    fn missing_projection_row_from_source_fails_closed() {
        let source = ProjectionSource::new(HashMap::from([(
            (0, 0),
            vec![Wgt::from_num(1.0), Wgt::from_num(0.0)],
        )]));
        let input =
            RasterActivationSequence::from_acts(vec![vec![Act::from_num(1.0), Act::from_num(0.5)]]);

        let error = project_sequence_with_source(&input, &source, 0, 2)
            .expect_err("missing projection row should fail");

        assert!(error
            .to_string()
            .contains("missing projection row 1 for layer 0"));
    }

    fn scaled_bits(sequence: &RasterActivationSequence) -> Vec<Vec<i32>> {
        sequence
            .rows()
            .iter()
            .map(|row| row.act_bits().to_vec())
            .collect()
    }

    struct ProjectionSource {
        rows: HashMap<(usize, usize), Vec<Wgt>>,
    }

    impl ProjectionSource {
        fn new(rows: HashMap<(usize, usize), Vec<Wgt>>) -> Self {
            Self { rows }
        }
    }

    impl AuthRead<GemmaPleModelProjectionRowRequest> for ProjectionSource {
        type Output = Vec<Wgt>;

        fn auth_read(&self, request: GemmaPleModelProjectionRowRequest) -> Result<Self::Output> {
            self.rows
                .get(&(request.layer_idx, request.row_idx))
                .cloned()
                .ok_or_else(|| {
                    anyhow!(
                        "missing projection row {} for layer {}",
                        request.row_idx,
                        request.layer_idx
                    )
                })
        }
    }
}
