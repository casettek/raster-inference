//! Flat, contiguous tensor buffers for the native deterministic path.
//!
//! These types replace nested `Vec<Vec<_>>` representations on the
//! deterministic hot paths. Only canonical `Act` values are stored — no f32
//! mirrors. In-memory layout is not contract surface; canonical serialized
//! bytes (see the commitment builders in `transformer_kernels`) are.

use anyhow::{bail, Result};

use crate::shared::numerics::det_num::Act;

/// Flat row-major activation matrix (`rows × cols`).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ActSlab {
    data: Vec<Act>,
    rows: usize,
    cols: usize,
}

impl ActSlab {
    pub(crate) fn zeroed(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![Act::from_bits(0); rows * cols],
            rows,
            cols,
        }
    }

    pub(crate) fn from_rows(rows: &[Vec<Act>]) -> Result<Self> {
        let cols = rows.first().map(Vec::len).unwrap_or(0);
        let mut data = Vec::with_capacity(rows.len() * cols);
        for (row_idx, row) in rows.iter().enumerate() {
            if row.len() != cols {
                bail!(
                    "activation row {row_idx} has width {}, expected {cols}",
                    row.len()
                );
            }
            data.extend_from_slice(row);
        }
        Ok(Self {
            data,
            rows: rows.len(),
            cols,
        })
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn cols(&self) -> usize {
        self.cols
    }

    pub(crate) fn row(&self, row_idx: usize) -> &[Act] {
        &self.data[row_idx * self.cols..(row_idx + 1) * self.cols]
    }

    pub(crate) fn row_mut(&mut self, row_idx: usize) -> &mut [Act] {
        &mut self.data[row_idx * self.cols..(row_idx + 1) * self.cols]
    }

    pub(crate) fn as_flat(&self) -> &[Act] {
        &self.data
    }

    pub(crate) fn as_flat_mut(&mut self) -> &mut [Act] {
        &mut self.data
    }

    /// Row-disjoint mutable chunks (one per row), for row-parallel writers.
    pub(crate) fn rows_chunks_mut(&mut self) -> std::slice::ChunksMut<'_, Act> {
        self.data.chunks_mut(self.cols)
    }

    pub(crate) fn iter_rows(&self) -> std::slice::Chunks<'_, Act> {
        self.data.chunks(self.cols)
    }

    pub(crate) fn to_nested(&self) -> Vec<Vec<Act>> {
        self.iter_rows().map(<[Act]>::to_vec).collect()
    }
}

/// Flat head-major buffer (`heads × rows × cols` in one allocation).
///
/// Within a head, rows are contiguous, so `(start..=end)` row windows are a
/// single borrowed slice.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct HeadSlab {
    data: Vec<Act>,
    heads: usize,
    rows: usize,
    cols: usize,
}

impl HeadSlab {
    pub(crate) fn zeroed(heads: usize, rows: usize, cols: usize) -> Self {
        Self {
            data: vec![Act::from_bits(0); heads * rows * cols],
            heads,
            rows,
            cols,
        }
    }

    pub(crate) fn heads(&self) -> usize {
        self.heads
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn cols(&self) -> usize {
        self.cols
    }

    fn head_offset(&self, head_idx: usize) -> usize {
        head_idx * self.rows * self.cols
    }

    pub(crate) fn head_row(&self, head_idx: usize, row_idx: usize) -> &[Act] {
        let start = self.head_offset(head_idx) + row_idx * self.cols;
        &self.data[start..start + self.cols]
    }

    pub(crate) fn head_row_mut(&mut self, head_idx: usize, row_idx: usize) -> &mut [Act] {
        let start = self.head_offset(head_idx) + row_idx * self.cols;
        &mut self.data[start..start + self.cols]
    }

    /// Borrowed contiguous window of `len` rows starting at `start` within one head.
    pub(crate) fn head_rows_window(&self, head_idx: usize, start: usize, len: usize) -> &[Act] {
        let begin = self.head_offset(head_idx) + start * self.cols;
        &self.data[begin..begin + len * self.cols]
    }

    /// Head-disjoint mutable chunks (one per head), for head-parallel writers.
    pub(crate) fn heads_chunks_mut(&mut self) -> std::slice::ChunksMut<'_, Act> {
        self.data.chunks_mut(self.rows * self.cols)
    }
}

/// Append-only flat row buffer for one KV head with a logical start offset.
///
/// Sliding-window trimming advances `start_row`; the buffer is compacted
/// (rows memmoved to the front) once the dead prefix exceeds the live rows, so
/// logical rows always form a single contiguous `&[Act]` slice and amortized
/// append/trim cost stays O(row).
#[derive(Clone, Debug)]
pub(crate) struct ActHeadBuf {
    data: Vec<Act>,
    cols: usize,
    start_row: usize,
    len_rows: usize,
}

impl ActHeadBuf {
    pub(crate) fn new(cols: usize) -> Self {
        Self {
            data: Vec::new(),
            cols,
            start_row: 0,
            len_rows: 0,
        }
    }

    pub(crate) fn cols(&self) -> usize {
        self.cols
    }

    pub(crate) fn len(&self) -> usize {
        self.len_rows
    }

    pub(crate) fn reserve_rows(&mut self, additional_rows: usize) {
        self.data.reserve(additional_rows * self.cols);
    }

    pub(crate) fn push_row(&mut self, row: &[Act]) {
        if self.len_rows == 0 && self.data.is_empty() {
            // Adopt the row width on first append so callers that cannot know
            // the head dimension up front (empty caches) still work.
            self.cols = row.len();
            self.start_row = 0;
        }
        debug_assert_eq!(row.len(), self.cols);
        self.data.extend_from_slice(row);
        self.len_rows += 1;
    }

    pub(crate) fn pop_front_row(&mut self) {
        debug_assert!(self.len_rows > 0);
        self.start_row += 1;
        self.len_rows -= 1;
        if self.start_row > self.len_rows {
            self.compact();
        }
    }

    fn compact(&mut self) {
        let live = self.len_rows * self.cols;
        let start = self.start_row * self.cols;
        self.data.copy_within(start..start + live, 0);
        self.data.truncate(live);
        self.start_row = 0;
    }

    /// Borrowed contiguous window of `len` logical rows starting at `start`.
    pub(crate) fn rows_window(&self, start: usize, len: usize) -> &[Act] {
        let begin = (self.start_row + start) * self.cols;
        &self.data[begin..begin + len * self.cols]
    }

    pub(crate) fn iter_rows(&self) -> std::slice::Chunks<'_, Act> {
        self.rows_window(0, self.len_rows).chunks(self.cols.max(1))
    }
}

impl PartialEq for ActHeadBuf {
    fn eq(&self, other: &Self) -> bool {
        // Logical equality: compaction state is not observable.
        self.cols == other.cols
            && self.len_rows == other.len_rows
            && self.rows_window(0, self.len_rows) == other.rows_window(0, other.len_rows)
    }
}

/// Canonical per-layer KV cache: one flat append-only buffer per KV head for
/// keys and values. Replaces the nested `Vec<VecDeque<Vec<Act>>>` det track.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DetKvCacheData {
    keys: Vec<ActHeadBuf>,
    values: Vec<ActHeadBuf>,
}

impl DetKvCacheData {
    pub(crate) fn new(num_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            keys: vec![ActHeadBuf::new(head_dim); num_kv_heads],
            values: vec![ActHeadBuf::new(head_dim); num_kv_heads],
        }
    }

    /// Builds a prefill cache from head-major K/V slabs, dropping the first
    /// `retained` rows per head (sliding-window retention: keep the tail).
    pub(crate) fn from_head_slabs(keys: &HeadSlab, values: &HeadSlab, retained: usize) -> Self {
        let kept = keys.rows() - retained;
        let mut cache = Self::new(keys.heads(), keys.cols());
        for head_idx in 0..keys.heads() {
            let key_buf = &mut cache.keys[head_idx];
            key_buf.reserve_rows(kept);
            key_buf
                .data
                .extend_from_slice(keys.head_rows_window(head_idx, retained, kept));
            key_buf.len_rows = kept;
            let value_buf = &mut cache.values[head_idx];
            value_buf.reserve_rows(kept);
            value_buf
                .data
                .extend_from_slice(values.head_rows_window(head_idx, retained, kept));
            value_buf.len_rows = kept;
        }
        cache
    }

    pub(crate) fn from_nested_rows(
        keys: &[Vec<Vec<Act>>],
        values: &[Vec<Vec<Act>>],
        head_dim_hint: usize,
    ) -> Self {
        let head_dim = keys
            .iter()
            .find_map(|head| head.first().map(Vec::len))
            .unwrap_or(head_dim_hint);
        let mut cache = Self::new(keys.len(), head_dim);
        for (head_idx, head) in keys.iter().enumerate() {
            for row in head {
                cache.keys[head_idx].push_row(row);
            }
        }
        for (head_idx, head) in values.iter().enumerate() {
            for row in head {
                cache.values[head_idx].push_row(row);
            }
        }
        cache
    }

    pub(crate) fn num_heads(&self) -> usize {
        self.keys.len()
    }

    pub(crate) fn head_dim(&self) -> usize {
        self.keys.first().map(ActHeadBuf::cols).unwrap_or(0)
    }

    /// Logical sequence length (rows per head).
    pub(crate) fn len(&self) -> usize {
        self.keys.first().map(ActHeadBuf::len).unwrap_or(0)
    }

    /// Appends one row per head from flat `heads × head_dim` buffers with
    /// sliding-window trimming; zero allocation beyond amortized buffer growth.
    pub(crate) fn append_rows(
        &mut self,
        new_keys_flat: &[Act],
        new_values_flat: &[Act],
        head_dim: usize,
        sliding_window: Option<usize>,
    ) -> Result<()> {
        let heads = self.keys.len();
        if new_keys_flat.len() != heads * head_dim || new_values_flat.len() != heads * head_dim {
            bail!(
                "layer cache append head count mismatch: cache {} keys {} values {}",
                heads,
                new_keys_flat.len() / head_dim.max(1),
                new_values_flat.len() / head_dim.max(1)
            );
        }
        for head_idx in 0..heads {
            self.keys[head_idx]
                .push_row(&new_keys_flat[head_idx * head_dim..(head_idx + 1) * head_dim]);
            self.values[head_idx]
                .push_row(&new_values_flat[head_idx * head_dim..(head_idx + 1) * head_dim]);
            if let Some(window) = sliding_window {
                while self.keys[head_idx].len() > window {
                    self.keys[head_idx].pop_front_row();
                    self.values[head_idx].pop_front_row();
                }
            }
        }
        Ok(())
    }

    /// Appends one nested row per head with sliding-window trimming, matching
    /// the legacy `push_back` + `while len > window pop_front` semantics.
    pub(crate) fn append_step_nested(
        &mut self,
        key_rows: &[Vec<Act>],
        value_rows: &[Vec<Act>],
        sliding_window: Option<usize>,
    ) -> Result<()> {
        if key_rows.len() != self.keys.len() || value_rows.len() != self.values.len() {
            bail!(
                "layer cache append head count mismatch: cache {} keys {} values {}",
                self.keys.len(),
                key_rows.len(),
                value_rows.len()
            );
        }
        for head_idx in 0..self.keys.len() {
            self.keys[head_idx].push_row(&key_rows[head_idx]);
            self.values[head_idx].push_row(&value_rows[head_idx]);
            if let Some(window) = sliding_window {
                while self.keys[head_idx].len() > window {
                    self.keys[head_idx].pop_front_row();
                    self.values[head_idx].pop_front_row();
                }
            }
        }
        Ok(())
    }

    pub(crate) fn key_window(&self, head_idx: usize, start: usize, len: usize) -> &[Act] {
        self.keys[head_idx].rows_window(start, len)
    }

    pub(crate) fn value_window(&self, head_idx: usize, start: usize, len: usize) -> &[Act] {
        self.values[head_idx].rows_window(start, len)
    }

    pub(crate) fn key_heads(&self) -> &[ActHeadBuf] {
        &self.keys
    }

    pub(crate) fn value_heads(&self) -> &[ActHeadBuf] {
        &self.values
    }

    pub(crate) fn nested_key_rows(&self, head_idx: usize) -> Vec<Vec<Act>> {
        self.keys[head_idx].iter_rows().map(<[Act]>::to_vec).collect()
    }

    pub(crate) fn nested_value_rows(&self, head_idx: usize) -> Vec<Vec<Act>> {
        self.values[head_idx]
            .iter_rows()
            .map(<[Act]>::to_vec)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn act(value: i32) -> Act {
        Act::from_bits(value)
    }

    #[test]
    fn act_slab_round_trips_rows() {
        let rows = vec![vec![act(1), act(2)], vec![act(3), act(4)]];
        let slab = ActSlab::from_rows(&rows).expect("slab should build");
        assert_eq!(slab.rows(), 2);
        assert_eq!(slab.cols(), 2);
        assert_eq!(slab.row(0), &[act(1), act(2)]);
        assert_eq!(slab.row(1), &[act(3), act(4)]);
        assert_eq!(slab.to_nested(), rows);
    }

    #[test]
    fn head_slab_windows_are_contiguous() {
        let mut slab = HeadSlab::zeroed(2, 3, 2);
        for head in 0..2 {
            for row in 0..3 {
                let dst = slab.head_row_mut(head, row);
                dst[0] = act((head * 10 + row) as i32);
                dst[1] = act((head * 10 + row + 100) as i32);
            }
        }
        assert_eq!(
            slab.head_rows_window(1, 1, 2),
            &[act(11), act(111), act(12), act(112)]
        );
    }

    #[test]
    fn head_buf_trim_and_compaction_preserve_logical_rows() {
        let mut buf = ActHeadBuf::new(2);
        for idx in 0..6 {
            buf.push_row(&[act(idx), act(idx + 100)]);
        }
        for _ in 0..4 {
            buf.pop_front_row();
        }
        assert_eq!(buf.len(), 2);
        assert_eq!(buf.rows_window(0, 1), &[act(4), act(104)]);
        assert_eq!(buf.rows_window(0, 2), &[act(4), act(104), act(5), act(105)]);
        // Trim happened past the live length, so the buffer compacted.
        assert_eq!(buf.start_row, 0);
        buf.push_row(&[act(6), act(106)]);
        assert_eq!(buf.rows_window(2, 1), &[act(6), act(106)]);
    }

    #[test]
    fn kv_cache_append_matches_sliding_window_retention() {
        let mut cache = DetKvCacheData::new(1, 2);
        for step in 0..5 {
            let keys = vec![act(step), act(step)];
            let values = vec![act(step + 50), act(step + 50)];
            cache
                .append_rows(&keys, &values, 2, Some(3))
                .expect("append should succeed");
        }
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.key_window(0, 0, 1), &[act(2), act(2)]);
        assert_eq!(cache.key_window(0, 2, 1), &[act(4), act(4)]);
        assert_eq!(cache.value_window(0, 0, 1), &[act(52), act(52)]);
    }

    #[test]
    fn prefill_retention_keeps_tail_rows() {
        let mut keys = HeadSlab::zeroed(1, 4, 1);
        let mut values = HeadSlab::zeroed(1, 4, 1);
        for row in 0..4 {
            keys.head_row_mut(0, row)[0] = act(row as i32);
            values.head_row_mut(0, row)[0] = act(row as i32 + 10);
        }
        let cache = DetKvCacheData::from_head_slabs(&keys, &values, 2);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.key_window(0, 0, 2), &[act(2), act(3)]);
        assert_eq!(cache.value_window(0, 0, 2), &[act(12), act(13)]);
    }

    #[test]
    fn logical_equality_ignores_compaction_state() {
        let mut lhs = ActHeadBuf::new(1);
        lhs.push_row(&[act(1)]);
        lhs.push_row(&[act(2)]);
        lhs.push_row(&[act(3)]);
        lhs.pop_front_row();

        let mut rhs = ActHeadBuf::new(1);
        rhs.push_row(&[act(2)]);
        rhs.push_row(&[act(3)]);

        assert_eq!(lhs, rhs);
    }
}
