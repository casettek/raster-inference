# detwgt v2 — Deterministic Weight Artifact Format

Status: CURRENT (canonical contract surface)

This document specifies the byte layout of `.detwgt` v2, the deterministic
weight artifact consumed by the native loader and the raster-side artifact
readers. The encoder lives in `src/shared/numerics/det_num/artifact.rs`; the
canonical parser is `DetNumTensorReader::load_artifact` in `src/io.rs`.

detwgt v2 supersedes v1. v2-pinned loaders **reject v1 artifacts fail-closed**;
there is no dual-version runtime support. Re-convert v1 models with
`gemma-det-num-wgt-converter`.

## Relationship to DET_NUM_SPEC

The artifact format version (`detwgt_version`) is independent of
`det_num_spec_version`:

- `detwgt_version = 2`
- `det_num_spec_version = 1`

v2 changes the **representation** of weights, not their values or any
arithmetic, so the numeric spec version is unchanged (see DET_NUM_SPEC,
"Width-reduced storage").

## Semantic invariant (normative)

Storage width is representation only. The canonical value of every weight is
its **sign-extended integer** (Q16.16 bit pattern in i32). All arithmetic
occurs after widening: i16-stored values are sign-extended to i32 on read,
and the exact widening multiply i32×i32→i64 produces identical product bits
regardless of storage width. A tensor may be stored i16 **iff** every one of
its values fits i16 (`-32768 <= wgt_bits <= 32767`, i.e. |w| < 0.5 in
Q16.16). Converting the same tensor at either width MUST produce identical
GEMV outputs; this is enforced by conformance tests.

## File layout

All integers are little-endian. There is no trailing data after the last
tensor payload.

### File header (24 bytes)

| Offset | Size | Field                  | Value                  |
|--------|------|------------------------|------------------------|
| 0      | 8    | magic                  | `b"DNWGTV0\0"`         |
| 8      | 4    | `detwgt_version` (u32) | `2`                    |
| 12     | 4    | `det_num_spec_version` (u32) | `1`              |
| 16     | 8    | `tensor_count` (u64)   |                        |

### Per-tensor record (repeated `tensor_count` times)

| Field             | Size            | Notes                                          |
|-------------------|-----------------|------------------------------------------------|
| `name_len` (u32)  | 4               |                                                |
| `name`            | `name_len`      | UTF-8, unique within the artifact              |
| `rank` (u32)      | 4               |                                                |
| `dims` (u64 each) | 8 × `rank`      |                                                |
| `element_count` (u64) | 8           | MUST equal the product of `dims`               |
| `element_width` (u32) | 8 → see note | `16` or `32` (bit count) — **new in v2**      |
| `payload_len` (u64) | 8             | MUST equal `element_count × element_width/8`   |
| `max_row_mass` (u64) | 8            | `max_r Σ_i \|wgt_bits[r][i]\|` over last-dim rows |
| padding           | 0–63            | zero bytes until the file offset is a multiple of 64 — **new in v2** |
| payload           | `payload_len`   | values as LE i16 or i32 at `element_width`     |

(The `element_width` field is 4 bytes; it sits between `element_count` and
`payload_len`.)

### Payload alignment

Payloads start at file offsets that are multiples of **64 bytes**
(`DET_WGT_PAYLOAD_ALIGNMENT`). Padding bytes are part of the canonical
encoding and MUST be zero; parsers MUST verify this. Because mmap bases are
page-aligned, 64-byte file-offset alignment guarantees element alignment in
memory, so little-endian hosts always take the zero-copy mmap view for
contiguous full-width slices (completing the WS2 zero-copy goal).

### Element width selection (converter, normative)

Per tensor: emit i16 iff every converted Q16.16 value fits i16; otherwise
emit i32. The conversion-time row-mass overflow bound
(`sum_i |wgt_bits[r][i]| < 2^31` for every rank ≥ 2 row) is checked
regardless of storage width, and `max_row_mass` is always computed over the
canonical widened values.

## Loader obligations (fail-closed)

- magic mismatch → reject
- `detwgt_version != 2` → reject (point the user at the converter)
- `det_num_spec_version != 1` → reject
- `element_width ∉ {16, 32}` → reject
- `element_count` ≠ shape product, or `payload_len` ≠
  `element_count × width/8` → reject
- non-zero padding → reject
- rank ≥ 2 tensor with `max_row_mass ≥ 2^31` → reject (reload-time re-check
  of the conversion bound)
- duplicate tensor names, truncated files, trailing bytes → reject

## Commitments

External-source Merkle roots commit canonical i32 `Wgt` bit patterns
(postcard-encoded row vectors), not artifact file bytes, and `det_*`
checkpoint commitments hash activations/logits/KV only. Re-converting a
model between widths therefore changes **no** registered root and **no**
checkpoint commitment; only the artifact bytes change.

## v1 → v2 migration

1. Re-run `gemma-det-num-wgt-converter` against the original safetensors
   model. The converter prints an i16 width summary (and per-tensor widths
   with `--report-widths`).
2. Replace the deployed `model.detwgt`. v1 files are rejected by v2 loaders.
