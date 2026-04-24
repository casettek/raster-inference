# `det_num` v0 Arithmetic and Model-Compilation Spec

## Purpose

`det_num` defines the canonical arithmetic semantics for deterministic inference in `raster-inference`.

Its purpose is to ensure that the same model, inputs, and code produce identical outputs across:

- different CPUs
- different compiler settings
- future RISC Zero guest execution

`det_num` is the source of truth for arithmetic semantics. External libraries like `fixed` and `fixed_analytics` are implementation dependencies only.

In v0, `det_num` also defines the canonical numeric representation of compiled model weights used by inference kernels.

---

## Design goals

- deterministic and portable
- simple enough to implement and test early
- efficient enough for CPU inference
- strict enough to preserve future zkVM parity
- small surface area for v0
- isolate arithmetic-path quality effects from storage-format compression effects

---

## Numeric model

### Core fixed-point types

The runtime defines three canonical scalar types:

- `Act`: activation type
- `Wgt`: weight type
- `Acc`: accumulator type

### Initial v0 choices

These are the canonical v0 scalar layouts:

- `Act`: signed 32-bit fixed-point scalar with 16 fractional bits
- `Wgt`: signed 32-bit fixed-point scalar with 16 fractional bits
- `Acc`: signed 64-bit fixed-point scalar with 32 fractional bits

Interpretation:

- activations and weights use a signed 32-bit fixed-point layout with 16 fractional bits
- accumulators use a signed 64-bit fixed-point layout with 32 fractional bits
- concrete host-library type names may differ as long as the bit layout and canonical arithmetic semantics are preserved

This gives:

- enough fractional precision to start safely
- wide accumulation headroom
- easy multiply/requantize behavior
- a single canonical executable weight format for v0

These bit layouts and semantics are part of the arithmetic contract.

### Real-value interpretation

Canonical interpretation is:

- `Act(x)` represents `x / 2^16`
- `Wgt(x)` represents `x / 2^16`
- `Acc(x)` represents `x / 2^32`

where `x` is the signed integer payload of the scalar.

---

## Canonical arithmetic rules

### 1. Multiplication

All scalar multiplication between `Act` and `Wgt` must use widening multiply.

Rule:

- `mul_wide(a: Act, b: Wgt) -> Acc`

Semantics:

- multiplication is performed in widened precision
- no narrow multiply is allowed in canonical kernels
- no implicit truncation is allowed during multiply
- the mathematical interpretation is Q16.16 × Q16.16 -> Q32.32

### 2. Multiply-accumulate

All dot products and affine transforms must accumulate only in `Acc`.

Rule:

- `mac(acc: Acc, a: Act, b: Wgt) -> Acc`

Semantics:

- compute `mul_wide(a, b)`
- add result to `acc`
- addition uses saturating arithmetic
- no narrowing during accumulation

### 3. Overflow policy

Canonical runtime overflow behavior is:

- **saturating**

This applies to:

- addition
- subtraction
- narrowing/requantization
- explicit clipping steps
- model-compilation conversion into canonical scalar types

Wrapping arithmetic is forbidden in canonical inference kernels unless a future spec version explicitly introduces it.

### 4. Requantization

Rule:

- `requantize(x: Acc) -> Act`

Semantics:

- narrowing from `Acc` to `Act` is always explicit
- rounding mode is:
  - **round to nearest**
  - **ties to even**
- narrowing uses saturating conversion
- no implicit casts are allowed in canonical kernels
- in v0, `clip_act(x)` is intentionally identical to `requantize(x)`

### 5. Right-shift semantics

All right shifts used for scaling/requantization must be routed through canonical helper functions.

Semantics:

- signed right shifts must preserve sign
- any rounding before shift must be explicit
- raw `>>` in model kernels is forbidden unless wrapped in `det_num`
- `rshift_round_ties_even(x, shift)` panics for `shift >= 64`

Rule:

- use helper such as `rshift_round_ties_even(...)`

### 6. Comparison semantics

Canonical scalar comparison is exact comparison on fixed-point encoded values.

For argmax:

- larger value wins
- if values are equal, **lowest index wins**

Rule:

- `argmax_first(xs: &[Act]) -> usize`

Behavior:

- `argmax_first` panics on empty slices

This tie-break rule is part of the contract.

### 7. RoPE semantics

Canonical deterministic RoPE is defined on `Act` rows and must not rely on host `f32`
transcendentals at execution time.

Rule:

- `rope_rotate_pairs(input: &[Act], rotary_dim: usize, freq_base_dim: usize, base: Acc, position: usize) -> Vec<Act>`

Semantics:

- `rotary_dim == 0` is a no-op
- otherwise, `rotary_dim` must be even and must fit within `input.len()`
- `freq_base_dim` must be even and at least 2
- `base` must be strictly positive
- pair `i` rotates `input[i]` with `input[i + rotary_dim / 2]`
- the per-pair inverse-frequency step is the canonical Q32.32 reciprocal of the canonical Q32.32 `freq_base_dim / 2`-th root of `base`
- pair `0` uses inverse frequency `1.0`, and each later pair multiplies by that step using canonical Q32.32 fixed-point multiply semantics
- the angle is `position * inverse_frequency`, represented canonically in Q32.32
- `sin` and `cos` are materialized canonically from that Q32.32 angle using only deterministic fixed-point helpers
- rotated outputs use canonical fixed-point multiply/requantize plus saturating add/sub semantics
- dimensions beyond `rotary_dim` are copied through unchanged

### 8. Attention score semantics

Canonical deterministic attention scores are defined on `Act` rows and must not rely on host `f32`
accumulation once deterministic execution is active.

Rule:

- `attention_score(query: &[Act], key: &[Act]) -> Act`

Semantics:

- `query` and `key` must be non-empty and have matching widths
- reduction order is left-to-right over the row
- each term uses canonical widening multiply semantics
- accumulation uses saturating `Acc`
- score materialization is explicit and uses canonical `requantize`
- no host-float dot-product fallback is allowed on the deterministic path

### 9. Attention softmax semantics

Canonical deterministic softmax is defined on `Act` logits and emits canonical `Act` weights.

Rule:

- `attention_softmax(logits: &[Act]) -> Vec<Act>`

Semantics:

- `logits` must be non-empty
- the maximum logit is selected with canonical `argmax_first`, so equal maxima resolve to the
  lowest index
- every exponent term is computed from `logit - max_logit`, so the largest shifted logit is exactly
  zero
- shifted logits are interpreted canonically in Q32.32 before exponent materialization
- exponent materialization uses deterministic range reduction by `ln(2)`, then a fixed polynomial on
  the reduced remainder; no host `exp` is allowed
- exponent terms smaller than the explicit underflow floor clamp to zero
- normalization divides each exponent by the canonical sum with ties-to-even rounding
- the final residual needed to make weights sum to exactly `1.0` in Q16.16 is assigned back to the
  winning `argmax_first` index

### 10. Attention weighted-value aggregation semantics

Canonical deterministic value mixing applies canonical attention weights to canonical value rows.

Rule:

- `attention_weighted_sum(weights: &[Act], value_rows: &[Vec<Act>]) -> Vec<Act>`

Semantics:

- `weights` and `value_rows` must be non-empty and have matching row counts
- all value rows must share one width
- reduction order is row-major and left-to-right within the weight/value pairing
- each multiply uses canonical widening precision
- accumulation uses saturating `Acc`
- each output dimension is materialized with canonical `requantize`

### 11. Deterministic tanh semantics

Canonical deterministic `tanh` is defined on `Act` inputs and must not rely on host `f32`
transcendentals at execution time.

Rule:

- `tanh_act(input: Act) -> Act`

Semantics:

- inputs are interpreted canonically in Q16.16
- saturation is explicit: `input >= 3.0` maps to `1.0`, and `input <= -3.0` maps to `-1.0`
- otherwise, evaluation uses the fixed rational approximation `x * (27 + x^2) / (27 + 9x^2)`
- `x^2` is materialized with canonical `mul_sat`
- numerator and denominator terms are formed with canonical saturating add/multiply helpers
- division uses canonical `div_act`, so rounding is ties-to-even
- evaluation order is fixed and part of the contract:
  1. compute `x^2`
  2. compute `27 + x^2`
  3. compute `27 + 9x^2`
  4. compute `x * (27 + x^2)`
  5. divide by `27 + 9x^2`

### 12. Deterministic GELU(tanh) semantics

Canonical deterministic GELU uses the repo's existing PyTorch tanh structure, but every step is
defined on canonical fixed-point inputs.

Rule:

- `gelu_pytorch_tanh_act(input: Act) -> Act`

Semantics:

- the helper implements `0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))`
- constants are encoded canonically in Q16.16 before evaluation:
  - `0.5 -> 32768`
  - `sqrt(2 / pi) -> 52290`
  - `0.044715 -> 2930`
- `x^2` and `x^3` use canonical `mul_sat`
- the cubic term is formed before adding back to `x`
- the inner scale multiplication happens after that addition
- the `tanh` stage must call canonical `tanh_act`
- the final multiply order is fixed and part of the contract:
  1. compute `x^2`
  2. compute `x^3`
  3. compute `0.044715 * x^3`
  4. compute `x + 0.044715 * x^3`
  5. compute `sqrt(2 / pi) * (...)`
  6. compute `tanh_act(...)`
  7. compute `0.5 * x`
  8. compute `1 + tanh(...)`
  9. multiply those two terms
- no host-float GELU or host `tanh` fallback is allowed once deterministic MLP execution is active

---

## Canonical source conversion rules

### Overview

`det_num` v0 defines a canonical conversion from source FP32 weights into executable `Wgt` tensors.

This conversion is part of the deterministic model-compilation process.

The source FP32 model is an upstream artifact only. It is **not** the canonical executable inference artifact.

The canonical executable model artifact for v0 stores weights directly as `Wgt` values.

### FP32 -> `Wgt` conversion

Rule:

- `f32_to_wgt(x: f32) -> Wgt`

Semantics:

1. Interpret `x` as a real-valued scalar.
2. Multiply by `2^16`.
3. Round to nearest, ties to even.
4. Saturate to signed 32-bit range.
5. Store the resulting signed 32-bit integer payload as `Wgt`.

Equivalent mathematical form:

- `q = sat_i32(round_ties_even(x * 65536.0))`

Interpretation:

- resulting `Wgt(q)` represents `q / 65536`

### v0 storage policy

v0 uses **direct canonical Q16.16 weight storage**.

v0 does **not** introduce:

- per-tensor external scales
- per-channel external scales
- block quantization metadata
- compressed weight encodings optimized for size

This is intentional, to isolate arithmetic-path effects from separate storage-format approximation schemes.

### Canonical model artifact

The v0 executable model artifact must store weights in canonical `Wgt` form.

The artifact may contain:

- format magic/version
- `det_num_spec_version`
- tensor metadata
- tensor shapes
- tensor role/type
- canonical `Wgt` payload bytes

The artifact must not require host-native floating-point interpretation during inference.

---

## Serialization rules

All serialized arithmetic values must use:

- **little-endian**
- canonical fixed-width byte encoding
- no host-dependent layout assumptions

Rules:

- all scalar numeric values serialized with explicit byte conversion
- no raw memory reinterpretation as canonical serialization format
- all checkpoint/state hashes must derive from canonical serialized bytes

For v0 compiled weights:

- each `Wgt` scalar is serialized as a 4-byte signed little-endian integer
- each `Act` scalar is serialized as a 4-byte signed little-endian integer
- each `Acc` scalar is serialized as an 8-byte signed little-endian integer

---

## Forbidden behavior

The following are forbidden in canonical arithmetic code:

- native floating-point as trusted semantics
- implicit narrowing conversions
- implicit overflow behavior
- direct use of transcendental host math (`exp`, `sin`, `cos`, `tanh`, etc.)
- non-deterministic reduction order
- platform-dependent serialization
- unordered or unstable argmax behavior

The following are additionally forbidden in the v0 executable inference path:

- reading FP32 weights as runtime arithmetic truth
- load-time weight interpretation that changes canonical numeric meaning
- alternate weight encodings that are not direct `Wgt` payloads

---

## Required wrapper API

`det_num` v0 must expose at least:

- `type Act`
- `type Wgt`
- `type Acc`

- `fn mul_wide(a: Act, b: Wgt) -> Acc`
- `fn mac(acc: Acc, a: Act, b: Wgt) -> Acc`

- `fn requantize(x: Acc) -> Act`
- `fn clip_act(x: Acc) -> Act`

- `fn argmax_first(xs: &[Act]) -> usize`
- `fn rope_rotate_pairs(input: &[Act], rotary_dim: usize, freq_base_dim: usize, base: Acc, position: usize) -> Vec<Act>`
- `fn attention_score(query: &[Act], key: &[Act]) -> Act`
- `fn attention_softmax(logits: &[Act]) -> Vec<Act>`
- `fn attention_weighted_sum(weights: &[Act], value_rows: &[Vec<Act>]) -> Vec<Act>`
- `fn tanh_act(input: Act) -> Act`
- `fn gelu_pytorch_tanh_act(input: Act) -> Act`

- `fn f32_to_wgt(x: f32) -> Wgt`

Recommended additional helpers:

- `fn add_sat(a: Act, b: Act) -> Act`
- `fn sub_sat(a: Act, b: Act) -> Act`
- `fn acc_add_sat(a: Acc, b: Acc) -> Acc`
- `fn rshift_round_ties_even(x: Acc, shift: u32) -> Acc`
- `fn act_to_le_bytes(x: Act) -> [u8; 4]`
- `fn wgt_to_le_bytes(x: Wgt) -> [u8; 4]`
- `fn acc_to_le_bytes(x: Acc) -> [u8; 8]`

---

## v0 scope limits

`det_num` v0 does **not** yet define canonical semantics for:

- `exp`
- `log`
- `sqrt`
- `rsqrt`
- `sin`
- `cos`
- `rmsnorm`

Those belong in later layers built on top of `det_num`.

v0 defines:

- the arithmetic substrate
- the canonical executable weight representation
- the canonical FP32 -> `Wgt` model-compilation rule

v0 does **not** yet define full end-to-end inference semantics.

---

## Testing requirements

Before using `det_num` in model kernels, it must pass:

### Golden scalar tests

- multiply widening behavior
- saturating add/sub behavior
- requantization edge cases
- ties-to-even narrowing cases
- signed shift behavior
- argmax tie-breaking
- attention score accumulation
- attention softmax normalization and tie handling
- attention weighted-value aggregation
- tanh saturation, sign, and ties-to-even division behavior
- GELU representative positive, negative, near-zero, and tie-sensitive cases

### Weight-conversion tests

- `f32_to_wgt` exactness for representative values
- ties-to-even conversion cases near half-step boundaries
- saturation behavior at extreme source values
- byte-stable serialized `Wgt` output

### Serialization tests

- byte encoding is stable
- little-endian output is exact
- cross-machine byte equality

### Cross-build tests

- debug vs release parity
- x86 vs ARM parity for the same inputs

### Transitional validation tests

During migration from the current FP32 inference path:

- compare converted-weight path against existing FP32 baseline
- measure token agreement
- measure logits drift
- record any quality changes before replacing higher-level arithmetic operators

Implementation note for this repo:

- the current Phase 1 deterministic comparison path may load canonical `Wgt` bytes from `model.detwgt` and reconstruct host runtime matrices to isolate converted-weight quality effects
- that comparison path is a migration aid only; it is **not** the final canonical end-to-end `det_num` runtime described by the rules above
- replacing host-runtime arithmetic with full `Act` / `Wgt` / `Acc` execution remains follow-up work after converted-weight parity is validated

---

## Versioning

This spec is versioned as:

- `det_num_spec_version = 0`

Any future change to:

- type aliases
- numeric interpretation
- source conversion rule
- rounding mode
- overflow policy
- shift policy
- tie-breaking
- serialization
- executable weight artifact format

must increment the spec version.

---

## Summary

`det_num` v0 uses:

- `Act`: signed 32-bit fixed-point scalar with 16 fractional bits
- `Wgt`: signed 32-bit fixed-point scalar with 16 fractional bits
- `Acc`: signed 64-bit fixed-point scalar with 32 fractional bits
- widening multiply always
- saturating arithmetic
- explicit requantization only
- round-to-nearest, ties-to-even
- little-endian serialization
- argmax ties resolved by lowest index
- direct canonical Q16.16 compiled weight storage
- explicit FP32 -> `Wgt` deterministic model compilation
- deterministic attention score, softmax, and value-mixing semantics
