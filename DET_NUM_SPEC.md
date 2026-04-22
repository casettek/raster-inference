# `det_num` v0 Arithmetic Spec

## Purpose

`det_num` defines the canonical arithmetic semantics for deterministic inference in `raster-inference`.

Its purpose is to ensure that the same model, inputs, and code produce identical outputs across:

- different CPUs
- different compiler settings
- future RISC Zero guest execution

`det_num` is the source of truth for arithmetic semantics. External libraries like `fixed` and `fixed_analytics` are implementation dependencies only.

---

## Design goals

- deterministic and portable
- simple enough to implement and test early
- efficient enough for CPU inference
- strict enough to preserve future zkVM parity
- small surface area for v0

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

These bit layouts and semantics are part of the arithmetic contract.

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

Recommended additional helpers:

- `fn add_sat(a: Act, b: Act) -> Act`
- `fn sub_sat(a: Act, b: Act) -> Act`
- `fn acc_add_sat(a: Acc, b: Acc) -> Acc`
- `fn rshift_round_ties_even(x: Acc, shift: u32) -> Acc`
- `fn act_to_le_bytes(x: Act) -> [u8; 4]`
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
- `tanh`
- `softmax`
- `rmsnorm`
- `rope`

Those belong in later layers built on top of `det_num`.

v0 only defines the arithmetic substrate those operators must use.

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

### Serialization tests

- byte encoding is stable
- little-endian output is exact
- cross-machine byte equality

### Cross-build tests

- debug vs release parity
- x86 vs ARM parity for the same inputs

---

## Versioning

This spec is versioned as:

- `det_num_spec_version = 0`

Any future change to:

- type aliases
- rounding mode
- overflow policy
- shift policy
- tie-breaking
- serialization

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
