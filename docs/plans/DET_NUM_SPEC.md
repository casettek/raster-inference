# `det_num` v1 Arithmetic and Model-Compilation Spec

Status: CURRENT — supersedes `det_num` v0. v0 is archived unamended as `DET_NUM_SPEC_V0.md`.

---

## Changelog: v0 -> v1

Normative changes:

1. **MAC accumulation is wrapping, not saturating.** Dot-product accumulation in `Acc` uses two's-complement wrapping addition. Reduction order over MAC terms is no longer part of the contract (§2, §3).
2. **Reduction-order requirements removed for associative reductions.** The "left-to-right" ordering rules in attention-score and weighted-value-aggregation semantics are deleted (§8, §10). The set of terms per output element remains exactly specified; their grouping and order do not.
3. **Conversion-time overflow bound added (normative).** The model compiler must prove, per weight row, that MAC accumulation cannot exceed `2^62` in magnitude for any representable activation input, and must fail closed otherwise (Canonical source conversion rules).
4. **Parallelism legality section added.** Defines which execution schedules are conforming (new section).
5. **Wraparound determinism clause added.** Defines semantics when wrapping occurs in reductions that cannot be statically bounded (§3).
6. **Storage-width extension pre-authorized.** Exact, value-preserving width-reduced weight encodings are permitted in future artifact format versions; the v0 prohibition on alternate encodings is narrowed to value-changing encodings (v1 storage policy).
7. **Compatibility-commitment retirement for deterministic mode.** Deterministic-mode runs emit only canonical `det_*` commitments; f32 mirror computation and f32 compatibility commitments are removed from the deterministic path (Runtime state boundary).
8. `det_num_spec_version = 1`; mixed-version execution fails closed (Versioning).

Unchanged from v0 (explicitly reaffirmed): type layouts and real-value interpretation; widening multiply; `requantize` rounding (round-to-nearest, ties-to-even) and saturating narrowing; all `Act`-domain saturating elementwise ops (`add_sat`, `sub_sat`, `mul_sat`); right-shift semantics; comparison and `argmax_first` tie-breaking; RoPE semantics; softmax semantics including residual assignment; tanh, softcap, and GELU fixed evaluation orders; FP32 -> `Wgt` conversion rule; serialization rules.

---

## Purpose

`det_num` defines the canonical arithmetic semantics for deterministic inference in `raster-inference`.

Its purpose is to ensure that the same model, inputs, and code produce identical outputs across:

- different CPUs
- different compiler settings
- different execution schedules (serial, multicore, SIMD, GPU)
- future RISC Zero guest execution

`det_num` is the source of truth for arithmetic semantics. External libraries like `fixed` are implementation dependencies only.

v1's central change: the canonical dot-product reduction is made **associative and commutative** by adopting wrapping accumulation, so that conformance is a property of the *values computed*, never of the *schedule used to compute them*. A serial zkVM guest and a vectorized multicore native backend are equal by algebra, not by testing.

---

## Runtime state boundary

Deterministic execution requires a model loaded from a canonical weight artifact (`model.detwgt`). Mixed-mode inputs such as safetensors/f32 model provenance with `InferenceExecutionMode::Deterministic` are invalid and must fail closed.

Deterministic execution stores runtime state internally in canonical fixed-point form:

- layer activation rows use canonical `Act` rows as the source of truth
- deterministic KV cache keys and values retain canonical `Act` rows across decode steps
- logits carry canonical `Act` values through deterministic token selection
- config-derived deterministic scalars (RMS epsilon, RoPE bases) are converted once at load time into canonical `Acc` values
- deterministic scale/softcap scalars are converted once at load time into canonical `Act` values

**v1 change:** deterministic mode computes no `f32` mirror values downstream of weight load. Deterministic-mode outputs carry only canonical commitments (`det_*` fields) over canonical serialized bytes. The v0 f32 compatibility commitments (`activations_sha256`, `final_logits_sha256`, f32 `layer_caches` serializations) are not emitted by deterministic-mode runs. Fp32-mode execution and outputs are unchanged.

The in-memory representation of canonical values (nested vs. flat buffers, borrowed vs. owned, storage width of weights per the artifact format) is not contract surface. Only canonical values and their canonical serialized bytes are.

---

## Design goals

- deterministic and portable
- schedule-independent: conforming results must not depend on execution order, thread count, lane width, or hardware backend
- efficient enough for CPU and GPU inference without semantic compromise
- strict enough to preserve zkVM parity
- small surface area

---

## Numeric model

### Core fixed-point types

Unchanged from v0:

- `Act`: signed 32-bit fixed-point scalar, 16 fractional bits (Q16.16); `Act(x)` represents `x / 2^16`
- `Wgt`: signed 32-bit fixed-point scalar, 16 fractional bits (Q16.16); `Wgt(x)` represents `x / 2^16`
- `Acc`: signed 64-bit fixed-point scalar, 32 fractional bits (Q32.32); `Acc(x)` represents `x / 2^32`

These bit layouts and semantics are part of the arithmetic contract.

---

## Canonical arithmetic rules

### 1. Multiplication

Unchanged. All scalar multiplication between `Act` and `Wgt` uses widening multiply.

- `mul_wide(a: Act, b: Wgt) -> Acc`
- performed in widened precision; the product of two signed 32-bit payloads is always exactly representable in the signed 64-bit accumulator payload — widening multiplication is exact and cannot overflow
- no narrow multiply, no implicit truncation
- mathematical interpretation: Q16.16 × Q16.16 -> Q32.32

### 2. Multiply-accumulate

All dot products and affine transforms accumulate only in `Acc`.

Rule:

- `mac(acc: Acc, a: Act, b: Wgt) -> Acc`
- bit-level form: `mac_bits(acc_bits: i64, act_bits: i32, wgt_bits: i32) -> i64`

Semantics:

- compute the exact widened product `i64(act_bits) * i64(wgt_bits)`
- add the product to the accumulator using **two's-complement wrapping addition** (addition modulo `2^64` on the bit pattern, reinterpreted as signed)
- no narrowing during accumulation

Consequences (normative):

- wrapping addition is associative and commutative; therefore **the grouping and ordering of MAC terms is not part of the contract**
- any partition of a reduction's term set into sub-reductions, computed in any order, on any hardware, combined with wrapping `Acc` addition, is conforming and produces identical accumulator bits
- a reduction is specified by its **term set**: the exact multiset of `(activation, weight)` pairs contributing to each output element. The term set per output element is part of the contract; the schedule is not.
- partial accumulators combined across lanes, threads, warps, or devices must be combined with the same wrapping `Acc` addition. No other combination operation is conforming.

The serial left-to-right loop remains the **reference implementation** and the oracle for differential testing. It is no longer the only conforming schedule.

### 3. Overflow policy

v1 splits overflow behavior by domain:

**Reduction domain (`Acc` MAC accumulation): wrapping.**

- accumulation overflow wraps modulo `2^64`
- wrapping in this domain is fully deterministic and identical across all conforming backends

**Materialization and elementwise domain: saturating.** Unchanged from v0:

- `requantize` narrowing saturates
- `Act`-domain addition, subtraction, multiplication (`add_sat`, `sub_sat`, `mul_sat`) saturate
- explicit clipping saturates
- model-compilation conversion into canonical scalar types saturates

These saturating operations are applied either once per output element (narrowing) or with a fixed operand pairing (elementwise ops), so order-dependence cannot arise from them.

**Wraparound determinism clause.** Where accumulation wraparound is reachable, the result is still exactly specified: every conforming backend computes the identical wrapped bit pattern. The protocol's security property is bit-equality of committed canonical values; wraparound therefore cannot create divergence between honest conforming executors. Wraparound reachability per reduction site:

- **Linear projections (weights from the canonical artifact):** wraparound is statically impossible. The conversion-time overflow bound (below) guarantees `|Σ products| < 2^62` for every representable activation input. (Informative: this means even adversarially chosen activations cannot wrap a weight-bounded reduction.)
- **Attention weighted-value aggregation:** wraparound is structurally impossible when weights are canonical `attention_softmax` outputs: the weights are non-negative and sum to exactly `1.0` in Q16.16 (`Σ w_bits = 2^16`), so `|Σ w_i · v_i| ≤ 2^16 · 2^31 = 2^47 < 2^62`.
- **Attention score (q·k):** both operands are runtime activations, so no static bound exists; with extreme adversarially-constructed activations the accumulation can wrap. Honest activations under RMS normalization do not approach this regime. If wraparound occurs, the wrapped result is canonical and identical everywhere; downstream `requantize` saturation bounds the materialized score.

**Softmax exponent-term summation: saturating, retained.** `acc_add_sat` over softmax exponent terms remains the canonical operation. This does not reintroduce order-dependence: all exponent terms are non-negative by construction, and saturating summation of non-negative terms is order-independent — partial sums are monotonically non-decreasing under every ordering, so either no ordering reaches the ceiling (all orderings compute the exact sum) or every ordering reaches and remains at the ceiling. The term set, not the order, determines the result. (Informative: this site may therefore also be parallelized freely.)

### 4. Requantization

Unchanged from v0.

- `requantize(x: Acc) -> Act`
- narrowing from `Acc` to `Act` is always explicit
- rounding: round to nearest, ties to even
- narrowing uses saturating conversion
- no implicit casts in canonical kernels
- `clip_act(x)` remains intentionally identical to `requantize(x)`
- `requantize` is applied exactly once per reduction output, to the final combined accumulator

### 5. Right-shift semantics

Unchanged from v0. All scaling right shifts route through canonical helpers (`rshift_round_ties_even`); signed shifts preserve sign; raw `>>` is forbidden in model kernels outside `det_num`; `rshift_round_ties_even(x, shift)` panics for `shift >= 64`.

### 6. Comparison semantics

Unchanged from v0. Exact comparison on fixed-point encoded values. `argmax_first`: larger value wins; equal values resolve to lowest index; panics on empty slices. The tie-break rule is part of the contract.

### 7. RoPE semantics

Unchanged from v0 in full. (Informative: RoPE rotation is elementwise per pair with fixed operand pairing — no reduction — so it was never order-sensitive; per-(position, dimension, base) sin/cos values are pure canonical function outputs and may be precomputed and cached without affecting conformance.)

### 8. Attention score semantics

Rule:

- `attention_score(query: &[Act], key: &[Act]) -> Act`

Semantics:

- `query` and `key` must be non-empty and have matching widths
- the term set is `{(query[i], key[i]) : i in 0..width}`; each term uses canonical widening multiply
- accumulation uses **wrapping `Acc`** per §2/§3; grouping and ordering of terms are not part of the contract
- score materialization is explicit and uses canonical `requantize` on the final combined accumulator
- no host-float dot-product fallback is allowed on the deterministic path

### 9. Attention softmax semantics

Unchanged from v0 except the summation note in §3:

- maximum logit selected with canonical `argmax_first` (lowest index on ties)
- exponent terms computed from `logit - max_logit`; largest shifted logit is exactly zero
- shifted logits interpreted canonically in Q32.32 before exponent materialization
- exponent materialization uses deterministic range reduction by `ln(2)` then a fixed polynomial; no host `exp`
- exponent terms below the explicit underflow floor clamp to zero
- exponent-term summation uses `acc_add_sat` (order-independent over non-negative terms; see §3)
- normalization divides each exponent by the canonical sum with ties-to-even rounding
- the residual making weights sum to exactly `1.0` in Q16.16 is assigned to the winning `argmax_first` index

### 10. Attention weighted-value aggregation semantics

Rule:

- `attention_weighted_sum(weights: &[Act], value_rows: &[Vec<Act>]) -> Vec<Act>`

Semantics:

- `weights` and `value_rows` must be non-empty with matching row counts; all value rows share one width
- for output dimension `d`, the term set is `{(weights[r], value_rows[r][d]) : r in 0..rows}`; each multiply uses canonical widening precision
- accumulation uses **wrapping `Acc`** per §2/§3; grouping and ordering are not part of the contract
- each output dimension is materialized with canonical `requantize` on its final combined accumulator

### 11. Deterministic tanh semantics

Unchanged from v0 in full, including the fixed five-step evaluation order. (Informative: fixed evaluation orders for scalar polynomial/rational pipelines are not reduction orders; they remain contract because each step's saturating intermediate is semantically load-bearing.)

### 12. Deterministic logits softcap semantics

Unchanged from v0 in full, including the fixed three-step evaluation order and the `input == 0 -> 0` requirement.

### 13. Deterministic GELU(tanh) semantics

Unchanged from v0 in full, including the canonical constants (`0.5 -> 32768`, `sqrt(2/pi) -> 52290`, `0.044715 -> 2930`) and the fixed nine-step evaluation order.

---

## Parallelism legality

This section defines which execution schedules are conforming. It is normative.

1. **Across-output parallelism: always conforming.** Distinct output elements (output rows of a projection, attention outputs per (head, query), normalized rows, softmax rows, elementwise results) are independent canonical computations over read-only shared inputs. They may be computed concurrently in any order provided each result is placed at its specified output coordinate.

2. **Within-reduction parallelism: conforming iff the accumulation is order-independent under this spec.** Under v1 this holds for: wrapping `Acc` MAC reductions (§2), and saturating summation of provably non-negative term sets (§3, softmax sum). It does not hold for any saturating fold over sign-mixed terms; no such fold exists in the v1 canonical kernel set, and none may be introduced without a spec revision.

3. **Combination operations are contract.** Partial results of a split reduction must be combined with the reduction's own canonical accumulation operation (wrapping `Acc` add for MAC reductions; `acc_add_sat` for the softmax sum). Converting through any other type or operation during combination is non-conforming.

4. **Guest profile.** The zkVM guest executes the serial reference schedule. Canonical kernel functions (the `det_num` module) must remain free of parallelism constructs, SIMD intrinsics, and platform-conditional arithmetic; parallel and vectorized drivers live in native-only modules that invoke or reimplement the canonical semantics and are validated against the reference by the conformance requirements below.

5. **Schedule-dependence is the forbidden property.** Any implementation whose committed canonical values depend on thread count, lane width, chunk size, scheduling order, or hardware backend is non-conforming, regardless of how the dependence arises.

Enforcement (wired): the `parallelism_lint` test denies parallelism constructs in guest-profile sources (`det_num`, `raster_kernels`, `routines/*/raster`), and the `schedule_parity` test asserts identical det checkpoint commitments for the full prefill + decode pipeline across serial and 2/4/8-thread schedules.

---

## Canonical source conversion rules

### Overview

Unchanged framing from v0: `det_num` defines a canonical conversion from source FP32 weights into executable `Wgt` tensors; the FP32 model is an upstream artifact only; the canonical executable artifact stores weights as `Wgt` values.

### FP32 -> `Wgt` conversion

Unchanged from v0:

- `f32_to_wgt(x: f32) -> Wgt`
- `q = sat_i32(round_ties_even(x * 65536.0))`; `Wgt(q)` represents `q / 65536`

### Conversion-time overflow bound (new in v1, normative)

For every weight tensor used in MAC reductions, for every output row `r` with columns `i in 0..cols`:

```
sum_i |wgt_bits[r][i]| * A_MAX < 2^62        where A_MAX = 2^31
```

equivalently:

```
sum_i |wgt_bits[r][i]| < 2^31
```

- **Scope.** The bound applies to weight tensors whose last-dimension rows are MAC reduction rows: tensors of rank >= 2. Rank-1 tensors (RMSNorm gains, per-layer scalars) are elementwise operands under saturating `Act`-domain ops and never enter a MAC reduction; the compiler records their computed mass in artifact metadata for audit but MUST NOT enforce the bound against them. (Informative: Gemma-family norm gains legitimately carry whole-vector masses above `2^31`.)
- `A_MAX = 2^31` bounds the magnitude of every representable `Act` payload, so satisfying this bound guarantees the reduction's exact integer sum stays below `2^62` for **any** activation input, representable or adversarial. The `2^62` ceiling (vs. the `2^63` wrap point) is a deliberate 2x margin and is itself normative.
- The model compiler MUST evaluate this bound per row and MUST fail closed (refuse to emit the artifact) if any row violates it. Splitting, rescaling, or otherwise altering weights to pass the bound is a model-design decision outside this spec; the compiler must not silently modify values.
- The compiler MUST record per-tensor maximum row mass (`max_r sum_i |wgt_bits[r][i]|`) in artifact metadata for audit.
- Worked example (informative): Q16.16 weights with `|w| < 1` have `|wgt_bits| < 2^16`; a 4096-wide row gives row mass `< 2^28`, passing with 8x margin; a 16384-wide MLP row with all `|w| = 1` gives `2^30`, passing with 2x margin. Real trained-LLM rows sit far below these worst cases.

### v1 storage policy

v1 retains direct canonical Q16.16 weight storage as the baseline and continues to exclude per-tensor/per-channel external scales, block quantization metadata, and value-changing compressed encodings.

**v1 narrows the v0 encoding prohibition:** artifact format versions MAY define width-reduced storage of `Wgt` payloads (e.g., 16-bit two's-complement storage for tensors whose every value is representable in 16 bits), PROVIDED the stored encoding is exact and value-preserving: the canonical value of each weight is its sign-extended integer payload, all arithmetic occurs after widening to the canonical type, and the resulting products and reductions are bit-identical to direct Q16.16 storage. Storage width is representation, not semantics, and does not by itself require a `det_num` spec revision. The concrete layout, width tags, and alignment rules belong to the weight-artifact format spec (detwgt), not this document.

### Canonical model artifact

As v0, with one addition: the artifact MUST carry `det_num_spec_version`, and a runtime pinned to one spec version MUST fail closed when loading an artifact declaring another. The artifact must not require host floating-point interpretation during inference.

---

## Serialization rules

Unchanged from v0 in full: little-endian, canonical fixed-width byte encoding, explicit byte conversion, no raw memory reinterpretation as the canonical serialization format, all checkpoint/state hashes derive from canonical serialized bytes. `Wgt`/`Act` serialize as 4-byte signed little-endian; `Acc` as 8-byte signed little-endian. (Informative: in-memory layout may be flat, strided, mmap-backed, or width-reduced per the artifact format; serialization for hashing always produces these canonical bytes in the specified element order.)

---

## Forbidden behavior

Forbidden in canonical arithmetic code:

- native floating-point as trusted semantics
- implicit narrowing conversions
- implicit or unspecified overflow behavior (v1: `Acc` MAC wrapping and the listed saturating sites are the only specified overflow behaviors; anything else is unspecified and forbidden)
- direct use of transcendental host math (`exp`, `sin`, `cos`, `tanh`, etc.)
- schedule-dependent results (any dependence of canonical values on execution order, thread count, lane width, or backend)
- combining split-reduction partials with any operation other than the reduction's canonical accumulation
- platform-dependent serialization
- unordered or unstable argmax behavior

Additionally forbidden in the executable inference path:

- reading FP32 weights as runtime arithmetic truth
- load-time weight interpretation that changes canonical numeric meaning
- weight encodings that are not exact value-preserving representations of canonical `Wgt` payloads

Removed from the v0 list: the blanket prohibition on wrapping arithmetic (now the canonical MAC accumulation behavior) and the blanket prohibition on non-left-to-right reduction order (now free for the order-independent reductions defined above).

---

## Required wrapper API

As v0, with semantics updated per this spec:

- `type Act`, `type Wgt`, `type Acc`
- `fn mul_wide(a: Act, b: Wgt) -> Acc`
- `fn mac(acc: Acc, a: Act, b: Wgt) -> Acc` — wrapping accumulation per §2
- `fn mac_bits(acc_bits: i64, act_bits: i32, wgt_bits: i32) -> i64` — promoted from helper to required API: it is the shared bit-level MAC used by both native and raster kernels and is the precise locus of the v0->v1 semantic change
- `fn requantize(x: Acc) -> Act`
- `fn clip_act(x: Acc) -> Act`
- `fn argmax_first(xs: &[Act]) -> usize`
- `fn rope_rotate_pairs(...)`, `fn attention_score(...)`, `fn attention_softmax(...)`, `fn attention_weighted_sum(...)`, `fn tanh_act(...)`, `fn gelu_pytorch_tanh_act(...)`
- `fn f32_to_wgt(x: f32) -> Wgt`

Recommended helpers unchanged (`add_sat`, `sub_sat`, `acc_add_sat`, `rshift_round_ties_even`, `*_to_le_bytes`), plus recommended for v1:

- `fn acc_combine(a: Acc, b: Acc) -> Acc` — wrapping combination of partial MAC accumulators, for split-reduction drivers; semantically identical to adding a partial's bits with wrapping `i64` addition

---

## Testing requirements

All v0 golden scalar, weight-conversion, serialization, and cross-build tests carry forward, with saturating-MAC edge tests replaced by the following.

### Wrapping MAC tests

- golden vectors pinning exact wrapped bit patterns for accumulations engineered to cross `i64::MAX` and `i64::MIN`, including sign-mixed term sets
- confirmation that `mul_wide` is exact for extreme `i32` payload pairs

### Associativity property tests (new, permanent conformance fixtures)

For randomized and adversarial term sets — including sets engineered to wrap, sets with all extreme-magnitude payloads, and every reduction length tail class relevant to vector lane widths — assert identical final accumulator bits across at least:

- left-to-right serial fold (reference)
- reversed-order fold
- random-permutation fold
- balanced pairwise tree reduction
- chunked partial accumulators combined with `acc_combine`, for multiple chunk sizes

These tests are the executable form of §2's contract and must run in CI permanently.

### Softmax-sum order-independence tests

- property test asserting `acc_add_sat` folds over non-negative term sets yield identical results under permutation and chunked combination, including saturating cases

### Conversion bound tests

- converter rejects a synthetic tensor violating the row-mass bound; accepts one at the boundary minus one
- per-tensor max row mass appears in artifact metadata and matches an independent recomputation
- reference-model conversion succeeds with the margin report committed

### Cross-schedule end-to-end tests

- full reference-model deterministic prefill + decode under: serial reference, multicore output-parallel, and (when present) vectorized backends — identical canonical checkpoint commitments at every checkpoint (implemented: `tests/schedule_parity.rs`)
- debug vs release parity; x86 vs ARM parity

### Migration validation

- one-time check: regenerate end-to-end golden checkpoint commitments under v1 and compare to v0 goldens for honest reference traffic. Expected result: identical (honest traffic should never have engaged MAC saturation). If any commitment differs, the divergence indicates honest-path saturation under v0 and MUST be investigated and documented before v1 ships.

---

## Versioning

This spec is versioned as:

- `det_num_spec_version = 1`

Runtime, weight artifacts, and any serialized checkpoint metadata embedding a spec version must agree; mismatches fail closed. There is no dual-version runtime support.

Any future change to type aliases, numeric interpretation, source conversion rule, rounding mode, overflow policy (either domain), shift policy, tie-breaking, serialization, the conversion-time bound, parallelism legality, or the executable weight artifact contract must increment the spec version.

---

## Summary

`det_num` v1 uses:

- `Act`/`Wgt`: signed 32-bit Q16.16; `Acc`: signed 64-bit Q32.32
- widening multiply always; exact products
- **wrapping `Acc` accumulation for all MAC reductions; reduction schedule is not contract**
- saturating arithmetic at materialization and in `Act`-domain elementwise ops, unchanged
- explicit requantization only; round-to-nearest, ties-to-even
- a normative conversion-time row-mass bound making linear-layer wraparound statically impossible
- wraparound, where reachable, fully deterministic and identical across backends
- explicit parallelism legality: free across outputs; free within order-independent reductions; serial reference preserved as oracle and guest profile
- little-endian canonical serialization; argmax ties to lowest index
- direct canonical Q16.16 weight storage, with exact width-reduced storage pre-authorized for future artifact format versions
- deterministic-mode outputs commit canonical bytes only
