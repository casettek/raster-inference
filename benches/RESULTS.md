# Deterministic inference benchmark results

Protocol: `cargo bench --bench det_inference`, single-threaded (rayon pinned to
1 thread), synthetic detwgt fixture (hidden 64, 2 layers [sliding, full],
4 heads / 1 KV head, head_dim 16, ff 128, sliding_window 128).

Reference machine: Apple M2 Max, 32 GiB RAM, macOS 15.7.7, rustc 1.89.0,
release profile.

## Before (dual-track nested buffers, pre-refactor)

| Benchmark            | Time (median) | Throughput (median) |
|----------------------|---------------|---------------------|
| det_prefill/seq_128  | 34.164 ms     | 3.747 K tok/s       |
| det_prefill/seq_512  | 298.83 ms     | 1.713 K tok/s       |
| det_prefill/seq_2048 | 3.3852 s      | 605.0 tok/s         |
| det_decode/ctx_128   | 777.93 µs     | 1.286 K tok/s       |
| det_decode/ctx_1024  | 3.2767 ms     | 305.2 tok/s         |

## After (flat slabs, clone elimination, single-track det mode, mmap weight views)

| Benchmark            | Time (median) | Throughput (median) | Speedup |
|----------------------|---------------|---------------------|---------|
| det_prefill/seq_128  | 10.571 ms     | 12.11 K tok/s       | 3.23x   |
| det_prefill/seq_512  | 61.390 ms     | 8.340 K tok/s       | 4.87x   |
| det_prefill/seq_2048 | 517.34 ms     | 3.959 K tok/s       | 6.54x   |
| det_decode/ctx_128   | 135.19 µs     | 7.397 K tok/s       | 5.75x   |
| det_decode/ctx_1024  | 295.35 µs     | 3.386 K tok/s       | 11.09x  |

Target was ≥ 2x on prefill and decode; all configurations exceed it. The
decode advantage grows with context length because the legacy path cloned
every layer's KV cache per generated token (O(context) per step), while the
flat-slab path appends in place and borrows attention windows.

Zero-copy mmap weight views (aligned payloads only; detwgt v1 alignment is
incidental, detwgt v2 guarantees it) cost ~1–5% on this tiny fixture
relative to owned copies while eliminating the load-time full-model copy.

## SIMD MAC kernels (NEON) + detwgt v2 i16 weight tiles

Protocol additions:

- `cargo bench --bench det_gemv` — single-thread MAC-kernel microbenchmark,
  4096-wide dots; "hot" = 8 MiB of weights (cache-resident, compute-bound),
  "stream" = 128 MiB of weights (decode-like, streams past the caches).
  `scalar_ref` is the canonical `mac_bits` fold compiled in release mode;
  `dispatch_*` is the runtime-dispatched kernel (NEON on this machine).
- `cargo bench --bench det_inference` in three configurations:
  scalar (`RASTER_DET_KERNEL_BACKEND=scalar`, all-i32 fixture), SIMD-i32
  (`RASTER_BENCH_WGT_WIDTH=i32`), SIMD-i16 (auto width; the fixture's matrix
  tensors all fit i16).

### Kernel-level (det_gemv, single thread)

| Benchmark                       | Time (median) | Throughput (median) |
|---------------------------------|---------------|---------------------|
| det_gemv_hot/scalar_ref_i32     | 149.7 µs      | 14.01 G MAC/s       |
| det_gemv_hot/dispatch_i32       | 153.1 µs      | 13.70 G MAC/s       |
| det_gemv_hot/dispatch_i16       | 122.7 µs      | 17.09 G MAC/s       |
| det_gemv_stream/scalar_ref_i32  | 2.638 ms      | 12.72 G MAC/s       |
| det_gemv_stream/dispatch_i32    | 2.616 ms      | 12.83 G MAC/s       |
| det_gemv_stream/dispatch_i16    | 2.084 ms      | 16.10 G MAC/s       |

### End-to-end (det_inference, single thread, tiny synthetic fixture)

| Benchmark            | scalar        | SIMD-i32      | SIMD-i16      |
|----------------------|---------------|---------------|---------------|
| det_prefill/seq_128  | 8.649 ms      | 8.611 ms      | 8.783 ms      |
| det_prefill/seq_512  | 55.47 ms      | 53.70 ms      | 55.18 ms      |
| det_prefill/seq_2048 | 480.0 ms      | 459.5 ms      | 485.7 ms      |
| det_decode/ctx_128   | 110.3 µs      | 108.0 µs      | 113.8 µs      |
| det_decode/ctx_1024  | 258.8 µs      | 239.6 µs      | 244.3 µs      |

### Targets vs. achieved (Workstream 4)

- **"SIMD ≥ 3x single-thread GEMV over scalar": not met as stated, and the
  target's premise does not hold on this toolchain.** The wrapping i64 MAC
  reduction is associative, so LLVM auto-vectorizes the canonical scalar
  `mac_bits` fold in release builds — the "scalar" baseline is already NEON
  code. The explicit kernel matches it on i32 (parity at ~14 G MAC/s,
  ~2 MACs/cycle/lane-pair at the `smlal` issue limit) and is the insertion
  point for the i16 path, which the auto-vectorizer cannot derive (it
  requires the format-level width tag). Force-scalar mode
  (`RASTER_DET_KERNEL_BACKEND=scalar`) remains the canonical oracle; in
  debug builds (no auto-vectorization) the dispatched kernel is the only
  vectorized path.
- **i16 ≥ 1.7x decode at saturated threads: not measurable on this
  fixture.** The synthetic model (~200 KiB) is cache-resident, so i16's
  bandwidth halving cannot show: the kernel-level streaming regime shows
  1.26x single-thread (8.4 GB/s i32-equivalent bytes saved), and the e2e
  numbers above are flat-to-slightly-slower because the tiny 64–128-wide
  GEMV rows amortize nothing. The 1.7x claim is about the bandwidth-bound
  regime: real-model decode at saturated threads, where decode tok/s ≈
  bandwidth ÷ model bytes. Measure it against a real converted model with
  `RASTER_BENCH_MODEL_DIR=<dir> cargo bench --bench det_inference`
  (all-i32 vs auto-width conversions of the same model).
- **Bandwidth fraction** (`model_bytes × tok/s ÷ STREAM bandwidth`):
  single-thread streaming GEMV moves 51.3 GB/s (i32) / 32.2 GB/s (i16) of
  weight bytes — ~13% / 8% of the M2 Max's nominal 400 GB/s chip bandwidth,
  i.e. one core cannot saturate the memory system and the rayon row-parallel
  driver is what closes the gap at full thread count. Record the
  saturated-thread fraction when benchmarking a real model.
- **i16 storage ratio:** the converter prints the per-model i16 summary on
  every run (`i16 width report: ...`); record the real-Gemma ratio here when
  the production model is re-converted. Q16.16 LLM weights with |w| < 0.5
  qualify, which is expected to cover ≥ 90% of tensor bytes.
