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
incidental, detwgt v2 will guarantee it) cost ~1–5% on this tiny fixture
relative to owned copies while eliminating the load-time full-model copy.
