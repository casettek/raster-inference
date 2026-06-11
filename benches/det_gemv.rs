//! Single-thread GEMV MAC-kernel microbenchmarks.
//!
//! Compares the canonical scalar reference reduction (`mac_bits` fold)
//! against the dispatched SIMD kernels (NEON on aarch64) for i32 and i16
//! weight storage. Throughput is reported in MAC elements/sec.
//!
//! Force the scalar backend with `RASTER_DET_KERNEL_BACKEND=scalar` to
//! confirm the dispatch overhead is negligible.
//!
//! Run: `cargo bench --bench det_gemv`
//! Record results in `benches/RESULTS.md`.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};

use raster_inference::shared::numerics::det_num::mac_bits;
use raster_inference::shared::numerics::det_simd;

const COLS: usize = 4096;
/// Cache-resident regime: 8 MiB of i32 weights (compute-bound).
const HOT_ROWS: usize = 512;
/// Streaming regime: 128 MiB of i32 weights, larger than the last-level
/// cache, matching decode's stream-the-model-per-token behavior
/// (bandwidth-bound; this is where i16 storage approaches 2x).
const STREAM_ROWS: usize = 8192;

fn fixture_i32(len: usize, seed: u64) -> Vec<i32> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 16) as i32
        })
        .collect()
}

fn bench_shape(criterion: &mut Criterion, group_name: &str, rows: usize) {
    let input = fixture_i32(COLS, 7);
    let weights_i32 = fixture_i32(rows * COLS, 11);
    let weights_i16 = weights_i32
        .iter()
        .map(|bits| (bits % i32::from(i16::MAX)) as i16)
        .collect::<Vec<_>>();

    let mut group = criterion.benchmark_group(group_name);
    group.throughput(Throughput::Elements((rows * COLS) as u64));

    group.bench_function("scalar_ref_i32", |bencher| {
        bencher.iter(|| {
            let mut checksum = 0_i64;
            for row in weights_i32.chunks_exact(COLS) {
                let mut acc = 0_i64;
                for (act, wgt) in input.iter().zip(row) {
                    acc = mac_bits(acc, *act, *wgt);
                }
                checksum = checksum.wrapping_add(acc);
            }
            checksum
        })
    });

    group.bench_function("dispatch_i32", |bencher| {
        bencher.iter(|| {
            let mut checksum = 0_i64;
            for row in weights_i32.chunks_exact(COLS) {
                checksum = checksum.wrapping_add(det_simd::dot_act_i32(&input, row));
            }
            checksum
        })
    });

    group.bench_function("dispatch_i16", |bencher| {
        bencher.iter(|| {
            let mut checksum = 0_i64;
            for row in weights_i16.chunks_exact(COLS) {
                checksum = checksum.wrapping_add(det_simd::dot_act_i16(&input, row));
            }
            checksum
        })
    });

    group.finish();
}

fn gemv(criterion: &mut Criterion) {
    bench_shape(criterion, "det_gemv_hot", HOT_ROWS);
    bench_shape(criterion, "det_gemv_stream", STREAM_ROWS);
}

criterion_group!(benches, gemv);
criterion_main!(benches);
