use std::iter;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group};

fn from_elem(c: &mut Criterion) {
    static KB: usize = 1024;

    let mut group = c.benchmark_group("from_elem");
    for size in [KB, 2 * KB, 4 * KB, 8 * KB, 16 * KB].iter() {
        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| iter::repeat_n(0u8, size).collect::<Vec<_>>());
        });
    }
    group.finish();

    let mut group = c.benchmark_group("from_elem_decimal");
    for size in [KB, 2 * KB].iter() {
        group.throughput(Throughput::BytesDecimal(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| iter::repeat_n(0u8, size).collect::<Vec<_>>());
        });
    }
    group.finish();
}

criterion_group!(benches, from_elem);
