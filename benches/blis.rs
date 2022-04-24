use blis::*;
use criterion::{criterion_group, criterion_main, Criterion};

pub fn criterion_benchmark(criterion: &mut Criterion) {
    let mut mnks = vec![];

    mnks.push((1, 1, 1));
    for s in [16, 32, 64, 256] {
        mnks.push((s, 1, s));
    }
    for s in [16, 32, 64, 256] {
        mnks.push((1, s, s));
    }
    for s in [16, 32, 64, 256] {
        mnks.push((s, s, 1));
    }
    for s in [16, 32, 64, 256] {
        mnks.push((s, s, s));
    }

    for (m, n, k) in mnks.iter().copied() {
        let a = vec![0.0; m * k];
        let b = vec![0.0; k * n];
        let mut c = vec![0.0; m * n];

        let a = MatrixRef::try_from_slice(&a, m, k, 1, m).unwrap();
        let b = MatrixRef::try_from_slice(&b, k, n, 1, k).unwrap();
        let mut c = MatrixMut::try_from_mut_slice(&mut c, m, n, 1, m).unwrap();

        criterion.bench_function(&format!("gemm {}×{}×{}", m, n, k), |bench| {
            bench.iter(|| f64::gemm(c.rb_mut(), a, b, 1.0, 0.0, 0));
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
