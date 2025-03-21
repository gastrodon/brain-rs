use brain::random::{seed_urandom, WyRng};
use criterion::Criterion;
use rand::{
    rngs::{SmallRng, ThreadRng},
    RngCore, SeedableRng,
};

fn bench_threadrng(bench: &mut Criterion) {
    let mut rng = ThreadRng::default();

    bench.bench_function("random-threadrng", |b| {
        b.iter(|| rng.next_u64());
    });
}

fn bench_smallrng(bench: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(seed_urandom().unwrap());

    bench.bench_function("random-smallrng", |b| {
        b.iter(|| rng.next_u64());
    });
}

fn bench_wyhash(bench: &mut Criterion) {
    let mut rng = WyRng::seeded(seed_urandom().unwrap());

    bench.bench_function("random-wyhash", |b| {
        b.iter(|| rng.next_u64());
    });
}

pub fn benches() {
    #[cfg(not(feature = "smol_bench"))]
    let mut criterion: criterion::Criterion<_> = Criterion::default()
        .sample_size(1000)
        .significance_level(0.1);
    #[cfg(feature = "smol_bench")]
    let mut criterion: criterion::Criterion<_> = {
        use core::time::Duration;
        Criterion::default()
            .measurement_time(Duration::from_millis(1))
            .sample_size(10)
            .nresamples(1)
            .without_plots()
            .configure_from_args()
    };
    bench_threadrng(&mut criterion);
    bench_smallrng(&mut criterion);
    bench_wyhash(&mut criterion);
}

fn main() {
    benches();
    criterion::Criterion::default()
        .configure_from_args()
        .final_summary();
}
