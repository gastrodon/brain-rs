use brain::{random::default_rng, specie::InnoGen};
use criterion::Criterion;
use rand::Rng;

fn bench(bench: &mut Criterion) {
    let mut rng = default_rng();
    let mut inno = InnoGen::new(0);
    bench.bench_function("innogen", |b| {
        b.iter(|| inno.path((rng.random_range(0..=10_000), rng.random_range(0..=10_000))))
    });
}

pub fn benches() {
    #[cfg(not(feature = "smol_bench"))]
    let mut criterion: criterion::Criterion<_> = Criterion::default()
        .sample_size(2000)
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
    bench(&mut criterion);
}

fn main() {
    benches();
    criterion::Criterion::default()
        .configure_from_args()
        .final_summary();
}
