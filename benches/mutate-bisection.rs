use brain::{random::default_rng, specie::InnoGen, Genome};
use criterion::Criterion;

fn bench(bench: &mut Criterion) {
    let genome = Genome::from_str(include_str!("data/genome-rand-100.json")).unwrap();
    let mut rng = default_rng();
    bench.bench_function("mutate-bisection", |b| {
        b.iter(|| {
            genome
                .clone()
                .mutate_bisection(&mut rng, &mut InnoGen::new(300))
                .unwrap()
        })
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
