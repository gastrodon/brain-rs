use brain::{
    crossover::{avg_weight_diff, disjoint_excess_count},
    specie::speciate,
};
use core::iter::empty;
use criterion::Criterion;

fn bench_distance(bench: &mut Criterion) {
    let l_conn =
        serde_json::from_str::<Vec<_>>(include_str!("data/connection-rand-l.json")).unwrap();
    let r_conn =
        serde_json::from_str::<Vec<_>>(include_str!("data/connection-rand-r.json")).unwrap();

    bench.bench_function("disjoint-excess-count", |b| {
        b.iter(|| disjoint_excess_count(&l_conn, &r_conn))
    });

    bench.bench_function("avg-weight-diff", |b| {
        b.iter(|| avg_weight_diff(&l_conn, &r_conn))
    });
}

fn bench_speciate(bench: &mut Criterion) {
    let genomes = serde_json::from_str::<Vec<_>>(include_str!("data/genome-xor-100.json")).unwrap();
    bench.bench_function("speciate", |b| {
        b.iter(|| speciate(genomes.iter().cloned(), empty()))
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
    bench_distance(&mut criterion);
    bench_speciate(&mut criterion);
}

fn main() {
    benches();
    criterion::Criterion::default()
        .configure_from_args()
        .final_summary();
}
