use brain::{crossover::crossover, random::default_rng};
use core::cmp::Ordering;
use criterion::Criterion;

fn bench(bench: &mut Criterion) {
    let l_conn =
        serde_json::from_str::<Vec<_>>(include_str!("data/connection-rand-l.json")).unwrap();
    let r_conn =
        serde_json::from_str::<Vec<_>>(include_str!("data/connection-rand-r.json")).unwrap();

    bench.bench_function("crossover-eq", |b| {
        b.iter(|| crossover(&l_conn, &r_conn, Ordering::Equal, &mut default_rng()))
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
    bench(&mut criterion);
}

fn main() {
    benches();
    criterion::Criterion::default()
        .configure_from_args()
        .final_summary();
}
