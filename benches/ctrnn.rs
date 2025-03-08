#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]

use brain::{activate::relu, Ctrnn, Network};
use criterion::Criterion;

fn bench(bench: &mut Criterion) {
    let net = &mut Ctrnn::from_str(include_str!("data/ctrnn-rand-100.json")).unwrap();
    let i = vec![0.7, 0.3];

    bench.bench_function("ctrnn-step", |b| b.iter(|| net.step(100, &i, relu)));
}

pub fn benches() {
    #[cfg(not(feature = "smol_bench"))]
    let mut criterion: criterion::Criterion<_> = Criterion::default()
        .sample_size(1000)
        .significance_level(0.1);
    #[cfg(feature = "smol_bench")]
    let mut criterion: criterion::Criterion<_> = {
        use std::time::Duration;
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
