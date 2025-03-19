use brain::random::{rng_rngcore, rng_wyhash};
use criterion::Criterion;
use rand::rng;
use std::{fs::File, io::Read};

fn bench_rngcore(bench: &mut Criterion) {
    bench.bench_function("random-rngcore", |b| {
        let next_u64 = rng_rngcore(rng());
        b.iter(next_u64);
    });
}

fn bench_wyhash(bench: &mut Criterion) {
    let seed = {
        let mut file = File::open("/dev/urandom").unwrap();
        let mut buffer = [0u8; 8];
        file.read_exact(&mut buffer).unwrap();
        u64::from_le_bytes([
            buffer[0], buffer[1], buffer[2], buffer[3], buffer[4], buffer[5], buffer[6], buffer[7],
        ])
    };

    bench.bench_function("random-wyhash", |b| {
        let next_u64 = rng_wyhash(seed);
        b.iter(next_u64);
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
    bench_rngcore(&mut criterion);
    bench_wyhash(&mut criterion);
}

fn main() {
    benches();
    criterion::Criterion::default()
        .configure_from_args()
        .final_summary();
}
