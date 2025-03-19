use rand::RngCore;
use std::{
    fs::File,
    io::{self, Read},
};

pub const CHANCE_MUTATE_CONNECTION: f64 = 0.03;
pub const CHANCE_MUTATE_BISECTION: f64 = 0.05;
pub const CHANCE_MUTATE_WEIGHT: f64 = 0.8;
pub const CHANCE_PERTURB_WEIGHT: f64 = 0.9;
pub const CHANCE_NEW_WEIGHT: f64 = 0.1;
pub const CHANCE_NEW_DISABLED: f64 = 0.01;
pub const CHANCE_KEEP_DISABLED: f64 = 0.75;
pub const CHANCE_PICK_L_EQ: f64 = 0.5;
pub const CHANCE_PICK_L_NE: f64 = 0.5;

#[derive(Debug, Clone, Copy)]
pub enum EvolutionEvent {
    MutateConnection,
    MutateBisection,
    MutateWeight,
    NewWeight,
    PerturbWeight,
    NewDisabled,
    KeepDisabled,
    PickLEQ,
    PickLNE,
}

const fn percent(x: u64) -> u64 {
    x * (u64::MAX / 100)
}

pub const fn default_probabilities(evt: EvolutionEvent) -> u64 {
    match evt {
        EvolutionEvent::MutateConnection => percent(3),
        EvolutionEvent::MutateBisection => percent(5),
        EvolutionEvent::MutateWeight => percent(80),
        EvolutionEvent::PerturbWeight => percent(90),
        EvolutionEvent::NewWeight => percent(10),
        EvolutionEvent::NewDisabled => percent(1),
        EvolutionEvent::KeepDisabled => percent(75),
        EvolutionEvent::PickLEQ => percent(50),
        EvolutionEvent::PickLNE => percent(50),
    }
}

pub fn with_probabilities<P: Fn(EvolutionEvent) -> u64, R: FnMut() -> u64>(
    prob: P,
    rng: impl Fn() -> R,
) -> impl FnMut(EvolutionEvent) -> bool {
    let mut next_u64 = rng();
    move |evt| prob(evt) > next_u64()
}

pub fn seed_urandom() -> io::Result<u64> {
    let mut file = File::open("/dev/urandom")?;
    let mut buffer = [0u8; 8];
    file.read_exact(&mut buffer)?;
    Ok(u64::from_le_bytes([
        buffer[0], buffer[1], buffer[2], buffer[3], buffer[4], buffer[5], buffer[6], buffer[7],
    ]))
}

pub fn rng_rngcore(rng: impl RngCore) -> impl FnMut() -> u64 {
    let mut rng = rng;
    move || rng.next_u64()
}

pub fn rng_wyhash(seed: u64) -> impl FnMut() -> u64 {
    const WY_CONST_0: u64 = 0x2d35_8dcc_aa6c_78a5;
    const WY_CONST_1: u64 = 0x8bb8_4b93_962e_acc9;
    let mut state = seed;
    move || {
        state = state.wrapping_add(WY_CONST_0);
        let t = u128::from(state) * u128::from(state ^ WY_CONST_1);
        (t as u64) ^ (t >> 64) as u64
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use core::iter::once;
    use rand::rng;

    fn assert_within_deviation(
        evt: EvolutionEvent,
        chance: f64,
        range: f64,
        mut happens: impl FnMut(EvolutionEvent) -> bool,
    ) {
        let samples = 10_000.;
        let expected = chance * samples;
        let max_deviation = expected * range;
        for _ in 0..100 {
            let incidence = once(())
                .cycle()
                .take(samples as usize)
                .filter(|()| happens(evt))
                .count() as f64;
            assert!(
                (expected - incidence).abs() < max_deviation,
                "{evt:?}: {incidence} != {expected} ± {max_deviation}"
            );
        }
    }

    #[test]
    // baseline test of sorts
    fn test_deviation_rand() {
        for (evt, chance) in [
            (EvolutionEvent::MutateConnection, CHANCE_MUTATE_CONNECTION),
            (EvolutionEvent::MutateBisection, CHANCE_MUTATE_BISECTION),
            (EvolutionEvent::MutateWeight, CHANCE_MUTATE_WEIGHT),
            (EvolutionEvent::PerturbWeight, CHANCE_PERTURB_WEIGHT),
            (EvolutionEvent::NewWeight, CHANCE_NEW_WEIGHT),
            (EvolutionEvent::NewDisabled, CHANCE_NEW_DISABLED),
            (EvolutionEvent::KeepDisabled, CHANCE_KEEP_DISABLED),
            (EvolutionEvent::PickLEQ, CHANCE_PICK_L_EQ),
            (EvolutionEvent::PickLNE, CHANCE_PICK_L_NE),
        ] {
            assert_within_deviation(
                evt,
                chance,
                0.33,
                with_probabilities(default_probabilities, || rng_rngcore(rng())),
            );
        }
    }

    #[test]
    fn test_deviation_wyrand() {
        let seed = seed_urandom().unwrap();
        for (evt, chance) in [
            (EvolutionEvent::MutateConnection, CHANCE_MUTATE_CONNECTION),
            (EvolutionEvent::MutateBisection, CHANCE_MUTATE_BISECTION),
            (EvolutionEvent::MutateWeight, CHANCE_MUTATE_WEIGHT),
            (EvolutionEvent::PerturbWeight, CHANCE_PERTURB_WEIGHT),
            (EvolutionEvent::NewWeight, CHANCE_NEW_WEIGHT),
            (EvolutionEvent::NewDisabled, CHANCE_NEW_DISABLED),
            (EvolutionEvent::KeepDisabled, CHANCE_KEEP_DISABLED),
            (EvolutionEvent::PickLEQ, CHANCE_PICK_L_EQ),
            (EvolutionEvent::PickLNE, CHANCE_PICK_L_NE),
        ] {
            assert_within_deviation(
                evt,
                chance,
                0.33,
                with_probabilities(default_probabilities, || rng_wyhash(seed)),
            );
        }
    }
}
