use rand::{rngs::SmallRng, RngCore, SeedableRng};
use std::{
    cmp::min,
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

pub trait Probabilities {
    fn probability(&self, evt: EvolutionEvent) -> u64;
}

pub trait Happens {
    fn happens(&mut self, evt: EvolutionEvent) -> bool;
}

impl<T: RngCore + Probabilities> Happens for T {
    fn happens(&mut self, evt: EvolutionEvent) -> bool {
        self.probability(evt) > self.next_u64()
    }
}

pub struct ProbStatic {
    mutate_connection: u64,
    mutate_bisection: u64,
    mutate_weight: u64,
    perturb_weight: u64,
    new_weight: u64,
    new_disabled: u64,
    keep_disabled: u64,
    pick_leq: u64,
    pick_lne: u64,
}

impl Default for ProbStatic {
    fn default() -> Self {
        Self {
            mutate_connection: percent(3),
            mutate_bisection: percent(5),
            mutate_weight: percent(80),
            perturb_weight: percent(90),
            new_weight: percent(10),
            new_disabled: percent(1),
            keep_disabled: percent(75),
            pick_leq: percent(50),
            pick_lne: percent(50),
        }
    }
}

impl Probabilities for ProbStatic {
    fn probability(&self, evt: EvolutionEvent) -> u64 {
        match evt {
            EvolutionEvent::MutateConnection => self.mutate_connection,
            EvolutionEvent::MutateBisection => self.mutate_bisection,
            EvolutionEvent::MutateWeight => self.mutate_weight,
            EvolutionEvent::PerturbWeight => self.perturb_weight,
            EvolutionEvent::NewWeight => self.new_weight,
            EvolutionEvent::NewDisabled => self.new_disabled,
            EvolutionEvent::KeepDisabled => self.keep_disabled,
            EvolutionEvent::PickLEQ => self.pick_leq,
            EvolutionEvent::PickLNE => self.pick_lne,
        }
    }
}

pub struct WyRng {
    state: u64,
}

impl WyRng {
    pub fn seeded(state: u64) -> Self {
        Self { state }
    }
}

impl RngCore for WyRng {
    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }

    fn next_u64(&mut self) -> u64 {
        const WY_CONST_0: u64 = 0x2d35_8dcc_aa6c_78a5;
        const WY_CONST_1: u64 = 0x8bb8_4b93_962e_acc9;
        self.state = self.state.wrapping_add(WY_CONST_0);
        let t = u128::from(self.state) * u128::from(self.state ^ WY_CONST_1);
        (t as u64) ^ (t >> 64) as u64
    }

    // TODO test this
    fn fill_bytes(&mut self, dst: &mut [u8]) {
        if dst.is_empty() {
            return;
        }

        let mut idx = 0;
        while idx < dst.len() {
            let lim = min(8, dst.len() - idx);
            dst.copy_from_slice(&self.next_u64().to_ne_bytes()[..lim]);
            idx += lim;
        }
    }
}

pub struct ProbBinding<P: Probabilities, R: RngCore> {
    p: P,
    r: R,
}

impl<P: Probabilities, R: RngCore> ProbBinding<P, R> {
    pub fn new(p: P, r: R) -> Self {
        Self { p, r }
    }

    #[allow(clippy::should_implement_trait)] // type signature is incompatible with trait Default
    pub fn default() -> ProbBinding<impl Probabilities, impl RngCore> {
        ProbBinding {
            p: ProbStatic::default(),
            r: default_rng(),
        }
    }
}

impl<P: Probabilities, R: RngCore> Probabilities for ProbBinding<P, R> {
    fn probability(&self, evt: EvolutionEvent) -> u64 {
        self.p.probability(evt)
    }
}

impl<P: Probabilities, R: RngCore> RngCore for ProbBinding<P, R> {
    fn next_u32(&mut self) -> u32 {
        self.r.next_u32()
    }

    fn next_u64(&mut self) -> u64 {
        self.r.next_u64()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.r.fill_bytes(dest)
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

pub fn default_rng() -> impl RngCore {
    SmallRng::seed_from_u64(seed_urandom().unwrap())
}

#[cfg(test)]
mod test {
    use super::*;
    use core::iter::once;
    use rand::rngs::ThreadRng;

    fn assert_within_deviation(
        evt: EvolutionEvent,
        chance: f64,
        range: f64,
        happens: &mut impl Happens,
    ) {
        let samples = 10_000.;
        let expected = chance * samples;
        let max_deviation = expected * range;
        for _ in 0..100 {
            let incidence = once(())
                .cycle()
                .take(samples as usize)
                .filter(|()| happens.happens(evt))
                .count() as f64;
            assert!(
                (expected - incidence).abs() < max_deviation,
                "{evt:?}: {incidence} != {expected} ± {max_deviation}"
            );
        }
    }

    // controll test - we are confident that rand generates good random numbers
    #[test]
    fn test_deviation_rand() {
        let mut p_bind = ProbBinding::new(ProbStatic::default(), ThreadRng::default());
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
            assert_within_deviation(evt, chance, 0.33, &mut p_bind);
        }
    }

    #[test]
    fn test_deviation_wyrand() {
        let mut p_bind = ProbBinding::new(
            ProbStatic::default(),
            WyRng::seeded(seed_urandom().unwrap()),
        );
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
            assert_within_deviation(evt, chance, 0.33, &mut p_bind);
        }
    }
}
