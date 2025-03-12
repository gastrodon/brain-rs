#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]

use brain::{
    activate::relu, specie::population_init, Ctrnn, EvolutionTarget, Genome, Network, Scenario,
    Specie,
};

struct NesTetris;

impl Scenario for NesTetris {
    fn io(&self) -> (usize, usize) {
        (8, todo!()) // 8 controller buttons, ??? framedata ( board, next-tile? )
    }

    fn eval<F: Fn(f64) -> f64>(&self, genome: &Genome, σ: F) -> f64 {
        let mut network = Ctrnn::from_genome(genome);
        let mut fit = 0.;
        todo!()
    }
}
const POPULATION: usize = 100;

fn main() {
    let res = NesTetris {}.evolve(
        EvolutionTarget::Generation(10),
        |(i, o)| population_init(i, o, POPULATION),
        POPULATION,
        relu,
    );

    println!(
        "top score: {:?}",
        res.0
            .into_iter()
            .flat_map(|Specie { members, .. }| members)
            .max_by(|(_, l), (_, r)| l.partial_cmp(r).unwrap())
            .unwrap()
            .1
    );
}
