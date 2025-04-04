#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]

use brain::{
    activate::relu,
    genome::{node::NonBNode, CTRGenome, Genome, WConnection},
    network::{loss::decay_quadratic, ToNetwork},
    random::{
        default_rng, percent, EvolutionEvent, Happens, ProbBinding, ProbStatic, Probabilities,
    },
    scenario::{evolve, EvolutionHooks},
    specie::population_init,
    Connection, Ctrnn, Network, Node, Scenario, Stats,
};
use core::{f64, ops::ControlFlow};
use rand::RngCore;

const POPULATION: usize = 100;

struct Xor;

impl<
        N: Node,
        C: Connection<N>,
        G: Genome<N, C> + ToNetwork<Ctrnn, N, C>,
        H: RngCore + Probabilities + Happens,
        A: Fn(f64) -> f64,
    > Scenario<N, C, G, H, A> for Xor
{
    fn io(&self) -> (usize, usize) {
        (2, 1)
    }

    fn eval(&self, genome: &G, σ: &A) -> f64 {
        let mut network = genome.network();
        let mut fit = 0.;
        network.step(2, &[0., 0.], σ);
        fit += decay_quadratic(1., network.output()[0]);
        network.flush();

        network.step(2, &[1., 1.], σ);
        fit += decay_quadratic(1., network.output()[0]);
        network.flush();

        network.step(2, &[0., 1.], σ);
        fit += decay_quadratic(0., network.output()[0]);
        network.flush();

        network.step(2, &[1., 0.], σ);
        fit += decay_quadratic(0., network.output()[0]);

        fit / 4.
    }
}

fn hook<
    N: Node,
    C: Connection<N>,
    G: Genome<N, C>,
    H: RngCore + Probabilities<Update = (brain::random::EvolutionEvent, u64)> + Happens,
>(
    stats: &mut Stats<'_, N, C, G, H>,
) -> ControlFlow<()> {
    if stats.generation % 10 == 1 {
        let (_, f) = stats.fittest().unwrap();
        println!("fittest of gen {}: {:.4}", stats.generation, f);
    }

    if stats.any_fitter_than(0.749999) {
        let fittest = stats.fittest().unwrap();
        println!("target met in gen {}: {:.4}", stats.generation, fittest.1);
        fittest
            .0
            .to_file(format!("output/xor-{}.json", stats.generation))
            .unwrap();

        return ControlFlow::Break(());
    }

    if stats.generation == 100 {
        stats
            .rng
            .update((EvolutionEvent::MutateConnection, percent(35)));
        stats
            .rng
            .update((EvolutionEvent::MutateBisection, percent(35)));
    }

    ControlFlow::Continue(())
}

type N = NonBNode;
type C = WConnection<NonBNode>;
type G = CTRGenome<NonBNode, WConnection<NonBNode>>;

fn main() {
    evolve(
        Xor {},
        |(i, o)| population_init::<N, C, G>(i, o, POPULATION),
        relu,
        ProbBinding::new(ProbStatic::default(), default_rng()),
        EvolutionHooks::new(vec![Box::new(hook)]),
    );
}
