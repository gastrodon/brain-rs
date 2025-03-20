use crate::{
    random::default_rng,
    specie::{population_reproduce, speciate, Specie, SpecieRepr},
    Genome,
};
use core::f64;
use std::collections::HashMap;

const NO_IMPROVEMENT_TRUNCATE: usize = 10;

pub enum EvolutionTarget {
    Fitness(f64),
    Generation(usize),
}

impl EvolutionTarget {
    fn satisfied(&self, species: &[Specie], generation: usize) -> bool {
        if let Self::Fitness(t) = self {
            species
                .iter()
                .any(|f| f.members.first().is_some_and(|(_, f)| f >= t))
        } else if let Self::Generation(t) = self {
            t <= &generation
        } else {
            false
        }
    }
}

pub trait Scenario {
    fn io(&self) -> (usize, usize);
    fn eval<F: Fn(f64) -> f64>(&mut self, genome: &Genome, σ: F) -> f64;

    fn evolve(
        &mut self,
        target: EvolutionTarget,
        init: impl FnOnce((usize, usize)) -> (Vec<Specie>, usize),
        population_lim: usize,
        σ: impl Fn(f64) -> f64,
    ) -> (Vec<Specie>, usize) {
        let (mut pop_flat, mut inno_head) = {
            let (species, inno_head) = init(self.io());
            (
                species
                    .iter()
                    .flat_map(|Specie { members, .. }| {
                        members.iter().map(|(genome, _)| genome.clone())
                    })
                    .collect::<Vec<_>>(),
                inno_head,
            )
        };

        let mut scores: HashMap<SpecieRepr, _> = HashMap::new();

        let mut rng = default_rng();
        let mut gen_idx = 0;
        loop {
            let species = {
                let genomes = pop_flat.into_iter().map(|genome| {
                    let fitness = self.eval(&genome, &σ);
                    (genome, fitness)
                });
                let reprs = scores.keys().cloned();

                #[cfg(not(feature = "smol_bench"))]
                let species = speciate(genomes, reprs);
                #[cfg(feature = "smol_bench")]
                let species = speciate(
                    genomes.collect::<Vec<_>>().into_iter(),
                    reprs.collect::<Vec<_>>().into_iter(),
                );
                species
            };

            if target.satisfied(&species, gen_idx) {
                break (species, inno_head);
            };

            let scores_prev = scores;
            scores = species
                .iter()
                .filter_map(|Specie { repr, members }| {
                    let gen_max = members
                        .iter()
                        .max_by(|(_, l), (_, r)| l.partial_cmp(r).unwrap());
                    let past_max = scores_prev.get(repr);

                    match (gen_max, past_max) {
                        (Some((_, gen_max)), Some((past_max, past_idx))) => {
                            if gen_max > past_max {
                                Some((repr.clone(), (*gen_max, gen_idx)))
                            } else {
                                Some((repr.clone(), (*past_max, *past_idx)))
                            }
                        }
                        (Some((_, gen_max)), None) => Some((repr.clone(), (*gen_max, gen_idx))),
                        (None, _) => None,
                    }
                })
                .collect();

            let p_scored = species
                .into_iter()
                .map(|s| {
                    let (min_fit, gen_achieved) =
                        *scores_prev.get(&s.repr).unwrap_or(&(f64::MIN, gen_idx));

                    if gen_achieved + NO_IMPROVEMENT_TRUNCATE <= gen_idx && s.members.len() > 2 {
                        (
                            Specie {
                                repr: s.repr,
                                members: {
                                    let mut trunc = s.members;
                                    trunc.sort_by(|(_, l), (_, r)| r.partial_cmp(l).unwrap());
                                    trunc[..2].to_vec()
                                },
                            },
                            f64::MIN,
                        )
                    } else {
                        (s, min_fit)
                    }
                })
                .collect::<Vec<_>>();

            (pop_flat, inno_head) =
                population_reproduce(&p_scored, population_lim, inno_head, &mut rng);

            debug_assert!(!pop_flat.is_empty(), "nobody past {gen_idx}");
            gen_idx += 1
        }
    }
}
