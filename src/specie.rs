use crate::{
    crossover::delta,
    genome::{Connection, Genome},
};
use core::{error::Error, f64};
use fxhash::FxHashMap;
use rand::rngs::ThreadRng;
use std::{
    collections::HashMap,
    hash::{DefaultHasher, Hash, Hasher},
};

pub struct InnoGen {
    pub head: usize,
    seen: FxHashMap<(usize, usize), usize>,
}

impl InnoGen {
    pub fn new(head: usize) -> Self {
        Self {
            head,
            seen: FxHashMap::default(),
        }
    }

    pub fn path(&mut self, v: (usize, usize)) -> usize {
        match self.seen.get(&v) {
            Some(n) => *n,
            None => {
                let n = self.head;
                self.head += 1;
                self.seen.insert(v, n);
                n
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpecieRepr(pub Vec<Connection>);

impl SpecieRepr {
    fn delta(&self, other: &[Connection]) -> f64 {
        delta(&self.0, other)
    }

    #[inline]
    fn cloned(&self) -> Vec<Connection> {
        self.0.to_vec()
    }
}

impl SpecieRepr {
    fn id(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.hash(&mut h);
        h.finish()
    }
}

impl Hash for SpecieRepr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl PartialEq for SpecieRepr {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl Eq for SpecieRepr {}

impl AsRef<[Connection]> for SpecieRepr {
    fn as_ref(&self) -> &[Connection] {
        &self.0
    }
}

#[derive(Debug)]
pub struct Specie {
    pub repr: SpecieRepr,
    pub members: Vec<(Genome, f64)>,
}

impl Specie {
    #[inline]
    pub fn len(&self) -> usize {
        self.members.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    #[inline]
    pub fn last(&self) -> Option<&(Genome, f64)> {
        self.members.last()
    }

    #[inline]
    pub fn cloned(&self) -> (Vec<Connection>, Vec<(Genome, f64)>) {
        (
            self.repr.cloned(),
            self.members
                .iter()
                .map(|(g, s)| ((*g).clone(), *s))
                .collect(),
        )
    }

    pub fn fit_adjusted(&self) -> f64 {
        let l = self.len() as f64;
        self.members.iter().fold(0., |acc, (_, fit)| acc + *fit / l)
    }
}

fn reproduce_crossover(
    genomes: &[(Genome, f64)],
    size: usize,
    rng: &mut ThreadRng,
    innogen: &mut InnoGen,
) -> Result<Vec<Genome>, Box<dyn Error>> {
    if size == 0 {
        return Ok(vec![]);
    }

    if genomes.len() < 2 {
        return Err(format!(
            "too few members to crossover (wanted to produce {size} from {}",
            genomes.len()
        )
        .into());
    }

    let pairs = {
        let mut pairs = genomes
            .iter()
            .enumerate()
            .flat_map(|(l_idx, (l, l_fit))| {
                genomes
                    .iter()
                    .enumerate()
                    .filter_map(move |(r_idx, (r, r_fit))| {
                        if l_fit > r_fit || (l_fit == r_fit && l_idx > r_idx) {
                            Some(((l, l_fit), (r, r_fit)))
                        } else {
                            None
                        }
                    })
            })
            .collect::<Vec<_>>();
        pairs.sort_by(|l, r| (r.0 .1 + r.1 .1).partial_cmp(&(l.0 .1 + l.1 .1)).unwrap());
        pairs
    };

    pairs
        .into_iter()
        .cycle()
        .take(size)
        .map(|((l, _), (r, _))| {
            let mut child = l.reproduce_with(r, std::cmp::Ordering::Greater, rng);
            child.maybe_mutate(rng, innogen)?;
            Ok(child)
        })
        .collect()
}

fn reproduce_copy(
    genomes: &[(Genome, f64)],
    size: usize,
    rng: &mut ThreadRng,
    innogen: &mut InnoGen,
) -> Result<Vec<Genome>, Box<dyn Error>> {
    if size == 0 {
        return Ok(vec![]);
    }

    if genomes.is_empty() {
        return Err(format!(
            "too few members to copy (wanted to produce {size} from {}",
            genomes.len()
        )
        .into());
    }

    let mut top = genomes.iter().collect::<Vec<_>>();
    top.sort_by(|(_, l), (_, r)| r.partial_cmp(l).unwrap());
    top.into_iter()
        .cycle()
        .take(size)
        .map(|(genome, _)| {
            let mut child = genome.clone();
            child.maybe_mutate(rng, innogen)?;
            Ok(child)
        })
        .collect()
}

pub fn reproduce(
    genomes: Vec<(Genome, f64)>,
    size: usize,
    innogen: &mut InnoGen,
    rng: &mut ThreadRng,
) -> Result<Vec<Genome>, Box<dyn Error>> {
    if size == 0 {
        return Ok(vec![]);
    }

    if genomes.is_empty() {
        return Err(format!(
            "too few members to reproduce (wanted to produce {size} from {}",
            genomes.len()
        )
        .into());
    }

    let mut pop: Vec<Genome> = Vec::with_capacity(size);
    pop.push(
        genomes
            .iter()
            .max_by(|(_, l), (_, r)| l.partial_cmp(r).unwrap())
            .unwrap()
            .0
            .clone(),
    );

    if size == 1 {
        return Ok(pop);
    }

    let size = size - 1;
    let size_copy = size / 4;
    let size_copy = if size_copy == 0 || genomes.len() == 1 {
        size
    } else {
        size_copy
    };

    // TODO reproduce_crossover and reproduce_copy can potentially be made faster
    // if they're handed a slice to write into intead of returning a vec that we then need to copy
    reproduce_copy(&genomes, size_copy, rng, innogen)?
        .into_iter()
        .for_each(|genome| pop.push(genome));

    let size_crossover = size - size_copy;
    reproduce_crossover(&genomes, size_crossover, rng, innogen)?
        .into_iter()
        .for_each(|genome| pop.push(genome));

    Ok(pop)
}

/// allocate a target population for every specie in an existing population
fn population_alloc<'a>(
    species: impl Iterator<Item = &'a Specie>,
    population: usize,
) -> HashMap<SpecieRepr, usize> {
    let species_fitted = species
        .map(|s| (s.repr.clone(), s.fit_adjusted()))
        .collect::<Vec<_>>();

    let fit_total = species_fitted.iter().fold(0., |acc, (_, n)| acc + n);
    let population_f = population as f64;
    species_fitted
        .into_iter()
        .map(|(specie_repr, fit_adjusted)| {
            (
                specie_repr,
                f64::round(population_f * fit_adjusted / fit_total) as usize,
            )
        })
        .collect()
}

/// initial population of a single specie consisting of single connection genomes
/// while it's not necessarily recommended to do an initual mutation, it allows us to mutate a
/// bisection on any genome without the need to check for existing connections beforehand
pub fn population_init(sensory: usize, action: usize, population: usize) -> (Vec<Specie>, usize) {
    let (genome, inno_head) = Genome::new(sensory, action);
    (
        vec![Specie {
            repr: SpecieRepr(genome.connections.clone()),
            members: vec![(genome, f64::MIN); population],
        }],
        inno_head,
    )
}

fn population_allocated<'a, T: Iterator<Item = &'a (Specie, f64)>>(
    species: T,
    population: usize,
) -> impl Iterator<Item = (Vec<(Genome, f64)>, usize)> + use<'a, T> {
    let viable = species
        .filter_map(|(specie, min_fitness)| {
            let viable = specie
                .members
                .iter()
                .filter(|&pair| (&pair.1 >= min_fitness))
                .cloned()
                .collect::<Vec<_>>();

            // (!viable.is_empty()).then_some((&specie.repr, viable));
            (!viable.is_empty()).then(|| Specie {
                repr: specie.repr.clone(),
                members: viable,
            })
        })
        .collect::<Vec<_>>();

    let alloc = population_alloc(viable.iter(), population);

    viable
        .into_iter()
        .filter_map(move |specie| alloc.get(&specie.repr).map(|pop| (specie.members, *pop)))
}

// reproduce a whole speciated population into a non-speciated population
pub fn population_reproduce(
    species: &[(Specie, f64)],
    population: usize,
    inno_head: usize,
    rng: &mut ThreadRng,
) -> (Vec<Genome>, usize) {
    // let species = population_viable(species.into_iter());
    // let species_pop = population_alloc(species, population);
    let mut innogen = InnoGen::new(inno_head);
    (
        population_allocated(species.iter(), population)
            .flat_map(|(members, pop)| reproduce(members, pop, &mut innogen, rng).unwrap())
            .collect::<Vec<_>>(),
        innogen.head,
    )
}

const SPECIE_THRESHOLD: f64 = 4.;

pub fn speciate(
    genomes: impl Iterator<Item = (Genome, f64)>,
    reprs: impl Iterator<Item = SpecieRepr>,
) -> Vec<Specie> {
    let mut sp = Vec::from_iter(reprs.map(|repr| Specie {
        repr,
        members: Vec::new(),
    }));

    for (genome, fitness) in genomes {
        match sp
            .iter_mut()
            .find(|Specie { repr, .. }| repr.delta(&genome.connections) < SPECIE_THRESHOLD)
        {
            Some(Specie { members, .. }) => members.push((genome, fitness)),
            None => {
                sp.push(Specie {
                    repr: SpecieRepr(genome.connections.clone()),
                    members: vec![(genome, fitness)],
                });
            }
        }
    }

    sp
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rng;

    #[test]
    fn test_inno_gen() {
        let mut inno = InnoGen::new(0);
        assert_eq!(inno.head, 0);
        assert_eq!(inno.path((0, 1)), 0);
        assert_eq!(inno.path((1, 2)), 1);
        assert_eq!(inno.path((0, 1)), 0);
        assert_eq!(inno.head, 2);

        let mut inno2 = InnoGen::new(inno.head);
        assert_eq!(inno2.path((1, 0)), 2);
        assert_eq!(inno2.path((0, 1)), 3);
    }

    #[test]
    fn test_population_init() {
        let count = 40;
        let (species, inno_head) = population_init(2, 2, count);
        assert_eq!(
            count,
            species
                .iter()
                .fold(0, |acc, Specie { members, .. }| acc + members.len())
        );
        assert!(inno_head != 0);
        for specie in species.iter() {
            assert_ne!(0, specie.len());
        }
        for (Genome { connections, .. }, fit) in
            species.iter().flat_map(|Specie { members, .. }| members)
        {
            assert_eq!(0, connections.len());
            assert_eq!(f64::MIN, *fit);
        }
    }

    #[test]
    fn test_specie_reproduce() {
        let mut rng = rng();
        let count = 40;
        let (species, inno_head) = population_init(2, 2, count);

        for specie in species {
            for i in [0, 1, count, count * 10] {
                assert_eq!(
                    i,
                    reproduce(
                        specie.members.clone(),
                        i,
                        &mut InnoGen::new(inno_head),
                        &mut rng
                    )
                    .unwrap()
                    .len()
                );
            }
        }
    }
}
