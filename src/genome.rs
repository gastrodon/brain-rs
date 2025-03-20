use crate::{
    crossover::crossover,
    random::{CHANCE_MUTATE_BISECTION, CHANCE_MUTATE_CONNECTION, CHANCE_MUTATE_WEIGHT},
    specie::InnoGen,
};
use core::{
    cmp::{max, Ordering},
    error::Error,
    hash::Hash,
    iter::once,
};
use rand::{seq::IteratorRandom, Rng, RngCore};
use rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, fs, path::Path};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Node {
    Sensory,
    Action,
    Bias(f64),
    Internal,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Connection {
    pub inno: usize,
    pub from: usize,
    pub to: usize,
    pub weight: f64,
    pub enabled: bool,
}

impl Hash for Connection {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.inno.hash(state);
        self.from.hash(state);
        self.to.hash(state);
        ((1000. * self.weight) as usize).hash(state);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Genome {
    pub sensory: usize,
    pub action: usize,
    pub nodes: Vec<Node>,
    pub connections: Vec<Connection>,
}

impl Genome {
    pub fn to_string(&self) -> Result<String, Box<dyn Error>> {
        Ok(serde_json::to_string(self)?)
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self, Box<dyn Error>> {
        serde_json::from_str(s).map_err(|op| op.into())
    }

    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn Error>> {
        fs::write(path, self.to_string()?)?;
        Ok(())
    }

    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn Error>> {
        Self::from_str(&fs::read_to_string(path)?)
    }
}

impl Genome {
    const MUTATE_WEIGHT_FAC: f64 = 0.05;

    pub fn new(sensory: usize, action: usize) -> (Self, usize) {
        let mut nodes = Vec::with_capacity(sensory + action + 1);
        for _ in 0..sensory {
            nodes.push(Node::Sensory);
        }
        for _ in sensory..sensory + action {
            nodes.push(Node::Action);
        }
        nodes.push(Node::Bias(1.));

        (
            Self {
                sensory,
                action,
                nodes,
                connections: vec![],
            },
            (sensory + 1) * action,
        )
    }

    pub fn mutate_weights(&mut self, rng: &mut impl RngCore) {
        for conn in self.connections.iter_mut() {
            if rng.random_ratio(1, 10) {
                conn.weight = rng.sample(StandardNormal);
            } else {
                conn.weight += Self::MUTATE_WEIGHT_FAC * rng.sample::<f64, _>(StandardNormal)
            }
        }
    }

    // picks an unconnected pair, generates a connection between them, and applies it
    // fails if no pair can be picked
    pub fn mutate_connection(
        &mut self,
        rng: &mut impl RngCore,
        inext: &mut InnoGen,
    ) -> Result<(), Box<dyn Error>> {
        if let Some((from, to)) = gen_connection(self, rng) {
            self.connections.push(Connection {
                inno: inext.path((from, to)),
                from,
                to,
                weight: 1.,
                enabled: true,
            });
            Ok(())
        } else {
            Err("connections on genome are fully saturated".into())
        }
    }

    // Picks a source connection, bisects it, and applies it
    // picked source connection is marked as disabled
    pub fn mutate_bisection(
        &mut self,
        rng: &mut impl RngCore,
        inext: &mut InnoGen,
    ) -> Result<(), Box<dyn Error>> {
        if self.connections.is_empty() {
            return Err("no connections available to bisect".into());
        }

        let pick_idx = rng.random_range(0..self.connections.len());
        let new_node_idx = self.nodes.len();
        let (lower, upper) = {
            // possibly: would it make sense for a bisection to require a new inno?
            let pick = self.connections.get_mut(pick_idx).unwrap();
            pick.enabled = false;
            (
                // from -{1.}> bisect-node
                Connection {
                    inno: inext.path((pick.from, new_node_idx)),
                    from: pick.from,
                    to: new_node_idx,
                    weight: 1.,
                    enabled: true,
                },
                // bisect-node -{w}> to
                Connection {
                    inno: inext.path((new_node_idx, pick.to)),
                    from: new_node_idx,
                    to: pick.to,
                    weight: pick.weight,
                    enabled: true,
                },
            )
        };

        self.nodes.push(Node::Internal);
        self.connections.push(lower);
        self.connections.push(upper);
        Ok(())
    }

    pub fn maybe_mutate(
        &mut self,
        rng: &mut impl RngCore,
        innogen: &mut InnoGen,
    ) -> Result<(), Box<dyn Error>> {
        if rng.random_bool(CHANCE_MUTATE_WEIGHT) {
            self.mutate_weights(rng);
        }
        if rng.random_bool(CHANCE_MUTATE_CONNECTION) {
            self.mutate_connection(rng, innogen)?;
        }
        if rng.random_bool(CHANCE_MUTATE_BISECTION) && !self.connections.is_empty() {
            self.mutate_bisection(rng, innogen)?;
        }

        Ok(())
    }

    pub fn reproduce_with(
        &self,
        other: &Genome,
        self_fit: Ordering,
        rng: &mut impl RngCore,
    ) -> Self {
        let connections = crossover(&self.connections, &other.connections, self_fit, rng);
        let nodes_size = connections
            .iter()
            .fold(0, |prev, Connection { from, to, .. }| {
                max(prev, max(*from, *to))
            });

        let mut nodes = Vec::with_capacity(self.sensory + self.action + 1);
        for _ in 0..self.sensory {
            nodes.push(Node::Sensory);
        }
        for _ in self.sensory..self.sensory + self.action {
            nodes.push(Node::Action);
        }
        nodes.push(Node::Bias(1.));
        for _ in self.sensory + self.action..nodes_size {
            nodes.push(Node::Internal);
        }

        debug_assert!(
            connections
                .iter()
                .fold(0, |acc, c| max(acc, max(c.from, c.to)))
                < nodes.len()
        );

        Self {
            sensory: self.sensory,
            action: self.action,
            nodes,
            connections,
        }
    }
}

/// Given a genome with 0 or more nodes, try to generate a connection between nodes
/// a connection should have a unique (from, to) from any other connection on genome,
/// and the connection should not describe a node that points to itself
fn gen_connection(genome: &Genome, rng: &mut impl RngCore) -> Option<(usize, usize)> {
    let mut saturated = HashSet::new();
    loop {
        let from = (0..genome.nodes.len())
            .filter(|from| !saturated.contains(from))
            .choose(rng)?;

        let exclude = genome
            .connections
            .iter()
            .filter_map(|c| (c.from == from).then_some(c.to))
            .chain(once(from))
            .collect::<HashSet<_>>();

        if let Some(to) = (0..genome.nodes.len())
            .filter(|to| !exclude.contains(to))
            .choose(rng)
        {
            break Some((from, to));
        }

        saturated.insert(from);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{random::default_rng, specie::InnoGen};
    use std::vec;

    #[test]
    fn test_genome_creation() {
        let (genome, inno_head) = Genome::new(3, 2);
        assert_eq!(inno_head, 8);
        assert_eq!(genome.sensory, 3);
        assert_eq!(genome.action, 2);
        assert_eq!(genome.nodes.len(), 6);
        assert!(matches!(genome.nodes[0], Node::Sensory));
        assert!(matches!(genome.nodes[3], Node::Action));
        assert!(matches!(genome.nodes[5], Node::Bias(_)));

        let (genome_empty, inno_head) = Genome::new(0, 0);
        assert_eq!(inno_head, 0);
        assert_eq!(genome_empty.sensory, 0);
        assert_eq!(genome_empty.action, 0);
        assert_eq!(genome_empty.nodes.len(), 1);
        assert!(matches!(genome_empty.nodes[0], Node::Bias(_)));

        let (genome_only_sensory, inno_head) = Genome::new(3, 0);
        assert_eq!(inno_head, 0);
        assert_eq!(genome_only_sensory.sensory, 3);
        assert_eq!(genome_only_sensory.action, 0);
        assert_eq!(genome_only_sensory.nodes.len(), 4);
        assert!(matches!(genome_only_sensory.nodes[0], Node::Sensory));
        assert!(matches!(genome_only_sensory.nodes[2], Node::Sensory));
        assert!(matches!(genome_only_sensory.nodes[3], Node::Bias(_)));

        let (genome_only_action, inno_head) = Genome::new(0, 3);
        assert_eq!(inno_head, 3);
        assert_eq!(genome_only_action.sensory, 0);
        assert_eq!(genome_only_action.action, 3);
        assert_eq!(genome_only_action.nodes.len(), 4);
        assert!(matches!(genome_only_action.nodes[0], Node::Action));
        assert!(matches!(genome_only_action.nodes[2], Node::Action));
        assert!(matches!(genome_only_action.nodes[3], Node::Bias(_)));
    }

    #[test]
    fn test_gen_connection() {
        let genome = Genome {
            sensory: 1,
            action: 1,
            nodes: vec![Node::Sensory, Node::Action],
            connections: vec![],
        };
        for _ in 0..100 {
            match gen_connection(&genome, &mut default_rng()) {
                Some((0, o)) | Some((o, 0)) => assert_eq!(o, 1),
                Some(p) => unreachable!("invalid pair {p:?} gen'd"),
                None => unreachable!("no path gen'd"),
            }
        }
    }

    #[test]
    fn test_gen_connection_no_dupe() {
        let genome = Genome {
            sensory: 1,
            action: 1,
            nodes: vec![Node::Sensory, Node::Action],
            connections: vec![Connection {
                inno: 0,
                from: 0,
                to: 1,
                weight: 1.,
                enabled: true,
            }],
        };
        for _ in 0..100 {
            assert_eq!(gen_connection(&genome, &mut default_rng()), Some((1, 0)));
        }
    }

    #[test]
    fn test_gen_connection_none_possble() {
        assert_eq!(
            gen_connection(
                &Genome {
                    sensory: 0,
                    action: 0,
                    nodes: vec![],
                    connections: vec![Connection {
                        inno: 0,
                        from: 0,
                        to: 1,
                        weight: 1.,
                        enabled: true,
                    }],
                },
                &mut default_rng()
            ),
            None
        );
    }

    #[test]
    fn test_gen_connection_saturated() {
        assert_eq!(
            gen_connection(
                &Genome {
                    sensory: 2,
                    action: 2,
                    nodes: vec![
                        Node::Action,
                        Node::Action,
                        Node::Sensory,
                        Node::Sensory,
                        Node::Bias(1.),
                    ],
                    connections: (0..5)
                        .flat_map(|from| {
                            (0..5).filter_map(move |to| {
                                (from != to).then_some(Connection {
                                    inno: 0,
                                    from,
                                    to,
                                    weight: 1.,
                                    enabled: true,
                                })
                            })
                        })
                        .collect(),
                },
                &mut default_rng()
            ),
            None
        )
    }

    #[test]
    fn test_mutate_connection() {
        let (mut genome, _) = Genome::new(4, 4);
        let mut inext = InnoGen::new(0);
        genome.connections = vec![
            Connection {
                inno: inext.path((0, 1)),
                from: 0,
                to: 1,
                weight: 0.5,
                enabled: true,
            },
            Connection {
                inno: inext.path((1, 2)),
                from: 1,
                to: 2,
                weight: 0.5,
                enabled: true,
            },
        ];

        let before = genome.clone();
        genome
            .mutate_connection(&mut default_rng(), &mut inext)
            .unwrap();

        assert_eq!(genome.connections.len(), before.connections.len() + 1);

        let tail = genome.connections.last().unwrap();
        assert!(!before.connections.iter().any(|c| c.inno == tail.inno));
        assert!(!before
            .connections
            .iter()
            .any(|c| (c.from, c.to) == (tail.from, tail.to)));
        assert_eq!(tail.weight, 1.);
    }

    #[test]
    fn test_mutate_bisection() {
        let (mut genome, _) = Genome::new(0, 1);
        genome.connections = vec![Connection {
            inno: 0,
            from: 0,
            to: 1,
            weight: 0.5,
            enabled: true,
        }];
        let innogen = &mut InnoGen::new(1);
        genome
            .mutate_bisection(&mut default_rng(), innogen)
            .unwrap();

        assert!(!genome.connections[0].enabled);

        assert_eq!(genome.connections[1].from, 0);
        assert_eq!(genome.connections[1].to, 2);
        assert_eq!(genome.connections[1].weight, 1.0);
        assert!(genome.connections[1].enabled);
        assert_eq!(
            genome.connections[1].inno,
            innogen.path((genome.connections[1].from, genome.connections[1].to))
        );

        assert_eq!(genome.connections[2].from, 2);
        assert_eq!(genome.connections[2].to, 1);
        assert_eq!(genome.connections[2].weight, 0.5);
        assert!(genome.connections[2].enabled);
        assert_eq!(
            genome.connections[2].inno,
            innogen.path((genome.connections[2].from, genome.connections[2].to))
        );

        assert_ne!(genome.connections[0].inno, genome.connections[1].inno);
        assert_ne!(genome.connections[1].inno, genome.connections[2].inno);
        assert_ne!(genome.connections[0].inno, genome.connections[2].inno);
    }

    #[test]
    fn test_mutate_bisection_empty_genome() {
        let (mut genome, _) = Genome::new(0, 0);
        let result = genome.mutate_bisection(&mut default_rng(), &mut InnoGen::new(0));
        assert!(result.is_err());
    }

    #[test]
    fn test_mutate_bisection_no_connections() {
        let (mut genome, _) = Genome::new(2, 2);
        genome.connections = vec![];
        let result = genome.mutate_bisection(&mut default_rng(), &mut InnoGen::new(0));
        assert!(result.is_err());
    }

    #[test]
    fn test_state_head() {
        let mut state = vec![0.; 5];
        {
            let size = 3;
            let state: &mut [f64] = &mut state;
            assert!(state.len() >= size);
            &mut state[0..size]
        }
        .clone_from_slice(&[1., 2., 3.]);
        assert_eq!(state, vec![1., 2., 3., 0., 0.])
    }
}
