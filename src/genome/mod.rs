pub mod connection;
pub mod node;
pub mod recurrent;

pub use connection::WConnection;
pub use recurrent::CTRGenome;

use crate::{
    random::{EvolutionEvent, Happens},
    specie::InnoGen,
};
use core::{cmp::Ordering, error::Error, fmt::Debug, hash::Hash};
use rand::{Rng, RngCore};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NodeKind {
    Sensory,
    Action,
    Internal,
    Static,
}

pub trait Node: Serialize + for<'de> Deserialize<'de> + Clone + Debug + PartialEq {
    /// A new node of some kind
    fn new(kind: NodeKind) -> Self;

    /// The cond of node this is, where
    /// - sensory reads input to a network
    /// - action describes output from a network
    /// - internal is a hidden node of a network
    /// - static is a hidden node, but unaffected by input, yielding a static value
    fn kind(&self) -> NodeKind;

    /// The bias of a node, returning 0. for nodes who can't have bias
    fn bias(&self) -> f64;
}

pub trait Connection<N: Node>:
    Serialize + for<'de> Deserialize<'de> + Clone + Hash + PartialEq + Default + Debug
{
    const EXCESS_COEFFICIENT: f64;
    const DISJOINT_COEFFICIENT: f64;
    const PARAM_COEFFICIENT: f64;

    fn new(from: usize, to: usize, inno: &mut InnoGen) -> Self;

    /// gene innovation id
    fn inno(&self) -> usize;

    /// whether or not this connection is active, and therefore affects its genomes behavior
    fn enabled(&self) -> bool;

    /// unconditionally enable this connection
    fn enable(&mut self);

    /// unconditionally disable this connection
    fn disable(&mut self);

    /// (from, to) path of this connection
    fn path(&self) -> (usize, usize);

    /// path source
    fn from(&self) -> usize {
        self.path().0
    }

    fn weight(&self) -> f64;

    /// path destination
    fn to(&self) -> usize {
        self.path().1
    }

    /// mutate connection parameters
    fn mutate_params(&mut self, rng: &mut (impl RngCore + Happens));

    /// bisect this connection; disabling it, and returning the (upper, lower) bisection pair
    fn bisect(&mut self, center: usize, inno: &mut InnoGen) -> (Self, Self);

    /// difference of connection parameters ( for example, weight )
    /// between this and another connection with the same innovation id
    fn param_diff(&self, other: &Self) -> f64;
}

pub trait Genome<N: Node, C: Connection<N>>: Serialize + for<'de> Deserialize<'de> + Clone {
    /// A new genome of this type, with a known input and output size
    fn new(sensory: usize, action: usize) -> (Self, usize);

    fn nodes(&self) -> &[N];

    /// Push a new node onto the genome
    fn push_node(&mut self, node: N);

    /// A collection to the connections comprising this genome
    fn connections(&self) -> &[C];

    /// Mutable reference to the connections comprising this genome
    fn connections_mut(&mut self) -> &mut [C];

    /// Push a connection onto the genome
    fn push_connection(&mut self, connection: C);

    /// Push 2 connections onto the genome, first then second.
    /// The idea with this is that we'll often do so as a result of bisection,
    /// so this gives us a chance to grow the connections just once if we want
    fn push_2_connections(&mut self, first: C, second: C) {
        self.push_connection(first);
        self.push_connection(second);
    }

    /// Perform a ( possible? TODO ) mutation across every weight
    fn mutate_params(&mut self, rng: &mut (impl RngCore + Happens));

    /// Find some open path ( that is, a path between nodes from -> to )
    /// that no connection is occupying if any exist
    fn open_path(&self, rng: &mut (impl RngCore + Happens)) -> Option<(usize, usize)>;

    /// Generate a new connection between unconnected nodes.
    /// Panics if all possible connections between nodes are saturated
    fn mutate_connection(&mut self, rng: &mut (impl RngCore + Happens), inno: &mut InnoGen) {
        if let Some((from, to)) = self.open_path(rng) {
            self.push_connection(C::new(from, to, inno));
        } else {
            panic!("connections on genome are fully saturated")
        }
    }

    /// Bisect an existing connection. Should panic if there are no connections to bisect
    fn mutate_bisection(&mut self, rng: &mut (impl RngCore + Happens), inno: &mut InnoGen) {
        if self.connections().is_empty() {
            panic!("no connections available to bisect");
        }

        let center = self.nodes().len();
        let source = rng.random_range(0..self.connections().len());
        let (lower, upper) = self
            .connections_mut()
            .get_mut(source)
            .unwrap()
            .bisect(center, inno);

        self.push_node(N::new(NodeKind::Internal));
        self.push_2_connections(lower, upper);
    }

    /// Perform 0 or more mutations on this genome ( should this be the only mutator exposed? TODO )
    fn mutate(&mut self, rng: &mut (impl RngCore + Happens), innogen: &mut InnoGen) {
        if rng.happens(EvolutionEvent::MutateWeight) {
            self.mutate_params(rng);
        }
        if rng.happens(EvolutionEvent::MutateConnection) {
            self.mutate_connection(rng, innogen);
        }
        if rng.happens(EvolutionEvent::MutateBisection) && !self.connections().is_empty() {
            self.mutate_bisection(rng, innogen);
        }
    }

    /// Perform crossover reproduction with other, where our fitness is `fitness_cmp` compared to other
    fn reproduce_with(
        &self,
        other: &Self,
        fitness_cmp: Ordering,
        rng: &mut (impl RngCore + Happens),
    ) -> Self;

    fn to_string(&self) -> Result<String, Box<dyn Error>> {
        Ok(serde_json::to_string(self)?)
    }

    #[allow(clippy::should_implement_trait)]
    fn from_str(s: &str) -> Result<Self, Box<dyn Error>> {
        serde_json::from_str(s).map_err(|op| op.into())
    }

    fn to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn Error>> {
        fs::write(path, self.to_string()?)?;
        Ok(())
    }

    fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn Error>> {
        Self::from_str(&fs::read_to_string(path)?)
    }
}
