#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]

pub mod crossover;
pub mod genome;
pub mod macros;
pub mod network;
pub mod random;
pub mod scenario;
pub mod specie;

pub use genome::{Connection, Genome};
pub use network::{activate, Ctrnn, Network};
pub use random::{Happens, Probabilities};
pub use scenario::{Hook, Scenario, Stats};
pub use specie::Specie;
