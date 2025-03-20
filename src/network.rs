use crate::{genome::Node, Connection, Genome};
use core::error::Error;
use rulinalg::matrix::{BaseMatrix, BaseMatrixMut, Matrix};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

pub mod activate {
    use core::f64::consts::E;

    pub fn steep_sigmoid(x: f64) -> f64 {
        1. / (1. + E.powf(-4.9 * x))
    }

    pub fn relu(x: f64) -> f64 {
        if x < 0. {
            0.
        } else {
            x
        }
    }
}

pub mod loss {
    pub fn decay_quadratic(want: f64, x: f64) -> f64 {
        1. - (want - x).abs().powf(2.)
    }
}

pub trait Network: Serialize + for<'de> Deserialize<'de> {
    fn from_genome(genome: &Genome) -> Self;
    fn step<F: Fn(f64) -> f64>(&mut self, prec: usize, input: &[f64], σ: F);
    fn flush(&mut self);
    fn output(&self) -> &[f64];

    fn to_string(&self) -> Result<String, Box<dyn Error>> {
        Ok(serde_json::to_string(self)?)
    }

    fn from_str(s: &str) -> Result<Self, Box<dyn Error>>
    where
        Self: Sized,
    {
        serde_json::from_str(s).map_err(|op| op.into())
    }

    fn to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn Error>> {
        fs::write(path, self.to_string()?)?;
        Ok(())
    }

    fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn Error>>
    where
        Self: Sized,
    {
        Self::from_str(&fs::read_to_string(path)?)
    }
}

fn serialize<S>(matrix: &Matrix<f64>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    // Convert f64 values to u64 bits for precise serialization
    let bits: Vec<u64> = matrix.data().iter().map(|&f| f64::to_bits(f)).collect();

    bits.serialize(serializer)
}

fn deserialize_flat<'de, D>(deserializer: D) -> Result<Matrix<f64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Vec::<u64>::deserialize(deserializer).map(|v| {
        // Convert u64 bits back to f64 values
        let float_data: Vec<f64> = v.into_iter().map(f64::from_bits).collect();

        Matrix::new(1, float_data.len(), float_data)
    })
}

fn deserialize_square<'de, D>(deserializer: D) -> Result<Matrix<f64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Vec::<u64>::deserialize(deserializer).map(|v| {
        // Convert u64 bits back to f64 values
        let float_data: Vec<f64> = v.into_iter().map(f64::from_bits).collect();

        let n = (float_data.len() as f64).sqrt() as usize;
        debug_assert_eq!(n * n, float_data.len(), "non-square weight vec");
        Matrix::new(n, n, float_data)
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Ctrnn {
    #[serde(serialize_with = "serialize", deserialize_with = "deserialize_flat")]
    pub y: Matrix<f64>, // 1d state of neurons 0-N
    #[serde(serialize_with = "serialize", deserialize_with = "deserialize_flat")]
    pub θ: Matrix<f64>, // 1d bias of neurons 0-N               (\u3b8)
    #[serde(serialize_with = "serialize", deserialize_with = "deserialize_flat")]
    pub τ: Matrix<f64>, // 1d membrane resistance time constant (\u3c4)
    #[serde(serialize_with = "serialize", deserialize_with = "deserialize_square")]
    pub w: Matrix<f64>, // Nd weights between neurons, indexed as [from, to]
    pub sensory: (usize, usize),
    pub action: (usize, usize),
}

impl Network for Ctrnn {
    fn from_genome(genome: &Genome) -> Self {
        let cols = genome.nodes.len();
        Self {
            y: Matrix::zeros(1, cols),
            θ: Matrix::new(
                1,
                cols,
                genome
                    .nodes
                    .iter()
                    .map(|n| if let Node::Bias(b) = n { *b } else { 0. })
                    .collect::<Vec<_>>(),
            ),
            τ: Matrix::ones(1, cols),
            w: {
                let mut w = vec![0.; cols * cols];
                for Connection {
                    from, to, weight, ..
                } in genome.connections.iter().filter(|c| c.enabled)
                {
                    w[from * cols + to] = *weight;
                }
                Matrix::new(cols, cols, w)
            },
            sensory: (0, genome.sensory),
            action: (genome.sensory, genome.sensory + genome.action),
        }
    }

    fn step<F: Fn(f64) -> f64>(&mut self, prec: usize, input: &[f64], σ: F) {
        let mut m_input = Matrix::zeros(1, self.y.cols());
        m_input.mut_data()[self.sensory.0..self.sensory.1].copy_from_slice(input);

        let inv = 1. / (prec as f64);
        for _ in 0..prec {
            self.y += (((&self.y + &self.θ).apply(&σ) * &self.w) - &self.y + &m_input)
                .elediv(&self.τ)
                .apply(&|v| v * inv);
        }
    }

    fn flush(&mut self) {
        self.y = Matrix::zeros(1, self.y.cols());
    }

    fn output(&self) -> &[f64] {
        &self.y.data()[self.action.0..self.action.1]
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::random::default_rng;
    use rand_distr::{num_traits::Float, Distribution, Uniform};

    // Macro for comparing f64 arrays with epsilon tolerance
    macro_rules! assert_matrices_f64_eq {
        ($a:expr, $b:expr) => {
            assert_eq!($a.len(), $b.len(), "Matrices have different lengths");

            for (i, (l, r)) in $a.iter().zip($b.iter()).enumerate() {
                let diff = (l - r).abs();
                assert!(
                    diff < f64::EPSILON,
                    "[{}]: {} != {} (diff: {})",
                    i,
                    l,
                    r,
                    diff
                );
            }
        };
    }

    #[test]
    fn test_ctrnn_serialization_deserialization() {
        let n_neurons = 10;
        let mut rng = default_rng();
        let dist = Uniform::new(-10., 10.).unwrap();

        let mut y_data = vec![0.0; n_neurons];
        let mut theta_data = vec![0.0; n_neurons];
        let mut tau_data = vec![0.0; n_neurons];
        let mut w_data = vec![0.0; n_neurons * n_neurons];

        for i in 0..n_neurons {
            y_data[i] = dist.sample(&mut rng);
            theta_data[i] = dist.sample(&mut rng);
            tau_data[i] = dist.sample(&mut rng).abs() + 0.1;

            for j in 0..n_neurons {
                w_data[i * n_neurons + j] = dist.sample(&mut rng);
            }
        }

        let original = Ctrnn {
            y: Matrix::new(1, n_neurons, y_data),
            θ: Matrix::new(1, n_neurons, theta_data),
            τ: Matrix::new(1, n_neurons, tau_data),
            w: Matrix::new(n_neurons, n_neurons, w_data),
            sensory: (0, 2),
            action: (3, 5),
        };

        let serialized = original.to_string().expect("Failed to serialize");

        let deserialized = Ctrnn::from_str(&serialized).expect("Failed to deserialize");

        assert_matrices_f64_eq!(original.y.data(), deserialized.y.data());
        assert_matrices_f64_eq!(original.θ.data(), deserialized.θ.data());
        assert_matrices_f64_eq!(original.τ.data(), deserialized.τ.data());
        assert_matrices_f64_eq!(original.w.data(), deserialized.w.data());

        assert_eq!(original.sensory, deserialized.sensory);
        assert_eq!(original.action, deserialized.action);
    }

    #[test]
    fn test_ctrnn_behavioral_equivalence() {
        let n_neurons = 10;
        let mut rng = default_rng();
        let dist = Uniform::new(-10., 10.).unwrap();

        let mut y_data = vec![0.0; n_neurons];
        let mut θ_data = vec![0.0; n_neurons];
        let mut τ_data = vec![0.0; n_neurons];
        let mut w_data = vec![0.0; n_neurons * n_neurons];

        for i in 0..n_neurons {
            y_data[i] = dist.sample(&mut rng);
            θ_data[i] = dist.sample(&mut rng);
            τ_data[i] = dist.sample(&mut rng).abs() + 0.1;

            for j in 0..n_neurons {
                w_data[i * n_neurons + j] = dist.sample(&mut rng);
            }
        }

        let mut original = Ctrnn {
            y: Matrix::new(1, n_neurons, y_data),
            θ: Matrix::new(1, n_neurons, θ_data),
            τ: Matrix::new(1, n_neurons, τ_data),
            w: Matrix::new(n_neurons, n_neurons, w_data),
            sensory: (0, 2),
            action: (3, 5),
        };

        let mut deserialized = Ctrnn::from_str(&original.to_string().expect("Failed to serialize"))
            .expect("Failed to deserialize");

        let precision = 10;
        let n_steps = 500;

        for __ in 0..n_steps {
            let input: Vec<f64> = (0..2).map(|_| dist.sample(&mut rng)).collect();

            original.step(precision, &input, activate::steep_sigmoid);
            deserialized.step(precision, &input, activate::steep_sigmoid);

            let original_output = original.output();
            let deserialized_output = deserialized.output();

            assert_matrices_f64_eq!(original_output, deserialized_output);
        }
    }

    macro_rules! assert_f64_approx {
        ($l:expr, $r:expr) => {
            assert!(
                ($l - $r).abs() < f64::EPSILON,
                "assertion failed: {} !~ {}",
                $l,
                $r
            )
        };
        ($l:expr, $r:expr, $msg:expr) => {
            assert!(
                ($l - $r).abs() < f64::EPSILON,
                "assertion failed: {} !~ {}: {}",
                $l,
                $r,
                $msg
            )
        };
    }

    #[test]
    fn test_from_genome() {
        let (mut genome, _) = Genome::new(2, 2);
        genome.connections = vec![
            Connection {
                inno: 0,
                from: 0,
                to: 3,
                weight: 0.5,
                enabled: true,
            },
            Connection {
                inno: 1,
                from: 0,
                to: 1,
                weight: -1.,
                enabled: true,
            },
            Connection {
                inno: 2,
                from: 0,
                to: 1,
                weight: 1.2,
                enabled: false,
            },
        ];

        let nn = Ctrnn::from_genome(&genome);
        unsafe {
            for Connection {
                from, to, weight, ..
            } in genome.connections.iter().filter(|c| c.enabled)
            {
                assert_f64_approx!(nn.w.get_unchecked([*from, *to]), weight);
            }

            for (i, node) in genome.nodes.iter().enumerate() {
                assert_f64_approx!(
                    nn.θ.get_unchecked([0, i]),
                    if let Node::Bias(b) = node { b } else { &0. }
                )
            }
        }

        for i in nn.sensory.0..nn.sensory.1 {
            assert!(genome
                .nodes
                .get(i)
                .is_some_and(|n| matches!(n, Node::Sensory)))
        }
        for i in nn.action.0..nn.action.1 {
            assert!(genome
                .nodes
                .get(i)
                .is_some_and(|n| matches!(n, Node::Action)))
        }
    }
}
