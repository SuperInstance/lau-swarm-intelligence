//! Bacterial Foraging Optimization — chemotaxis, swarming, reproduction, elimination-dispersal.

use rand::Rng;
use serde::{Deserialize, Serialize};

/// Configuration for Bacterial Foraging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacterialConfig {
    /// Number of bacteria.
    pub num_bacteria: usize,
    /// Number of chemotactic steps.
    pub num_chemotactic: usize,
    /// Number of swim steps per tumble.
    pub num_swim: usize,
    /// Number of reproduction steps.
    pub num_reproduction: usize,
    /// Number of elimination-dispersal events.
    pub num_elimination: usize,
    /// Step size for chemotaxis (C_i).
    pub step_size: f64,
    /// Elimination probability.
    pub elim_probability: f64,
    /// Attractant depth (d_attract).
    pub d_attract: f64,
    /// Attractant width (w_attract).
    pub w_attract: f64,
    /// Repellent height (h_repel).
    pub h_repel: f64,
    /// Repellent width (w_repel).
    pub w_repel: f64,
    /// Search space bounds.
    pub bounds: Vec<(f64, f64)>,
}

impl Default for BacterialConfig {
    fn default() -> Self {
        Self {
            num_bacteria: 50,
            num_chemotactic: 100,
            num_swim: 4,
            num_reproduction: 4,
            num_elimination: 2,
            step_size: 0.1,
            elim_probability: 0.25,
            d_attract: 0.1,
            w_attract: 0.2,
            h_repel: 0.1,
            w_repel: 10.0,
            bounds: vec![(-5.0, 5.0)],
        }
    }
}

/// A single bacterium.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bacterium {
    pub position: Vec<f64>,
    pub fitness: f64,
    pub health: f64,
}

/// Result of bacterial foraging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacterialResult {
    pub best_position: Vec<f64>,
    pub best_fitness: f64,
    pub history: Vec<f64>,
}

/// Bacterial Foraging Optimization.
pub struct BacterialForaging {
    config: BacterialConfig,
    dims: usize,
}

impl BacterialForaging {
    pub fn new(config: BacterialConfig) -> Self {
        let dims = config.bounds.len();
        Self { config, dims }
    }

    /// Run the bacterial foraging algorithm minimizing the objective.
    pub fn run<F>(&self, objective: F) -> BacterialResult
    where
        F: Fn(&[f64]) -> f64,
    {
        let mut rng = rand::thread_rng();
        let mut bacteria: Vec<Bacterium> = (0..self.config.num_bacteria)
            .map(|_| {
                let pos = self.random_position(&mut rng);
                Bacterium {
                    fitness: f64::MAX,
                    health: 0.0,
                    position: pos,
                }
            })
            .collect();

        let mut best_pos = bacteria[0].position.clone();
        let mut best_fit = f64::MAX;
        let mut history = Vec::new();

        // Elimination-dispersal loop
        for _ in 0..self.config.num_elimination {
            // Reproduction loop
            for _ in 0..self.config.num_reproduction {
                // Chemotaxis loop
                for _ in 0..self.config.num_chemotactic {
                    for i in 0..bacteria.len() {
                        // Tumble: random direction
                        let mut direction = vec![0.0; self.dims];
                        let mut norm: f64 = 0.0;
                        for d in 0..self.dims {
                            direction[d] = rng.gen_range(-1.0..1.0);
                            norm += direction[d] * direction[d];
                        }
                        norm = norm.sqrt().max(1e-10);
                        for d in 0..self.dims {
                            direction[d] /= norm;
                        }

                        // Swim
                        let mut j_fitness = self.eval_with_swarm(&bacteria, i, &objective);
                        bacteria[i].health = j_fitness;

                        for _ in 0..self.config.num_swim {
                            let mut new_pos = bacteria[i].position.clone();
                            for d in 0..self.dims {
                                new_pos[d] += self.config.step_size * direction[d];
                                new_pos[d] = new_pos[d]
                                    .clamp(self.config.bounds[d].0, self.config.bounds[d].1);
                            }
                            let new_j = {
                                let old = bacteria[i].position.clone();
                                bacteria[i].position = new_pos.clone();
                                let j = self.eval_with_swarm(&bacteria, i, &objective);
                                bacteria[i].position = old;
                                j
                            };

                            if new_j < j_fitness {
                                bacteria[i].position = new_pos;
                                bacteria[i].health += new_j;
                                j_fitness = new_j;
                            } else {
                                break;
                            }
                        }

                        bacteria[i].fitness = objective(&bacteria[i].position);
                    }
                }

                // Reproduction: sort by health (lower = better), replace worst half
                bacteria.sort_by(|a, b| a.health.partial_cmp(&b.health).unwrap());
                let half = bacteria.len() / 2;
                let n = bacteria.len();
                let clones: Vec<Bacterium> = (0..half).map(|i| bacteria[i].clone()).collect();
                for (i, clone) in clones.into_iter().enumerate() {
                    bacteria[n - 1 - i] = clone;
                }
            }

            // Elimination-dispersal
            for bacterium in &mut bacteria {
                if rng.gen::<f64>() < self.config.elim_probability {
                    bacterium.position = self.random_position(&mut rng);
                    bacterium.fitness = objective(&bacterium.position);
                }
            }

            // Track best
            for b in &bacteria {
                if b.fitness < best_fit {
                    best_fit = b.fitness;
                    best_pos = b.position.clone();
                }
            }
            history.push(best_fit);
        }

        BacterialResult {
            best_position: best_pos,
            best_fitness: best_fit,
            history,
        }
    }

    /// Evaluate fitness with swarming effect (cell-to-cell signaling).
    fn eval_with_swarm<F>(&self, bacteria: &[Bacterium], idx: usize, objective: &F) -> f64
    where
        F: Fn(&[f64]) -> f64,
    {
        let base_fitness = objective(&bacteria[idx].position);

        // Compute swarming effect
        let mut swarm = 0.0;
        for (j, other) in bacteria.iter().enumerate() {
            if j != idx {
                let diff: f64 = bacteria[idx]
                    .position
                    .iter()
                    .zip(other.position.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                swarm += -self.config.d_attract
                    * (-self.config.w_attract * diff).exp()
                    + self.config.h_repel * (-self.config.w_repel * diff).exp();
            }
        }

        base_fitness + swarm
    }

    fn random_position(&self, rng: &mut impl Rng) -> Vec<f64> {
        (0..self.dims)
            .map(|d| {
                let (lo, hi) = self.config.bounds[d];
                rng.gen_range(lo..hi)
            })
            .collect()
    }
}
