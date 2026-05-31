//! Stochastic Diffusion Search — hypothesis-based search with diffusion.

use rand::Rng;
use serde::{Deserialize, Serialize};

/// Configuration for SDS.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdsConfig {
    /// Number of agents.
    pub num_agents: usize,
    /// Number of iterations.
    pub max_iterations: usize,
    /// Threshold for partial evaluation.
    pub threshold: f64,
    /// Search space bounds.
    pub bounds: Vec<(f64, f64)>,
}

impl Default for SdsConfig {
    fn default() -> Self {
        Self {
            num_agents: 50,
            max_iterations: 100,
            threshold: 0.5,
            bounds: vec![(-5.0, 5.0)],
        }
    }
}

/// An SDS agent with a hypothesis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdsAgent {
    pub hypothesis: Vec<f64>,
    pub active: bool,
    pub fitness: f64,
}

/// Result of SDS.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdsResult {
    pub best_position: Vec<f64>,
    pub best_fitness: f64,
    pub history: Vec<f64>,
    pub cluster_counts: Vec<usize>,
}

/// Stochastic Diffusion Search.
pub struct StochasticDiffusionSearch {
    config: SdsConfig,
    dims: usize,
}

impl StochasticDiffusionSearch {
    pub fn new(config: SdsConfig) -> Self {
        let dims = config.bounds.len();
        Self { config, dims }
    }

    /// Run SDS minimizing the objective.
    pub fn run<F>(&self, objective: F) -> SdsResult
    where
        F: Fn(&[f64]) -> f64,
    {
        let mut rng = rand::thread_rng();

        // Initialize agents with random hypotheses
        let mut agents: Vec<SdsAgent> = (0..self.config.num_agents)
            .map(|_| {
                let pos = self.random_position(&mut rng);
                let fit = objective(&pos);
                SdsAgent {
                    hypothesis: pos,
                    active: false,
                    fitness: fit,
                }
            })
            .collect();

        let mut best_pos = agents[0].hypothesis.clone();
        let mut best_fit = agents[0].fitness;
        for a in &agents {
            if a.fitness < best_fit {
                best_fit = a.fitness;
                best_pos = a.hypothesis.clone();
            }
        }

        let mut history = Vec::new();
        let mut cluster_counts = Vec::new();

        for _ in 0..self.config.max_iterations {
            // Test phase: evaluate each agent's hypothesis
            for agent in &mut agents {
                // Score: how close to the threshold (lower fitness = better)
                agent.active = agent.fitness < self.config.threshold
                    || rng.gen::<f64>() > 0.5;
            }

            // Diffusion phase
            for i in 0..agents.len() {
                if !agents[i].active {
                    // Inactive: pick a random agent and adopt their hypothesis if active
                    let j = rng.gen_range(0..agents.len());
                    if agents[j].active {
                        agents[i].hypothesis = agents[j].hypothesis.clone();
                        agents[i].fitness = agents[j].fitness;
                    } else {
                        // Random new hypothesis
                        agents[i].hypothesis = self.random_position(&mut rng);
                        agents[i].fitness = objective(&agents[i].hypothesis);
                    }
                }
                // Active agents keep their hypothesis
            }

            // Context-sensitive diffusion: perturb hypotheses slightly
            for agent in &mut agents {
                if agent.active {
                    let mut new_hyp = agent.hypothesis.clone();
                    for d in 0..self.dims {
                        let range = self.config.bounds[d].1 - self.config.bounds[d].0;
                        let delta: f64 = rng.gen_range(-0.05 * range..0.05 * range);
                        new_hyp[d] += delta;
                        new_hyp[d] = new_hyp[d]
                            .clamp(self.config.bounds[d].0, self.config.bounds[d].1);
                    }
                    let new_fit = objective(&new_hyp);
                    if new_fit < agent.fitness {
                        agent.hypothesis = new_hyp;
                        agent.fitness = new_fit;
                    }
                }
            }

            // Track best and cluster size
            let mut active_count = 0;
            for a in &agents {
                if a.fitness < best_fit {
                    best_fit = a.fitness;
                    best_pos = a.hypothesis.clone();
                }
                if a.active {
                    active_count += 1;
                }
            }
            history.push(best_fit);
            cluster_counts.push(active_count);
        }

        SdsResult {
            best_position: best_pos,
            best_fitness: best_fit,
            history,
            cluster_counts,
        }
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
