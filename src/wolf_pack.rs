//! Wolf Pack Algorithm — leaders, scouts, and followers.

use rand::Rng;
use serde::{Deserialize, Serialize};

/// Wolf role in the pack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WolfRole {
    Leader,
    Scout,
    Follower,
}

/// A single wolf.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Wolf {
    pub position: Vec<f64>,
    pub fitness: f64,
    pub role: WolfRole,
}

/// Configuration for the Wolf Pack Algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WolfPackConfig {
    /// Total number of wolves.
    pub num_wolves: usize,
    /// Number of leaders (alpha wolves).
    pub num_leaders: usize,
    /// Number of scouts.
    pub num_scouts: usize,
    /// Step size for movement.
    pub step_size: f64,
    /// Scout range.
    pub scout_range: f64,
    /// Max iterations.
    pub max_iterations: usize,
    /// Search space bounds.
    pub bounds: Vec<(f64, f64)>,
}

impl Default for WolfPackConfig {
    fn default() -> Self {
        Self {
            num_wolves: 30,
            num_leaders: 3,
            num_scouts: 10,
            step_size: 0.5,
            scout_range: 2.0,
            max_iterations: 100,
            bounds: vec![(-5.0, 5.0)],
        }
    }
}

/// Result of the Wolf Pack Algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WolfPackResult {
    pub best_position: Vec<f64>,
    pub best_fitness: f64,
    pub history: Vec<f64>,
}

/// Wolf Pack Algorithm.
pub struct WolfPackAlgorithm {
    config: WolfPackConfig,
    dims: usize,
}

impl WolfPackAlgorithm {
    pub fn new(config: WolfPackConfig) -> Self {
        let dims = config.bounds.len();
        Self { config, dims }
    }

    /// Run the wolf pack algorithm minimizing the objective.
    pub fn run<F>(&self, objective: F) -> WolfPackResult
    where
        F: Fn(&[f64]) -> f64,
    {
        let mut rng = rand::thread_rng();
        let mut wolves: Vec<Wolf> = (0..self.config.num_wolves)
            .map(|_| {
                let pos = self.random_position(&mut rng);
                Wolf {
                    position: pos,
                    fitness: f64::MAX,
                    role: WolfRole::Follower,
                }
            })
            .collect();

        // Evaluate initial fitness
        for wolf in &mut wolves {
            wolf.fitness = objective(&wolf.position);
        }

        let mut history = Vec::new();

        for _ in 0..self.config.max_iterations {
            // Assign roles based on fitness ranking
            wolves.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
            for (i, wolf) in wolves.iter_mut().enumerate() {
                if i < self.config.num_leaders {
                    wolf.role = WolfRole::Leader;
                } else if i < self.config.num_leaders + self.config.num_scouts {
                    wolf.role = WolfRole::Scout;
                } else {
                    wolf.role = WolfRole::Follower;
                }
            }

            // Leader phase: small local search around best positions
            for wolf in wolves.iter_mut() {
                if wolf.role == WolfRole::Leader {
                    let mut new_pos = wolf.position.clone();
                    for d in 0..self.dims {
                        let delta: f64 = rng.gen_range(-self.config.step_size..self.config.step_size);
                        new_pos[d] += delta;
                        new_pos[d] = new_pos[d].clamp(self.config.bounds[d].0, self.config.bounds[d].1);
                    }
                    let new_fit = objective(&new_pos);
                    if new_fit < wolf.fitness {
                        wolf.position = new_pos;
                        wolf.fitness = new_fit;
                    }
                }
            }

            // Scout phase: wider exploration
            for wolf in wolves.iter_mut() {
                if wolf.role == WolfRole::Scout {
                    let mut new_pos = wolf.position.clone();
                    for d in 0..self.dims {
                        let delta: f64 = rng.gen_range(-self.config.scout_range..self.config.scout_range);
                        new_pos[d] += delta;
                        new_pos[d] = new_pos[d].clamp(self.config.bounds[d].0, self.config.bounds[d].1);
                    }
                    let new_fit = objective(&new_pos);
                    if new_fit < wolf.fitness {
                        wolf.position = new_pos;
                        wolf.fitness = new_fit;
                    }
                }
            }

            // Follower phase: move toward nearest leader
            // Collect leader positions first to avoid borrow conflict
            let leader_positions: Vec<Vec<f64>> = wolves.iter()
                .filter(|w| w.role == WolfRole::Leader)
                .map(|w| w.position.clone())
                .collect();
            for wolf in wolves.iter_mut() {
                if wolf.role == WolfRole::Follower && !leader_positions.is_empty() {
                    // Find nearest leader
                    let nearest = leader_positions
                        .iter()
                        .min_by(|a, b| {
                            let da = self.distance(&wolf.position, a);
                            let db = self.distance(&wolf.position, b);
                            da.partial_cmp(&db).unwrap()
                        })
                        .unwrap();

                    // Move toward leader
                    for d in 0..self.dims {
                        let direction = nearest[d] - wolf.position[d];
                        let dist = direction.abs().max(1e-10);
                        let step = self.config.step_size * direction / dist;
                        wolf.position[d] += step;
                        wolf.position[d] = wolf.position[d]
                            .clamp(self.config.bounds[d].0, self.config.bounds[d].1);
                    }
                    wolf.fitness = objective(&wolf.position);
                }
            }

            // Track best
            let best = wolves
                .iter()
                .min_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
                .unwrap();
            history.push(best.fitness);
        }

        let best = wolves
            .into_iter()
            .min_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
            .unwrap();

        WolfPackResult {
            best_position: best.position,
            best_fitness: best.fitness,
            history,
        }
    }

    fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
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
