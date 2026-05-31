//! Ant Colony Optimization — pheromone trails and path construction.

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::f64;

/// Configuration for an ACO solver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcoConfig {
    /// Number of ants per iteration.
    pub num_ants: usize,
    /// Pheromone influence (α).
    pub alpha: f64,
    /// Heuristic influence (β).
    pub beta: f64,
    /// Pheromone evaporation rate (ρ).
    pub evaporation: f64,
    /// Pheromone deposit factor (Q).
    pub q: f64,
    /// Initial pheromone value.
    pub initial_pheromone: f64,
    /// Maximum iterations.
    pub max_iterations: usize,
}

impl Default for AcoConfig {
    fn default() -> Self {
        Self {
            num_ants: 20,
            alpha: 1.0,
            beta: 3.0,
            evaporation: 0.5,
            q: 100.0,
            initial_pheromone: 0.1,
            max_iterations: 200,
        }
    }
}

/// Result of an ACO run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcoResult {
    /// Best path found (indices into the distance matrix).
    pub best_path: Vec<usize>,
    /// Length of the best path.
    pub best_length: f64,
    /// Pheromone matrix after convergence.
    pub pheromones: Vec<Vec<f64>>,
    /// History of best lengths per iteration.
    pub history: Vec<f64>,
}

/// Ant Colony Optimization for the Traveling Salesman Problem.
pub struct AntColonyOptimization {
    config: AcoConfig,
    distances: Vec<Vec<f64>>,
    pheromones: Vec<Vec<f64>>,
    num_cities: usize,
}

impl AntColonyOptimization {
    /// Create a new ACO instance with a square distance matrix.
    pub fn new(config: AcoConfig, distances: Vec<Vec<f64>>) -> Self {
        let n = distances.len();
        let pheromones = vec![vec![config.initial_pheromone; n]; n];
        Self {
            config,
            distances,
            pheromones,
            num_cities: n,
        }
    }

    /// Run the ACO algorithm.
    pub fn run(&mut self) -> AcoResult {
        let mut best_path: Vec<usize> = Vec::new();
        let mut best_length = f64::MAX;
        let mut history = Vec::new();

        for _ in 0..self.config.max_iterations {
            let mut all_paths = Vec::new();
            let mut all_lengths = Vec::new();

            for _ in 0..self.config.num_ants {
                let (path, length) = self.construct_solution();
                all_paths.push(path);
                all_lengths.push(length);
            }

            // Find best in this iteration
            let (iter_best_idx, &iter_best_len) = all_lengths
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();

            if iter_best_len < best_length {
                best_length = iter_best_len;
                best_path = all_paths[iter_best_idx].clone();
            }

            history.push(best_length);

            // Evaporate pheromones
            for i in 0..self.num_cities {
                for j in 0..self.num_cities {
                    self.pheromones[i][j] *= 1.0 - self.config.evaporation;
                }
            }

            // Deposit pheromones
            for (path, &length) in all_paths.iter().zip(all_lengths.iter()) {
                if length > 0.0 {
                    let deposit = self.config.q / length;
                    for k in 0..path.len() - 1 {
                        let i = path[k];
                        let j = path[k + 1];
                        self.pheromones[i][j] += deposit;
                        self.pheromones[j][i] += deposit;
                    }
                    // Close the loop
                    if path.len() > 1 {
                        let last = *path.last().unwrap();
                        let first = path[0];
                        self.pheromones[last][first] += deposit;
                        self.pheromones[first][last] += deposit;
                    }
                }
            }
        }

        AcoResult {
            best_path,
            best_length,
            pheromones: self.pheromones.clone(),
            history,
        }
    }

    fn construct_solution(&self) -> (Vec<usize>, f64) {
        let mut rng = rand::thread_rng();
        let mut visited = vec![false; self.num_cities];
        let start = rng.gen_range(0..self.num_cities);
        let mut path = vec![start];
        visited[start] = true;
        let mut total_length = 0.0;

        for _ in 1..self.num_cities {
            let current = *path.last().unwrap();
            let mut probabilities = Vec::new();
            let mut sum_prob = 0.0;

            for j in 0..self.num_cities {
                if !visited[j] && self.distances[current][j] > 0.0 {
                    let tau = self.pheromones[current][j].powf(self.config.alpha);
                    let eta = (1.0 / self.distances[current][j]).powf(self.config.beta);
                    let p = tau * eta;
                    probabilities.push((j, p));
                    sum_prob += p;
                } else {
                    probabilities.push((j, 0.0));
                }
            }

            // Select next city
            let next = if sum_prob > 0.0 {
                let r: f64 = rng.gen_range(0.0..sum_prob);
                let mut cumulative = 0.0;
                let mut chosen = 0;
                for &(j, p) in &probabilities {
                    if !visited[j] {
                        cumulative += p;
                        if cumulative >= r {
                            chosen = j;
                            break;
                        }
                    }
                }
                chosen
            } else {
                // Fallback: pick any unvisited
                (0..self.num_cities).find(|&j| !visited[j]).unwrap()
            };

            visited[next] = true;
            total_length += self.distances[current][next];
            path.push(next);
        }

        // Close the tour
        let last = *path.last().unwrap();
        total_length += self.distances[last][start];

        (path, total_length)
    }

    /// Get current pheromone matrix.
    pub fn pheromones(&self) -> &Vec<Vec<f64>> {
        &self.pheromones
    }
}
