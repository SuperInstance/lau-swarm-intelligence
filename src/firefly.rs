//! Firefly Algorithm — attractiveness based on distance and brightness.

use rand::Rng;
use serde::{Deserialize, Serialize};

/// Configuration for the Firefly Algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FireflyConfig {
    /// Number of fireflies.
    pub num_fireflies: usize,
    /// Base attractiveness at distance zero (β₀).
    pub beta0: f64,
    /// Light absorption coefficient (γ).
    pub gamma: f64,
    /// Randomness parameter (α).
    pub alpha: f64,
    /// Max iterations.
    pub max_iterations: usize,
    /// Search space bounds.
    pub bounds: Vec<(f64, f64)>,
}

impl Default for FireflyConfig {
    fn default() -> Self {
        Self {
            num_fireflies: 25,
            beta0: 1.0,
            gamma: 1.0,
            alpha: 0.2,
            max_iterations: 100,
            bounds: vec![(-5.0, 5.0)],
        }
    }
}

/// A single firefly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Firefly {
    pub position: Vec<f64>,
    pub brightness: f64,
}

/// Result of the Firefly Algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FireflyResult {
    pub best_position: Vec<f64>,
    pub best_fitness: f64,
    pub history: Vec<f64>,
    pub fireflies: Vec<Firefly>,
}

/// Firefly Algorithm.
pub struct FireflyAlgorithm {
    config: FireflyConfig,
    dims: usize,
}

impl FireflyAlgorithm {
    pub fn new(config: FireflyConfig) -> Self {
        let dims = config.bounds.len();
        Self { config, dims }
    }

    /// Run the firefly algorithm minimizing the objective.
    pub fn run<F>(&self, objective: F) -> FireflyResult
    where
        F: Fn(&[f64]) -> f64,
    {
        let mut rng = rand::thread_rng();
        let mut fireflies: Vec<Firefly> = (0..self.config.num_fireflies)
            .map(|_| {
                let pos = self.random_position(&mut rng);
                let brightness = objective(&pos);
                Firefly {
                    position: pos,
                    brightness,
                }
            })
            .collect();

        let mut history = Vec::new();

        for _ in 0..self.config.max_iterations {
            for i in 0..fireflies.len() {
                for j in 0..fireflies.len() {
                    // Lower fitness = brighter (minimization)
                    if fireflies[j].brightness < fireflies[i].brightness {
                        let dist = self.distance(&fireflies[i].position, &fireflies[j].position);
                        let beta = self.config.beta0
                            * (-self.config.gamma * dist * dist).exp();

                        for d in 0..self.dims {
                            let epsilon: f64 = rng.gen_range(-0.5..0.5);
                            fireflies[i].position[d] = fireflies[i].position[d]
                                + beta * (fireflies[j].position[d] - fireflies[i].position[d])
                                + self.config.alpha * epsilon;

                            fireflies[i].position[d] = fireflies[i].position[d]
                                .clamp(self.config.bounds[d].0, self.config.bounds[d].1);
                        }

                        fireflies[i].brightness = objective(&fireflies[i].position);
                    }
                }
            }

            // Track best
            let best = fireflies
                .iter()
                .min_by(|a, b| a.brightness.partial_cmp(&b.brightness).unwrap())
                .unwrap();
            history.push(best.brightness);
        }

        let best = fireflies
            .iter()
            .min_by(|a, b| a.brightness.partial_cmp(&b.brightness).unwrap())
            .unwrap();

        FireflyResult {
            best_position: best.position.clone(),
            best_fitness: best.brightness,
            history,
            fireflies,
        }
    }

    /// Compute attractiveness β at distance r.
    pub fn attractiveness(&self, distance: f64) -> f64 {
        self.config.beta0 * (-self.config.gamma * distance * distance).exp()
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
