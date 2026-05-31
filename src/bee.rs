//! Bee Algorithm — scout/forager roles with neighborhood search.

use rand::Rng;
use serde::{Deserialize, Serialize};

/// Configuration for the Bee Algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeeConfig {
    /// Number of scout bees.
    pub num_scouts: usize,
    /// Number of elite sites.
    pub num_elite: usize,
    /// Number of foragers per elite site.
    pub foragers_elite: usize,
    /// Number of foragers per non-elite selected site.
    pub foragers_other: usize,
    /// Neighborhood search radius.
    pub neighborhood_radius: f64,
    /// Max iterations.
    pub max_iterations: usize,
    /// Search space bounds: (min, max) per dimension.
    pub bounds: Vec<(f64, f64)>,
    /// Number of selected sites (total, including elite).
    pub num_selected: usize,
    /// Stagnation limit before abandoning a site.
    pub stagnation_limit: usize,
}

impl Default for BeeConfig {
    fn default() -> Self {
        Self {
            num_scouts: 30,
            num_elite: 3,
            foragers_elite: 10,
            foragers_other: 5,
            neighborhood_radius: 1.0,
            max_iterations: 100,
            bounds: vec![(-5.0, 5.0)],
            num_selected: 10,
            stagnation_limit: 10,
        }
    }
}

/// A food source (solution).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoodSource {
    pub position: Vec<f64>,
    pub fitness: f64,
    pub stagnation: usize,
}

/// Result of the bee algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeeResult {
    pub best_position: Vec<f64>,
    pub best_fitness: f64,
    pub history: Vec<f64>,
}

/// The Bee Algorithm.
pub struct BeeAlgorithm {
    config: BeeConfig,
    dims: usize,
}

impl BeeAlgorithm {
    pub fn new(config: BeeConfig) -> Self {
        let dims = config.bounds.len();
        Self { config, dims }
    }

    /// Run the bee algorithm minimizing the objective.
    pub fn run<F>(&self, objective: F) -> BeeResult
    where
        F: Fn(&[f64]) -> f64,
    {
        let mut rng = rand::thread_rng();
        let mut sites: Vec<FoodSource> = Vec::new();
        let mut history = Vec::new();

        // Initial scout phase
        for _ in 0..self.config.num_scouts {
            let pos = self.random_position(&mut rng);
            let fit = objective(&pos);
            sites.push(FoodSource {
                position: pos,
                fitness: fit,
                stagnation: 0,
            });
        }

        let mut best_pos = sites[0].position.clone();
        let mut best_fit = sites[0].fitness;
        for s in &sites {
            if s.fitness < best_fit {
                best_fit = s.fitness;
                best_pos = s.position.clone();
            }
        }

        for _ in 0..self.config.max_iterations {
            // Sort sites by fitness (ascending = minimizing)
            sites.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());

            let mut new_sites = Vec::new();

            // Elite sites neighborhood search
            let elite_count = self.config.num_elite.min(sites.len());
            for i in 0..elite_count {
                let best_in_site = self.neighborhood_search(&sites[i], self.config.foragers_elite, &objective, &mut rng);
                if best_in_site.fitness <= sites[i].fitness {
                    new_sites.push(FoodSource {
                        position: best_in_site.position,
                        fitness: best_in_site.fitness,
                        stagnation: 0,
                    });
                } else {
                    let mut s = sites[i].clone();
                    s.stagnation += 1;
                    new_sites.push(s);
                }
            }

            // Selected (non-elite) sites
            let selected_count = self.config.num_selected.min(sites.len());
            for i in elite_count..selected_count {
                let best_in_site = self.neighborhood_search(&sites[i], self.config.foragers_other, &objective, &mut rng);
                if best_in_site.fitness <= sites[i].fitness {
                    new_sites.push(FoodSource {
                        position: best_in_site.position,
                        fitness: best_in_site.fitness,
                        stagnation: 0,
                    });
                } else {
                    let mut s = sites[i].clone();
                    s.stagnation += 1;
                    new_sites.push(s);
                }
            }

            // Abandon stagnant sites and send new scouts
            let remaining_scouts = self.config.num_scouts - selected_count;
            for _ in 0..remaining_scouts {
                let pos = self.random_position(&mut rng);
                let fit = objective(&pos);
                new_sites.push(FoodSource {
                    position: pos,
                    fitness: fit,
                    stagnation: 0,
                });
            }

            // Abandon sites that exceeded stagnation limit
            for site in &mut new_sites {
                if site.stagnation >= self.config.stagnation_limit {
                    let pos = self.random_position(&mut rng);
                    let fit = objective(&pos);
                    site.position = pos;
                    site.fitness = fit;
                    site.stagnation = 0;
                }
            }

            sites = new_sites;

            // Track best
            for s in &sites {
                if s.fitness < best_fit {
                    best_fit = s.fitness;
                    best_pos = s.position.clone();
                }
            }
            history.push(best_fit);
        }

        BeeResult {
            best_position: best_pos,
            best_fitness: best_fit,
            history,
        }
    }

    fn neighborhood_search<F>(
        &self,
        site: &FoodSource,
        num_foragers: usize,
        objective: &F,
        rng: &mut impl Rng,
    ) -> FoodSource
    where
        F: Fn(&[f64]) -> f64,
    {
        let mut best = site.clone();
        for _ in 0..num_foragers {
            let mut new_pos = site.position.clone();
            for d in 0..self.dims {
                let delta: f64 = rng.gen_range(-self.config.neighborhood_radius..self.config.neighborhood_radius);
                new_pos[d] += delta;
                new_pos[d] = new_pos[d].clamp(self.config.bounds[d].0, self.config.bounds[d].1);
            }
            let fit = objective(&new_pos);
            if fit < best.fitness {
                best = FoodSource {
                    position: new_pos,
                    fitness: fit,
                    stagnation: 0,
                };
            }
        }
        best
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
