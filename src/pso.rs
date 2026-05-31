//! Particle Swarm Optimization — velocity/position update with inertia.

use rand::Rng;
use serde::{Deserialize, Serialize};

/// Configuration for PSO.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PsoConfig {
    /// Number of particles.
    pub num_particles: usize,
    /// Inertia weight.
    pub inertia: f64,
    /// Cognitive coefficient (personal best pull).
    pub cognitive: f64,
    /// Social coefficient (global best pull).
    pub social: f64,
    /// Maximum iterations.
    pub max_iterations: usize,
    /// Search space bounds: (min, max) per dimension.
    pub bounds: Vec<(f64, f64)>,
}

impl Default for PsoConfig {
    fn default() -> Self {
        Self {
            num_particles: 30,
            inertia: 0.729,
            cognitive: 1.49445,
            social: 1.49445,
            max_iterations: 200,
            bounds: vec![(-5.0, 5.0)],
        }
    }
}

/// A single particle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Particle {
    pub position: Vec<f64>,
    pub velocity: Vec<f64>,
    pub best_position: Vec<f64>,
    pub best_fitness: f64,
    pub fitness: f64,
}

/// Result of a PSO run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PsoResult {
    pub best_position: Vec<f64>,
    pub best_fitness: f64,
    pub history: Vec<f64>,
    pub particles: Vec<Particle>,
}

/// Particle Swarm Optimization.
pub struct ParticleSwarmOptimization {
    config: PsoConfig,
    dims: usize,
}

impl ParticleSwarmOptimization {
    pub fn new(config: PsoConfig) -> Self {
        let dims = config.bounds.len();
        Self { config, dims }
    }

    /// Run PSO minimizing the given objective function.
    pub fn run<F>(&self, objective: F) -> PsoResult
    where
        F: Fn(&[f64]) -> f64,
    {
        let mut rng = rand::thread_rng();
        let mut particles = self.init_particles(&mut rng);

        let mut global_best_pos = particles[0].best_position.clone();
        let mut global_best_fit = particles[0].best_fitness;
        let mut history = Vec::new();

        for p in &particles {
            if p.best_fitness < global_best_fit {
                global_best_fit = p.best_fitness;
                global_best_pos = p.best_position.clone();
            }
        }

        for _ in 0..self.config.max_iterations {
            for particle in &mut particles {
                // Update velocity
                for d in 0..self.dims {
                    let r1: f64 = rng.gen_range(0.0..1.0);
                    let r2: f64 = rng.gen_range(0.0..1.0);
                    particle.velocity[d] = self.config.inertia * particle.velocity[d]
                        + self.config.cognitive * r1 * (particle.best_position[d] - particle.position[d])
                        + self.config.social * r2 * (global_best_pos[d] - particle.position[d]);
                }

                // Clamp velocity
                for d in 0..self.dims {
                    let range = self.config.bounds[d].1 - self.config.bounds[d].0;
                    let max_vel = range * 0.5;
                    particle.velocity[d] = particle.velocity[d].clamp(-max_vel, max_vel);
                }

                // Update position
                for d in 0..self.dims {
                    particle.position[d] += particle.velocity[d];
                    // Clamp to bounds
                    particle.position[d] = particle.position[d]
                        .clamp(self.config.bounds[d].0, self.config.bounds[d].1);
                }

                // Evaluate
                particle.fitness = objective(&particle.position);
                if particle.fitness < particle.best_fitness {
                    particle.best_fitness = particle.fitness;
                    particle.best_position = particle.position.clone();
                    if particle.fitness < global_best_fit {
                        global_best_fit = particle.fitness;
                        global_best_pos = particle.position.clone();
                    }
                }
            }
            history.push(global_best_fit);
        }

        PsoResult {
            best_position: global_best_pos,
            best_fitness: global_best_fit,
            history,
            particles,
        }
    }

    fn init_particles(&self, rng: &mut impl Rng) -> Vec<Particle> {
        let mut particles = Vec::with_capacity(self.config.num_particles);
        for _ in 0..self.config.num_particles {
            let mut position = Vec::with_capacity(self.dims);
            let mut velocity = Vec::with_capacity(self.dims);
            for d in 0..self.dims {
                let (lo, hi) = self.config.bounds[d];
                let pos: f64 = rng.gen_range(lo..hi);
                let vel: f64 = rng.gen_range(-0.1..0.1);
                position.push(pos);
                velocity.push(vel);
            }
            // Dummy fitness — will be computed outside
            particles.push(Particle {
                position: position.clone(),
                velocity,
                best_position: position.clone(),
                best_fitness: f64::MAX,
                fitness: f64::MAX,
            });
        }
        particles
    }
}
