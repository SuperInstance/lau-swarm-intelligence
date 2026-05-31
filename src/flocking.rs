//! Flocking (Boids) — separation, alignment, cohesion.

use serde::{Deserialize, Serialize};

/// Configuration for the Boids flocking simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlockingConfig {
    /// Number of boids.
    pub num_boids: usize,
    /// Separation weight.
    pub separation_weight: f64,
    /// Alignment weight.
    pub alignment_weight: f64,
    /// Cohesion weight.
    pub cohesion_weight: f64,
    /// Perception radius.
    pub perception_radius: f64,
    /// Separation radius.
    pub separation_radius: f64,
    /// Maximum speed.
    pub max_speed: f64,
    /// Maximum force (steering).
    pub max_force: f64,
    /// World bounds.
    pub world_size: (f64, f64),
}

impl Default for FlockingConfig {
    fn default() -> Self {
        Self {
            num_boids: 50,
            separation_weight: 1.5,
            alignment_weight: 1.0,
            cohesion_weight: 1.0,
            perception_radius: 50.0,
            separation_radius: 25.0,
            max_speed: 4.0,
            max_force: 0.3,
            world_size: (800.0, 600.0),
        }
    }
}

/// A single boid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Boid {
    pub position: Vec<f64>,
    pub velocity: Vec<f64>,
}

impl Boid {
    pub fn new(position: Vec<f64>, velocity: Vec<f64>) -> Self {
        Self { position, velocity }
    }
}

/// Flocking simulation (Reynolds Boids).
pub struct Flocking {
    config: FlockingConfig,
    dims: usize,
}

impl Flocking {
    pub fn new(config: FlockingConfig) -> Self {
        let dims = 2; // 2D by default
        Self { config, dims }
    }

    /// Initialize a random flock.
    pub fn init_flock(&self) -> Vec<Boid> {
        let mut rng = rand::thread_rng();
        (0..self.config.num_boids)
            .map(|_| {
                let pos = vec![
                    rand::Rng::gen_range(&mut rng, 0.0..self.config.world_size.0),
                    rand::Rng::gen_range(&mut rng, 0.0..self.config.world_size.1),
                ];
                let vel = vec![
                    rand::Rng::gen_range(&mut rng, -1.0..1.0),
                    rand::Rng::gen_range(&mut rng, -1.0..1.0),
                ];
                Boid::new(pos, vel)
            })
            .collect()
    }

    /// Step the simulation forward.
    pub fn step(&self, boids: &mut Vec<Boid>) {
        let n = boids.len();
        let mut accelerations = vec![vec![0.0; self.dims]; n];

        for i in 0..n {
            let (sep, ali, coh) = self.compute_flocking_forces(boids, i);

            for d in 0..self.dims {
                accelerations[i][d] = self.config.separation_weight * sep[d]
                    + self.config.alignment_weight * ali[d]
                    + self.config.cohesion_weight * coh[d];
            }
        }

        // Apply acceleration
        for i in 0..n {
            for d in 0..self.dims {
                boids[i].velocity[d] += accelerations[i][d];
            }
            // Limit speed (magnitude)
            let speed: f64 = boids[i].velocity.iter().map(|v| v * v).sum::<f64>().sqrt();
            if speed > self.config.max_speed {
                for d in 0..self.dims {
                    boids[i].velocity[d] = boids[i].velocity[d] / speed * self.config.max_speed;
                }
            }
            for d in 0..self.dims {
                boids[i].position[d] += boids[i].velocity[d];
            }

            // Wrap around world
            boids[i].position[0] =
                (boids[i].position[0] + self.config.world_size.0) % self.config.world_size.0;
            boids[i].position[1] =
                (boids[i].position[1] + self.config.world_size.1) % self.config.world_size.1;
        }
    }

    /// Compute the three flocking forces for a boid.
    pub fn compute_flocking_forces(
        &self,
        boids: &[Boid],
        idx: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let dims = self.dims;
        let mut separation = vec![0.0; dims];
        let mut alignment = vec![0.0; dims];
        let mut cohesion = vec![0.0; dims];
        let mut sep_count = 0;
        let mut neighbor_count = 0;

        for (j, other) in boids.iter().enumerate() {
            if j == idx {
                continue;
            }
            let dist = self.distance(&boids[idx].position, &other.position);

            // Separation
            if dist < self.config.separation_radius && dist > 0.0 {
                for d in 0..dims {
                    separation[d] += (boids[idx].position[d] - other.position[d]) / dist;
                }
                sep_count += 1;
            }

            // Alignment & Cohesion (within perception radius)
            if dist < self.config.perception_radius {
                for d in 0..dims {
                    alignment[d] += other.velocity[d];
                    cohesion[d] += other.position[d];
                }
                neighbor_count += 1;
            }
        }

        // Average separation
        if sep_count > 0 {
            for d in 0..dims {
                separation[d] /= sep_count as f64;
            }
            self.limit_force(&mut separation);
        }

        // Average alignment
        if neighbor_count > 0 {
            for d in 0..dims {
                alignment[d] /= neighbor_count as f64;
            }
            self.limit_force(&mut alignment);
        }

        // Cohesion: steer toward center
        if neighbor_count > 0 {
            for d in 0..dims {
                cohesion[d] = cohesion[d] / neighbor_count as f64 - boids[idx].position[d];
            }
            self.limit_force(&mut cohesion);
        }

        (separation, alignment, cohesion)
    }

    fn limit_force(&self, force: &mut [f64]) {
        let mag: f64 = force.iter().map(|f| f * f).sum::<f64>().sqrt();
        if mag > self.config.max_force {
            for f in force.iter_mut() {
                *f = *f / mag * self.config.max_force;
            }
        }
    }

    fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    pub fn config(&self) -> &FlockingConfig {
        &self.config
    }
}
