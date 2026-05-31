# lau-swarm-intelligence

Swarm intelligence algorithms in Rust — collective behavior emerging from simple agent rules.

## Algorithms

- **Ant Colony Optimization (ACO)** — Pheromone trails, path construction, TSP solver
- **Particle Swarm Optimization (PSO)** — Velocity/position update, inertia weight, cognitive/social components
- **Bee Algorithm** — Scout/forager roles, elite site neighborhood search
- **Firefly Algorithm** — Distance-based attractiveness, brightness-driven movement
- **Wolf Pack Algorithm** — Leaders, scouts, and followers with role-based movement
- **Bacterial Foraging Optimization** — Chemotaxis, swarming, reproduction, elimination-dispersal
- **Stochastic Diffusion Search (SDS)** — Hypothesis-based search with agent diffusion
- **Flocking (Boids)** — Separation, alignment, cohesion rules
- **Agent Swarm Coordination** — Emergent collective behavior in agent fleets

## Usage

```rust
use lau_swarm_intelligence::pso::{ParticleSwarmOptimization, PsoConfig};

fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|xi| xi * xi).sum()
}

fn main() {
    let config = PsoConfig {
        num_particles: 30,
        max_iterations: 200,
        bounds: vec![(-5.0, 5.0), (-5.0, 5.0)],
        ..Default::default()
    };
    let pso = ParticleSwarmOptimization::new(config);
    let result = pso.run(sphere);
    println!("Best fitness: {}", result.best_fitness);
    println!("Best position: {:?}", result.best_position);
}
```

## Dependencies

- `serde` — Serialization of configs and results
- `nalgebra` — Linear algebra primitives
- `rand` — Random number generation

## License

MIT
