# lau-swarm-intelligence

A Rust crate packing **nine swarm-intelligence algorithms** behind clean, serialisable configs and a uniform `run(objective) → result` API. Optimise continuous functions, solve the TSP, or simulate emergent flocking — all with zero external solver dependencies.

---

## What This Does

| Algorithm | Problem Type | Key Struct |
|---|---|---|
| **Ant Colony Optimization** (ACO) | Combinatorial — Travelling Salesman | `AntColonyOptimization` |
| **Particle Swarm Optimization** (PSO) | Continuous optimisation | `ParticleSwarmOptimization` |
| **Bee Algorithm** | Continuous optimisation | `BeeAlgorithm` |
| **Firefly Algorithm** | Continuous optimisation | `FireflyAlgorithm` |
| **Wolf Pack Algorithm** | Continuous optimisation | `WolfPackAlgorithm` |
| **Bacterial Foraging Optimization** (BFO) | Continuous optimisation | `BacterialForaging` |
| **Stochastic Diffusion Search** (SDS) | Continuous optimisation | `StochasticDiffusionSearch` |
| **Flocking (Boids)** | Emergent-behaviour simulation | `Flocking` |
| **Agent Swarm** | Role-based flocking coordination | `AgentSwarm` |

Every optimiser accepts a closure `Fn(&[f64]) -> f64` (minimisation) and returns a typed result struct carrying the best solution, fitness history, and algorithm-specific diagnostics. All configs and results derive `Serialize` / `Deserialize`.

---

## Key Idea

Swarm intelligence works by distributing search across many simple agents that communicate indirectly — through pheromone trails (ACO), velocity exchange (PSO), brightness (firefly), or spatial proximity (Boids). No single agent knows the global picture; the swarm *discovers* good solutions collectively. This crate captures that pattern in a uniform Rust API so you can swap algorithms without rewriting integration code.

---

## Install

```toml
[dependencies]
lau-swarm-intelligence = "0.1"
```

Requires Rust **2021 edition**. The crate depends on `serde`, `nalgebra`, and `rand`.

---

## Quick Start

```rust
use lau_swarm_intelligence::{ParticleSwarmOptimization, pso::PsoConfig};

fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|xi| xi * xi).sum()
}

fn main() {
    let config = PsoConfig {
        num_particles: 40,
        max_iterations: 200,
        bounds: vec![(-5.0, 5.0); 3], // 3-D search space
        ..Default::default()
    };

    let pso = ParticleSwarmOptimization::new(config);
    let result = pso.run(sphere);

    println!("Best position: {:?}", result.best_position);
    println!("Best fitness:  {:.6}", result.best_fitness);
    println!("History ({} iters): {:?}", result.history.len(), &result.history[..5]);
}
```

### Solving a TSP with ACO

```rust
use lau_swarm_intelligence::{AntColonyOptimization, aco::AcoConfig};

let distances = vec![
    vec![0.0, 2.0, 9.0, 10.0],
    vec![2.0, 0.0, 6.0,  4.0],
    vec![9.0, 6.0, 0.0,  8.0],
    vec![10.0,4.0, 8.0,  0.0],
];
let mut aco = AntColonyOptimization::new(AcoConfig::default(), distances);
let result = aco.run();
println!("Best tour: {:?}, length: {:.2}", result.best_path, result.best_length);
```

### Running a Boids simulation

```rust
use lau_swarm_intelligence::{Flocking, flocking::FlockingConfig};

let flocking = Flocking::new(FlockingConfig { num_boids: 100, ..Default::default() });
let mut boids = flocking.init_flock();
for _ in 0..500 {
    flocking.step(&mut boids);
}
```

---

## API Reference

### ACO — `aco` module

| Item | Description |
|---|---|
| `AcoConfig` | Number of ants, α/β, evaporation ρ, deposit Q, initial pheromone, max iterations |
| `AntColonyOptimization::new(config, distances)` | Construct with an N×N distance matrix |
| `.run() → AcoResult` | Returns best path, length, pheromone matrix, convergence history |
| `.pheromones()` | Inspect current pheromone trails |

### PSO — `pso` module

| Item | Description |
|---|---|
| `PsoConfig` | Particles, inertia, cognitive/social coefficients, bounds, max iterations |
| `Particle` | Position, velocity, personal-best position & fitness |
| `ParticleSwarmOptimization::new(config)` | Create solver |
| `.run(objective) → PsoResult` | Minimise a closure; returns best position/fitness, history, final particles |

### Bee Algorithm — `bee` module

| Item | Description |
|---|---|
| `BeeConfig` | Scouts, elite/selected sites, foragers per site, neighbourhood radius, stagnation limit |
| `FoodSource` | Position, fitness, stagnation counter |
| `BeeAlgorithm::new(config)` | Create solver |
| `.run(objective) → BeeResult` | Best position, fitness, history |

### Firefly — `firefly` module

| Item | Description |
|---|---|
| `FireflyConfig` | Population, β₀, γ (absorption), α (randomness), bounds |
| `Firefly` | Position, brightness |
| `FireflyAlgorithm::new(config)` | Create solver |
| `.run(objective) → FireflyResult` | Best position/fitness, history, final firefly positions |
| `.attractiveness(distance) → f64` | Compute β(r) = β₀ · e^(−γr²) |

### Wolf Pack — `wolf_pack` module

| Item | Description |
|---|---|
| `WolfPackConfig` | Wolves, leaders, scouts, step size, scout range, bounds |
| `Wolf` / `WolfRole` | Position, fitness, role (Leader / Scout / Follower) |
| `WolfPackAlgorithm::new(config)` | Create solver |
| `.run(objective) → WolfPackResult` | Best position, fitness, history |

### Bacterial Foraging — `bacterial` module

| Item | Description |
|---|---|
| `BacterialConfig` | Bacteria count, chemotactic/swim/reproduction/elimination steps, step size, attractant/repellent params |
| `Bacterium` | Position, fitness, health |
| `BacterialForaging::new(config)` | Create solver |
| `.run(objective) → BacterialResult` | Best position, fitness, history |

### SDS — `sds` module

| Item | Description |
|---|---|
| `SdsConfig` | Agents, iterations, activation threshold, bounds |
| `SdsAgent` | Hypothesis, active flag, fitness |
| `StochasticDiffusionSearch::new(config)` | Create solver |
| `.run(objective) → SdsResult` | Best position, fitness, history, cluster counts per iteration |

### Flocking — `flocking` module

| Item | Description |
|---|---|
| `FlockingConfig` | Boid count, separation/alignment/cohesion weights, radii, max speed/force, world size |
| `Boid` | Position, velocity |
| `Flocking::new(config)` | Create simulation |
| `.init_flock() → Vec<Boid>` | Random initial boids |
| `.step(&mut boids)` | Advance one tick |
| `.compute_flocking_forces(boids, idx) → (sep, ali, coh)` | Inspect raw forces |

### Agent Swarm — `agent_swarm` module

| Item | Description |
|---|---|
| `AgentSwarmConfig` | Agent count, nested `FlockingConfig` |
| `SwarmAgent` / `AgentRole` | ID, boid, role (Explorer / Worker / Coordinator), task score |
| `AgentSwarm::new(config)` | Create swarm |
| `.step() → SwarmState` | One tick; returns agents, average cohesion & alignment |

---

## How It Works

### Ant Colony Optimization
Artificial ants build tours through a weighted graph. Each ant selects the next city with probability proportional to τᵢⱼᵅ · ηᵢⱼᵝ, where τ is pheromone and η = 1/distance is a heuristic. After all ants complete a tour, pheromones evaporate (τ ← τ · (1−ρ)) and are reinforced along the best paths (τᵢⱼ += Q/L). Over iterations, short edges accumulate more pheromone, guiding future ants.

### Particle Swarm Optimization
Each particle maintains a position **x**, velocity **v**, and personal-best position **p**. The swarm also tracks the global best **g**. On each iteration:
```
v ← ω·v + c₁·r₁·(p − x) + c₂·r₂·(g − x)
x ← x + v
```
where ω is inertia, c₁/c₂ are cognitive/social coefficients, and r₁,r₂ ~ U(0,1). The clamping to bounds prevents escape.

### Bee Algorithm
Scout bees sample random positions. The best *e* become *elite sites* and receive many foragers; the next *m* are *selected sites* with fewer foragers. Foragers explore within a neighbourhood radius of their site. Stagnant sites (no improvement for *k* iterations) are abandoned and replaced by fresh scouts.

### Firefly Algorithm
Every firefly is attracted to every brighter firefly. Attractiveness decays exponentially with distance: β(r) = β₀ · e^(−γr²). A less-bright firefly *i* moves toward a brighter *j*:
```
xᵢ ← xᵢ + β(rᵢⱼ)·(xⱼ − xᵢ) + α·ε
```
where ε ~ U(−0.5, 0.5) adds randomness. In minimisation, lower fitness = brighter.

### Wolf Pack Algorithm
Wolves are ranked by fitness into leaders (local search), scouts (wide exploration), and followers (move toward nearest leader). Leaders take small steps; scouts range farther; followers converge on leaders, creating a balance of exploration and exploitation.

### Bacterial Foraging Optimization
Four nested loops — elimination-dispersal > reproduction > chemotaxis > swim:
1. **Tumble**: pick a random unit direction.
2. **Swim**: take steps of size Cᵢ in that direction as long as fitness improves.
3. **Swarming**: cell-to-cell attractant/repellent modifies the objective: Jᵢ += −d_attract·e^(−w_attract·Δ²) + h_repel·e^(−w_repel·Δ²).
4. **Reproduction**: the healthiest half clones itself, replacing the worst half.
5. **Elimination-dispersal**: each bacterium is randomly repositioned with probability p_ed.

### Stochastic Diffusion Search
Agents hold hypotheses (candidate solutions). Each iteration:
1. **Test**: an agent becomes *active* if its fitness beats a threshold or with probability 0.5.
2. **Diffusion**: inactive agents adopt the hypothesis of a random active agent, or generate a new random hypothesis.
3. **Context-sensitive perturbation**: active agents make small local refinements, keeping improvements.

Over time, agents cluster around the best region.

### Flocking (Boids)
Craig Reynolds' three rules, computed per-boid over neighbours within a perception radius:
- **Separation**: steer away from neighbours within the separation radius.
- **Alignment**: steer toward the average heading of neighbours.
- **Cohesion**: steer toward the centre of mass of neighbours.

Forces are clamped to `max_force` and velocities to `max_speed`. Positions wrap around the world edges.

### Agent Swarm
Layers the Boids flocking engine with three agent roles: **Explorers** (speed boost), **Workers** (default cohesion), and **Coordinators** (dampened velocity to stay central). Each step returns metrics — average cohesion (mean distance to centroid) and alignment (mean cosine similarity with the flock's mean velocity).

---

## The Math

**ACO transition probability:**
$$P(i \to j) = \frac{\tau_{ij}^\alpha \cdot \eta_{ij}^\beta}{\sum_{k \notin \text{visited}} \tau_{ik}^\alpha \cdot \eta_{ik}^\beta}$$

**ACO pheromone update:**
$$\tau_{ij} \leftarrow (1-\rho)\,\tau_{ij} + \sum_{\text{ants}} \frac{Q}{L_{\text{ant}}} \cdot \mathbb{1}[(i,j) \in \text{path}]$$

**PSO velocity:**
$$v_d^{(t+1)} = \omega\,v_d^{(t)} + c_1\,r_1\,(p_d - x_d) + c_2\,r_2\,(g_d - x_d)$$

**Firefly attractiveness:**
$$\beta(r) = \beta_0\,e^{-\gamma r^2}$$

**BFO swarming (cell-to-cell signalling):**
$$J_{\text{swarm}}(i) = \sum_{j \neq i} \left[-d_{\text{attract}}\,e^{-w_{\text{attract}}\|\mathbf{x}_i - \mathbf{x}_j\|^2} + h_{\text{repel}}\,e^{-w_{\text{repel}}\|\mathbf{x}_i - \mathbf{x}_j\|^2}\right]$$

**Boids forces:**
$$\mathbf{F}_{\text{sep}} = \sum_{\|\mathbf{d}_{ij}\| < r_{\text{sep}}} \frac{\mathbf{x}_i - \mathbf{x}_j}{\|\mathbf{x}_i - \mathbf{x}_j\|}, \quad \mathbf{F}_{\text{ali}} = \frac{1}{|N_i|}\sum_{j \in N_i} \mathbf{v}_j, \quad \mathbf{F}_{\text{coh}} = \frac{1}{|N_i|}\sum_{j \in N_i} \mathbf{x}_j - \mathbf{x}_i$$

---

## Tests

The crate ships **48 integration tests** covering convergence, bounds enforcement, history monotonicity, force correctness, serde round-trips, and a cross-algorithm smoke test. Run them with:

```bash
cargo test
```

---

## License

MIT
