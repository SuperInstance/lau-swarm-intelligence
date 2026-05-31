use lau_swarm_intelligence::*;
use approx::assert_relative_eq;

fn simple_distance_matrix() -> Vec<Vec<f64>> {
    vec![
        vec![0.0, 1.0, 2.0, 3.0, 4.0],
        vec![1.0, 0.0, 1.0, 2.0, 3.0],
        vec![2.0, 1.0, 0.0, 1.0, 2.0],
        vec![3.0, 2.0, 1.0, 0.0, 1.0],
        vec![4.0, 3.0, 2.0, 1.0, 0.0],
    ]
}

fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|xi| xi * xi).sum()
}

fn rastrigin(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let sum: f64 = x.iter().map(|xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos()).sum();
    10.0 * n + sum
}

// === ACO Tests ===

#[test]
fn test_aco_convergence() {
    let config = aco::AcoConfig {
        num_ants: 10,
        max_iterations: 50,
        ..Default::default()
    };
    let mut aco = aco::AntColonyOptimization::new(config, simple_distance_matrix());
    let result = aco.run();
    assert!(result.best_length <= 10.0, "got {}", result.best_length);
    assert_eq!(result.best_path.len(), 5);
    let mut visited = result.best_path.clone();
    visited.sort();
    assert_eq!(visited, vec![0, 1, 2, 3, 4]);
}

#[test]
fn test_aco_history_decreases() {
    let config = aco::AcoConfig { num_ants: 10, max_iterations: 30, ..Default::default() };
    let mut aco = aco::AntColonyOptimization::new(config, simple_distance_matrix());
    let result = aco.run();
    for i in 1..result.history.len() {
        assert!(result.history[i] <= result.history[i - 1] + 1e-10);
    }
}

#[test]
fn test_aco_pheromone_matrix_size() {
    let config = aco::AcoConfig { num_ants: 5, max_iterations: 5, ..Default::default() };
    let mut aco = aco::AntColonyOptimization::new(config, simple_distance_matrix());
    let result = aco.run();
    assert_eq!(result.pheromones.len(), 5);
    assert_eq!(result.pheromones[0].len(), 5);
}

#[test]
fn test_aco_all_pheromones_positive() {
    let config = aco::AcoConfig { num_ants: 5, max_iterations: 5, ..Default::default() };
    let mut aco = aco::AntColonyOptimization::new(config, simple_distance_matrix());
    let result = aco.run();
    for row in &result.pheromones {
        for &val in row { assert!(val >= 0.0); }
    }
}

#[test]
fn test_aco_symmetric_pheromones() {
    let config = aco::AcoConfig { num_ants: 5, max_iterations: 10, ..Default::default() };
    let mut aco = aco::AntColonyOptimization::new(config, simple_distance_matrix());
    let result = aco.run();
    let n = result.pheromones.len();
    for i in 0..n {
        for j in 0..n {
            assert_relative_eq!(result.pheromones[i][j], result.pheromones[j][i], epsilon = 1e-10);
        }
    }
}

#[test]
fn test_aco_path_is_valid_tour() {
    let config = aco::AcoConfig { num_ants: 10, max_iterations: 20, ..Default::default() };
    let mut aco = aco::AntColonyOptimization::new(config, simple_distance_matrix());
    let result = aco.run();
    let mut sorted = result.best_path.clone();
    sorted.sort();
    assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
}

// === PSO Tests ===

#[test]
fn test_pso_finds_sphere_minimum() {
    let config = pso::PsoConfig {
        num_particles: 30, max_iterations: 200,
        bounds: vec![(-5.0, 5.0); 2], ..Default::default()
    };
    let pso = pso::ParticleSwarmOptimization::new(config);
    let result = pso.run(sphere);
    assert!(result.best_fitness < 0.1, "got {}", result.best_fitness);
}

#[test]
fn test_pso_velocity_update() {
    let config = pso::PsoConfig {
        num_particles: 10, max_iterations: 1,
        bounds: vec![(-5.0, 5.0); 2],
        inertia: 0.5, cognitive: 1.0, social: 1.0, ..Default::default()
    };
    let pso = pso::ParticleSwarmOptimization::new(config);
    let result = pso.run(sphere);
    assert_eq!(result.particles.len(), 10);
    for p in &result.particles { assert_eq!(p.velocity.len(), 2); }
}

#[test]
fn test_pso_history_length() {
    let config = pso::PsoConfig {
        num_particles: 5, max_iterations: 50,
        bounds: vec![(-5.0, 5.0); 2], ..Default::default()
    };
    let pso = pso::ParticleSwarmOptimization::new(config);
    let result = pso.run(sphere);
    assert_eq!(result.history.len(), 50);
}

#[test]
fn test_pso_best_within_bounds() {
    let bounds = vec![(-5.0, 5.0); 2];
    let pso = pso::ParticleSwarmOptimization::new(pso::PsoConfig {
        num_particles: 10, max_iterations: 50,
        bounds: bounds.clone(), ..Default::default()
    });
    let result = pso.run(sphere);
    for (val, (lo, hi)) in result.best_position.iter().zip(bounds.iter()) {
        assert!(*val >= *lo && *val <= *hi);
    }
}

#[test]
fn test_pso_particle_best_initialized() {
    let config = pso::PsoConfig {
        num_particles: 10, max_iterations: 5,
        bounds: vec![(-5.0, 5.0); 2], ..Default::default()
    };
    let pso = pso::ParticleSwarmOptimization::new(config);
    let result = pso.run(sphere);
    for p in &result.particles {
        assert!(p.best_fitness < f64::MAX);
        assert_eq!(p.best_position.len(), 2);
    }
}

#[test]
fn test_pso_convergence_on_rastrigin() {
    let config = pso::PsoConfig {
        num_particles: 40, max_iterations: 300,
        bounds: vec![(-5.12, 5.12); 2], ..Default::default()
    };
    let pso = pso::ParticleSwarmOptimization::new(config);
    let result = pso.run(rastrigin);
    assert!(result.best_fitness < 5.0, "got {}", result.best_fitness);
}

// === Bee Algorithm Tests ===

#[test]
fn test_bee_finds_minimum() {
    let config = bee::BeeConfig {
        num_scouts: 20, max_iterations: 50,
        bounds: vec![(-5.0, 5.0); 2], ..Default::default()
    };
    let bee = bee::BeeAlgorithm::new(config);
    let result = bee.run(sphere);
    assert!(result.best_fitness < 1.0, "got {}", result.best_fitness);
}

#[test]
fn test_bee_history_decreases() {
    let config = bee::BeeConfig {
        num_scouts: 10, max_iterations: 30,
        bounds: vec![(-5.0, 5.0); 2], ..Default::default()
    };
    let bee = bee::BeeAlgorithm::new(config);
    let result = bee.run(sphere);
    for i in 1..result.history.len() {
        assert!(result.history[i] <= result.history[i - 1] + 1e-10);
    }
}

#[test]
fn test_bee_neighborhood_search() {
    let config = bee::BeeConfig {
        num_scouts: 15, neighborhood_radius: 0.5,
        max_iterations: 40, bounds: vec![(-5.0, 5.0); 2], ..Default::default()
    };
    let bee = bee::BeeAlgorithm::new(config);
    let result = bee.run(sphere);
    assert!(result.best_fitness < 1.0);
}

#[test]
fn test_bee_config_elite_less_than_scouts() {
    let config = bee::BeeConfig {
        num_scouts: 20, num_elite: 3, num_selected: 10,
        max_iterations: 20, bounds: vec![(-5.0, 5.0); 2], ..Default::default()
    };
    let bee = bee::BeeAlgorithm::new(config);
    let result = bee.run(sphere);
    assert!(result.best_fitness < f64::MAX);
}

// === Firefly Tests ===

#[test]
fn test_firefly_attractiveness() {
    let config = firefly::FireflyConfig::default();
    let fa = firefly::FireflyAlgorithm::new(config);
    assert_relative_eq!(fa.attractiveness(0.0), 1.0);
    assert!(fa.attractiveness(100.0) < 0.01);
}

#[test]
fn test_firefly_attractiveness_decreases_with_distance() {
    let config = firefly::FireflyConfig::default();
    let fa = firefly::FireflyAlgorithm::new(config);
    let a1 = fa.attractiveness(1.0);
    let a2 = fa.attractiveness(2.0);
    let a3 = fa.attractiveness(5.0);
    assert!(a1 > a2);
    assert!(a2 > a3);
}

#[test]
fn test_firefly_finds_minimum() {
    let config = firefly::FireflyConfig {
        num_fireflies: 20, max_iterations: 50,
        bounds: vec![(-5.0, 5.0); 2], ..Default::default()
    };
    let fa = firefly::FireflyAlgorithm::new(config);
    let result = fa.run(sphere);
    assert!(result.best_fitness < 1.0, "got {}", result.best_fitness);
}

#[test]
fn test_firefly_history_length() {
    let config = firefly::FireflyConfig {
        num_fireflies: 10, max_iterations: 30,
        bounds: vec![(-5.0, 5.0); 2], ..Default::default()
    };
    let fa = firefly::FireflyAlgorithm::new(config);
    let result = fa.run(sphere);
    assert_eq!(result.history.len(), 30);
}

#[test]
fn test_firefly_fireflies_within_bounds() {
    let bounds = vec![(-5.0, 5.0); 2];
    let config = firefly::FireflyConfig {
        num_fireflies: 15, max_iterations: 20,
        bounds: bounds.clone(), ..Default::default()
    };
    let fa = firefly::FireflyAlgorithm::new(config);
    let result = fa.run(sphere);
    for ff in &result.fireflies {
        for (val, (lo, hi)) in ff.position.iter().zip(bounds.iter()) {
            assert!(*val >= *lo && *val <= *hi);
        }
    }
}

// === Wolf Pack Tests ===

#[test]
fn test_wolf_pack_finds_minimum() {
    let config = wolf_pack::WolfPackConfig {
        num_wolves: 20, max_iterations: 50,
        bounds: vec![(-5.0, 5.0); 2], ..Default::default()
    };
    let wpa = wolf_pack::WolfPackAlgorithm::new(config);
    let result = wpa.run(sphere);
    assert!(result.best_fitness < 1.0, "got {}", result.best_fitness);
}

#[test]
fn test_wolf_pack_roles_assigned() {
    let config = wolf_pack::WolfPackConfig {
        num_wolves: 20, num_leaders: 3, num_scouts: 5,
        max_iterations: 5, bounds: vec![(-5.0, 5.0); 2], ..Default::default()
    };
    let wpa = wolf_pack::WolfPackAlgorithm::new(config);
    let _ = wpa.run(sphere);
}

#[test]
fn test_wolf_pack_history_length() {
    let config = wolf_pack::WolfPackConfig {
        num_wolves: 10, max_iterations: 30,
        bounds: vec![(-5.0, 5.0); 2], ..Default::default()
    };
    let wpa = wolf_pack::WolfPackAlgorithm::new(config);
    let result = wpa.run(sphere);
    assert_eq!(result.history.len(), 30);
}

// === Bacterial Foraging Tests ===

#[test]
fn test_bacterial_finds_minimum() {
    let config = bacterial::BacterialConfig {
        num_bacteria: 20, num_chemotactic: 20, num_reproduction: 2, num_elimination: 2,
        bounds: vec![(-5.0, 5.0); 2], ..Default::default()
    };
    let bf = bacterial::BacterialForaging::new(config);
    let result = bf.run(sphere);
    assert!(result.best_fitness < 5.0, "got {}", result.best_fitness);
}

#[test]
fn test_bacterial_chemotaxis_moves_bacteria() {
    let config = bacterial::BacterialConfig {
        num_bacteria: 10, num_chemotactic: 10, num_reproduction: 1, num_elimination: 1,
        step_size: 0.5, bounds: vec![(-5.0, 5.0); 2], ..Default::default()
    };
    let bf = bacterial::BacterialForaging::new(config);
    let result = bf.run(sphere);
    assert!(result.history.len() > 0);
}

#[test]
fn test_bacterial_history_length() {
    let config = bacterial::BacterialConfig {
        num_bacteria: 10, num_chemotactic: 5, num_reproduction: 3, num_elimination: 2,
        bounds: vec![(-5.0, 5.0); 2], ..Default::default()
    };
    let bf = bacterial::BacterialForaging::new(config);
    let result = bf.run(sphere);
    assert_eq!(result.history.len(), 2);
}

#[test]
fn test_bacterial_best_within_bounds() {
    let bounds = vec![(-5.0, 5.0); 2];
    let bf = bacterial::BacterialForaging::new(bacterial::BacterialConfig {
        num_bacteria: 10, num_chemotactic: 10, num_reproduction: 2, num_elimination: 2,
        bounds: bounds.clone(), ..Default::default()
    });
    let result = bf.run(sphere);
    for (val, (lo, hi)) in result.best_position.iter().zip(bounds.iter()) {
        assert!(*val >= *lo && *val <= *hi);
    }
}

// === SDS Tests ===

#[test]
fn test_sds_finds_minimum() {
    let config = sds::SdsConfig {
        num_agents: 30, max_iterations: 80, threshold: 1.0,
        bounds: vec![(-5.0, 5.0); 2],
    };
    let sds = sds::StochasticDiffusionSearch::new(config);
    let result = sds.run(sphere);
    assert!(result.best_fitness < 5.0, "got {}", result.best_fitness);
}

#[test]
fn test_sds_convergence() {
    let config = sds::SdsConfig {
        num_agents: 40, max_iterations: 100, threshold: 1.0,
        bounds: vec![(-5.0, 5.0); 2],
    };
    let sds = sds::StochasticDiffusionSearch::new(config);
    let result = sds.run(sphere);
    if result.history.len() > 1 {
        assert!(result.history.last().unwrap() <= &result.history[0]);
    }
}

#[test]
fn test_sds_cluster_counts() {
    let config = sds::SdsConfig {
        num_agents: 20, max_iterations: 30, threshold: 0.5,
        bounds: vec![(-5.0, 5.0); 2],
    };
    let sds = sds::StochasticDiffusionSearch::new(config);
    let result = sds.run(sphere);
    assert_eq!(result.cluster_counts.len(), 30);
    for &count in &result.cluster_counts { assert!(count <= 20); }
}

#[test]
fn test_sds_history_length() {
    let config = sds::SdsConfig {
        num_agents: 10, max_iterations: 50,
        bounds: vec![(-5.0, 5.0); 2], ..Default::default()
    };
    let sds = sds::StochasticDiffusionSearch::new(config);
    let result = sds.run(sphere);
    assert_eq!(result.history.len(), 50);
}

// === Flocking (Boids) Tests ===

#[test]
fn test_flocking_init() {
    let config = flocking::FlockingConfig { num_boids: 20, ..Default::default() };
    let flocking = flocking::Flocking::new(config);
    let boids = flocking.init_flock();
    assert_eq!(boids.len(), 20);
    for boid in &boids {
        assert_eq!(boid.position.len(), 2);
        assert_eq!(boid.velocity.len(), 2);
    }
}

#[test]
fn test_flocking_separation() {
    let config = flocking::FlockingConfig {
        num_boids: 2, separation_radius: 30.0, perception_radius: 50.0, ..Default::default()
    };
    let flocking = flocking::Flocking::new(config);
    let boids = vec![
        flocking::Boid::new(vec![100.0, 100.0], vec![1.0, 0.0]),
        flocking::Boid::new(vec![101.0, 100.0], vec![0.0, 1.0]),
    ];
    let (sep, _, _) = flocking.compute_flocking_forces(&boids, 0);
    assert!(sep[0] < 0.0, "sep[0]={}", sep[0]);
}

#[test]
fn test_flocking_alignment() {
    let config = flocking::FlockingConfig {
        num_boids: 3, perception_radius: 200.0, separation_radius: 1.0, ..Default::default()
    };
    let flocking = flocking::Flocking::new(config);
    let boids = vec![
        flocking::Boid::new(vec![100.0, 100.0], vec![3.0, 0.0]),
        flocking::Boid::new(vec![110.0, 100.0], vec![3.0, 0.0]),
        flocking::Boid::new(vec![120.0, 100.0], vec![3.0, 0.0]),
    ];
    let (_, ali, _) = flocking.compute_flocking_forces(&boids, 0);
    let ali_mag = (ali[0].powi(2) + ali[1].powi(2)).sqrt();
    assert!(ali_mag < 1.0, "ali_mag={}", ali_mag);
}

#[test]
fn test_flocking_cohesion() {
    let config = flocking::FlockingConfig {
        num_boids: 3, perception_radius: 500.0, separation_radius: 1.0, ..Default::default()
    };
    let flocking = flocking::Flocking::new(config);
    let boids = vec![
        flocking::Boid::new(vec![100.0, 100.0], vec![0.0, 0.0]),
        flocking::Boid::new(vec![200.0, 100.0], vec![0.0, 0.0]),
        flocking::Boid::new(vec![300.0, 100.0], vec![0.0, 0.0]),
    ];
    let (_, _, coh) = flocking.compute_flocking_forces(&boids, 0);
    assert!(coh[0] > 0.0, "coh[0]={}", coh[0]);
}

#[test]
fn test_flocking_step_updates_positions() {
    let config = flocking::FlockingConfig { num_boids: 5, ..Default::default() };
    let flocking = flocking::Flocking::new(config);
    let mut boids = flocking.init_flock();
    let old: Vec<Vec<f64>> = boids.iter().map(|b| b.position.clone()).collect();
    flocking.step(&mut boids);
    let moved = boids.iter().zip(old.iter()).filter(|(b, o)| b.position != **o).count();
    assert!(moved > 0);
}

#[test]
fn test_flocking_world_wrapping() {
    let config = flocking::FlockingConfig {
        num_boids: 1, world_size: (100.0, 100.0), ..Default::default()
    };
    let flocking = flocking::Flocking::new(config);
    let mut boids = vec![flocking::Boid::new(vec![99.0, 50.0], vec![5.0, 0.0])];
    flocking.step(&mut boids);
    assert!(boids[0].position[0] < 100.0 && boids[0].position[0] >= 0.0);
}

#[test]
fn test_flocking_no_self_interaction() {
    let config = flocking::FlockingConfig { num_boids: 1, ..Default::default() };
    let flocking = flocking::Flocking::new(config);
    let boids = vec![flocking::Boid::new(vec![50.0, 50.0], vec![1.0, 0.0])];
    let (sep, ali, coh) = flocking.compute_flocking_forces(&boids, 0);
    assert_eq!(sep[0], 0.0);
    assert_eq!(ali[0], 0.0);
    assert_eq!(coh[0], 0.0);
}

#[test]
fn test_flocking_speed_limited() {
    let config = flocking::FlockingConfig { num_boids: 3, max_speed: 4.0, ..Default::default() };
    let flocking = flocking::Flocking::new(config);
    let mut boids = vec![
        flocking::Boid::new(vec![50.0, 50.0], vec![10.0, 10.0]),
        flocking::Boid::new(vec![200.0, 200.0], vec![10.0, 10.0]),
        flocking::Boid::new(vec![300.0, 300.0], vec![10.0, 10.0]),
    ];
    flocking.step(&mut boids);
    for boid in &boids {
        let speed = (boid.velocity[0].powi(2) + boid.velocity[1].powi(2)).sqrt();
        assert!(speed <= 4.0 + 1e-10, "speed={}", speed);
    }
}

// === Agent Swarm Tests ===

#[test]
fn test_agent_swarm_init() {
    let config = agent_swarm::AgentSwarmConfig { num_agents: 15, ..Default::default() };
    let swarm = agent_swarm::AgentSwarm::new(config);
    assert_eq!(swarm.agents().len(), 15);
}

#[test]
fn test_agent_swarm_roles() {
    let config = agent_swarm::AgentSwarmConfig { num_agents: 9, ..Default::default() };
    let swarm = agent_swarm::AgentSwarm::new(config);
    let e = swarm.agents().iter().filter(|a| a.role == agent_swarm::AgentRole::Explorer).count();
    let w = swarm.agents().iter().filter(|a| a.role == agent_swarm::AgentRole::Worker).count();
    let c = swarm.agents().iter().filter(|a| a.role == agent_swarm::AgentRole::Coordinator).count();
    assert_eq!(e, 3);
    assert_eq!(w, 3);
    assert_eq!(c, 3);
}

#[test]
fn test_agent_swarm_step() {
    let config = agent_swarm::AgentSwarmConfig { num_agents: 10, ..Default::default() };
    let mut swarm = agent_swarm::AgentSwarm::new(config);
    let state = swarm.step();
    assert_eq!(state.agents.len(), 10);
    assert!(state.avg_cohesion >= 0.0);
}

#[test]
fn test_agent_swarm_multiple_steps() {
    let config = agent_swarm::AgentSwarmConfig { num_agents: 20, ..Default::default() };
    let mut swarm = agent_swarm::AgentSwarm::new(config);
    for _ in 0..10 {
        let state = swarm.step();
        assert!(state.avg_cohesion >= 0.0);
    }
}

// === Serialization Tests ===

#[test]
fn test_aco_config_serde_roundtrip() {
    let config = aco::AcoConfig::default();
    let json = serde_json::to_string(&config).unwrap();
    let d: aco::AcoConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(config.num_ants, d.num_ants);
    assert_relative_eq!(config.alpha, d.alpha);
}

#[test]
fn test_pso_config_serde_roundtrip() {
    let config = pso::PsoConfig::default();
    let json = serde_json::to_string(&config).unwrap();
    let d: pso::PsoConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(config.num_particles, d.num_particles);
    assert_relative_eq!(config.inertia, d.inertia);
}

#[test]
fn test_firefly_config_serde_roundtrip() {
    let config = firefly::FireflyConfig::default();
    let json = serde_json::to_string(&config).unwrap();
    let d: firefly::FireflyConfig = serde_json::from_str(&json).unwrap();
    assert_relative_eq!(config.beta0, d.beta0);
    assert_relative_eq!(config.gamma, d.gamma);
}

#[test]
fn test_flocking_config_serde_roundtrip() {
    let config = flocking::FlockingConfig::default();
    let json = serde_json::to_string(&config).unwrap();
    let d: flocking::FlockingConfig = serde_json::from_str(&json).unwrap();
    assert_relative_eq!(config.separation_weight, d.separation_weight);
}

// === Integration ===

#[test]
fn test_all_algorithms_on_sphere() {
    let bounds = vec![(-5.0, 5.0); 2];
    let pso = pso::ParticleSwarmOptimization::new(pso::PsoConfig {
        num_particles: 10, max_iterations: 20, bounds: bounds.clone(), ..Default::default()
    });
    assert!(pso.run(sphere).best_fitness < f64::MAX);

    let bee = bee::BeeAlgorithm::new(bee::BeeConfig {
        num_scouts: 10, max_iterations: 20, bounds: bounds.clone(), ..Default::default()
    });
    assert!(bee.run(sphere).best_fitness < f64::MAX);

    let fa = firefly::FireflyAlgorithm::new(firefly::FireflyConfig {
        num_fireflies: 10, max_iterations: 20, bounds: bounds.clone(), ..Default::default()
    });
    assert!(fa.run(sphere).best_fitness < f64::MAX);

    let wp = wolf_pack::WolfPackAlgorithm::new(wolf_pack::WolfPackConfig {
        num_wolves: 10, max_iterations: 20, bounds: bounds.clone(), ..Default::default()
    });
    assert!(wp.run(sphere).best_fitness < f64::MAX);

    let sds = sds::StochasticDiffusionSearch::new(sds::SdsConfig {
        num_agents: 10, max_iterations: 20, bounds, ..Default::default()
    });
    assert!(sds.run(sphere).best_fitness < f64::MAX);
}
