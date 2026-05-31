//! Agent Swarm Coordination — emergent collective behavior in agent fleets.
//!
//! Combines flocking (Boids) with task-based coordination for agent swarms.

use serde::{Deserialize, Serialize};
use crate::flocking::{Boid, Flocking, FlockingConfig};

/// Role of an agent in the swarm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentRole {
    Explorer,
    Worker,
    Coordinator,
}

/// An agent in the swarm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmAgent {
    pub id: usize,
    pub boid: Boid,
    pub role: AgentRole,
    pub task_score: f64,
}

/// Configuration for the agent swarm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSwarmConfig {
    pub num_agents: usize,
    pub flocking_config: FlockingConfig,
}

impl Default for AgentSwarmConfig {
    fn default() -> Self {
        Self {
            num_agents: 20,
            flocking_config: FlockingConfig::default(),
        }
    }
}

/// Result of swarm step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmState {
    pub agents: Vec<SwarmAgent>,
    pub avg_cohesion: f64,
    pub avg_alignment: f64,
}

/// Agent Swarm with emergent coordination.
pub struct AgentSwarm {
    config: AgentSwarmConfig,
    flocking: Flocking,
    agents: Vec<SwarmAgent>,
}

impl AgentSwarm {
    pub fn new(config: AgentSwarmConfig) -> Self {
        let flocking = Flocking::new(config.flocking_config.clone());
        let mut rng = rand::thread_rng();

        let mut boids = Vec::new();
        let mut agents = Vec::new();

        for i in 0..config.num_agents {
            let pos = vec![
                rand::Rng::gen_range(&mut rng, 0.0..config.flocking_config.world_size.0),
                rand::Rng::gen_range(&mut rng, 0.0..config.flocking_config.world_size.1),
            ];
            let vel = vec![
                rand::Rng::gen_range(&mut rng, -1.0..1.0),
                rand::Rng::gen_range(&mut rng, -1.0..1.0),
            ];
            let boid = Boid::new(pos, vel);
            let role = match i % 3 {
                0 => AgentRole::Explorer,
                1 => AgentRole::Worker,
                _ => AgentRole::Coordinator,
            };
            agents.push(SwarmAgent {
                id: i,
                boid: boid.clone(),
                role,
                task_score: 0.0,
            });
            boids.push(boid);
        }

        Self {
            config,
            flocking,
            agents,
        }
    }

    /// Step the swarm forward, applying flocking + role-based behavior.
    pub fn step(&mut self) -> SwarmState {
        // Extract boids for flocking update
        let mut boids: Vec<Boid> = self.agents.iter().map(|a| a.boid.clone()).collect();
        self.flocking.step(&mut boids);

        // Update agents
        for (i, agent) in self.agents.iter_mut().enumerate() {
            agent.boid = boids[i].clone();

            // Role-based modifications
            match agent.role {
                AgentRole::Explorer => {
                    // Explorers move faster
                    agent.boid.velocity[0] *= 1.1;
                    agent.boid.velocity[1] *= 1.1;
                }
                AgentRole::Worker => {
                    // Workers are more cohesive
                    // (handled by flocking weights)
                }
                AgentRole::Coordinator => {
                    // Coordinators slow down to stay near center
                    agent.boid.velocity[0] *= 0.9;
                    agent.boid.velocity[1] *= 0.9;
                }
            }

            // Clamp speed
            for d in 0..2 {
                agent.boid.velocity[d] = agent.boid.velocity[d]
                    .clamp(-self.config.flocking_config.max_speed, self.config.flocking_config.max_speed);
            }
        }

        // Compute metrics
        let avg_cohesion = self.compute_avg_cohesion();
        let avg_alignment = self.compute_avg_alignment();

        SwarmState {
            agents: self.agents.clone(),
            avg_cohesion,
            avg_alignment,
        }
    }

    fn compute_avg_cohesion(&self) -> f64 {
        let n = self.agents.len() as f64;
        if n == 0.0 {
            return 0.0;
        }
        let cx: f64 = self.agents.iter().map(|a| a.boid.position[0]).sum::<f64>() / n;
        let cy: f64 = self.agents.iter().map(|a| a.boid.position[1]).sum::<f64>() / n;
        self.agents
            .iter()
            .map(|a| {
                ((a.boid.position[0] - cx).powi(2) + (a.boid.position[1] - cy).powi(2)).sqrt()
            })
            .sum::<f64>()
            / n
    }

    fn compute_avg_alignment(&self) -> f64 {
        let n = self.agents.len() as f64;
        if n == 0.0 {
            return 0.0;
        }
        let avg_vx: f64 = self.agents.iter().map(|a| a.boid.velocity[0]).sum::<f64>() / n;
        let avg_vy: f64 = self.agents.iter().map(|a| a.boid.velocity[1]).sum::<f64>() / n;
        let avg_mag = (avg_vx.powi(2) + avg_vy.powi(2)).sqrt();
        // Alignment: average dot product of individual velocities with mean velocity
        if avg_mag < 1e-10 {
            return 0.0;
        }
        self.agents
            .iter()
            .map(|a| {
                (a.boid.velocity[0] * avg_vx + a.boid.velocity[1] * avg_vy)
                    / ((a.boid.velocity[0].powi(2) + a.boid.velocity[1].powi(2)).sqrt().max(1e-10) * avg_mag)
            })
            .sum::<f64>()
            / n
    }

    pub fn agents(&self) -> &[SwarmAgent] {
        &self.agents
    }

    pub fn config(&self) -> &AgentSwarmConfig {
        &self.config
    }
}
