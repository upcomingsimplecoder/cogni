"""Genetic system: trait inheritance, mutation, and fitness.

Implements basic genetic algorithms for agent trait evolution:
- Trait inheritance with Gaussian mutation
- Crossover between two parents
- Fitness scoring based on survival metrics
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.identity import PersonalityTraits
    from src.simulation.entities import Agent


class GeneticSystem:
    """Handles genetic operations for agent evolution.

    Uses simple genetic algorithm principles:
    - Traits inherit from parent(s) with mutation
    - Mutation uses Gaussian noise
    - Fitness based on survival duration and resource accumulation
    """

    def __init__(self, mutation_rate: float = 0.05, mutation_magnitude: float = 0.1):
        """Initialize genetic system.

        Args:
            mutation_rate: Probability of mutation per trait (0.0-1.0)
            mutation_magnitude: Standard deviation of mutation noise
        """
        self.mutation_rate = mutation_rate
        self.mutation_magnitude = mutation_magnitude

    def inherit(
        self,
        parent_traits: PersonalityTraits,
    ) -> PersonalityTraits:
        """Create child traits from a single parent with mutation.

        Args:
            parent_traits: Parent's personality traits

        Returns:
            New PersonalityTraits for the child
        """
        from src.agents.identity import PersonalityTraits

        child_dict = {}

        for trait_name, value in parent_traits.as_dict().items():
            # Apply mutation with probability
            if random.random() < self.mutation_rate:
                mutation = random.gauss(0, self.mutation_magnitude)
                new_value = value + mutation
            else:
                new_value = value

            # Clamp to [0.0, 1.0]
            child_dict[trait_name] = max(0.0, min(1.0, new_value))

        return PersonalityTraits(**child_dict)

    def crossover(
        self,
        parent_a_traits: PersonalityTraits,
        parent_b_traits: PersonalityTraits,
    ) -> PersonalityTraits:
        """Create child traits from two parents via crossover.

        Uses uniform crossover: each trait randomly chosen from either parent,
        then mutated.

        Args:
            parent_a_traits: First parent's traits
            parent_b_traits: Second parent's traits

        Returns:
            New PersonalityTraits for the child
        """
        from src.agents.identity import PersonalityTraits

        child_dict = {}
        traits_a = parent_a_traits.as_dict()
        traits_b = parent_b_traits.as_dict()

        for trait_name in traits_a:
            # Uniform crossover: pick from either parent
            value = traits_a[trait_name] if random.random() < 0.5 else traits_b[trait_name]

            # Apply mutation with probability
            if random.random() < self.mutation_rate:
                mutation = random.gauss(0, self.mutation_magnitude)
                value = value + mutation

            # Clamp to [0.0, 1.0]
            child_dict[trait_name] = max(0.0, min(1.0, value))

        return PersonalityTraits(**child_dict)

    def fitness(self, agent: Agent) -> float:
        """Calculate fitness score for an agent.

        Fitness is based on:
        - Survival duration (ticks_alive)
        - Resource accumulation (inventory value)
        - Current health

        Args:
            agent: Agent to evaluate

        Returns:
            Fitness score (0.0-1.0+, higher is better)
        """
        if not agent.alive:
            # Dead agents have fitness based only on how long they survived
            return min(1.0, agent.ticks_alive / 1000.0)

        # Base fitness from survival duration
        survival_fitness = min(0.5, agent.ticks_alive / 1000.0)

        # Resource fitness (inventory value)
        inventory_value = sum(agent.inventory.values())
        resource_fitness = min(0.3, inventory_value / 50.0)

        # Health fitness
        health_fitness = agent.needs.health / 500.0  # Scale to max 0.2

        total_fitness = survival_fitness + resource_fitness + health_fitness

        return max(0.0, total_fitness)

    def select_parent(
        self,
        population: list[Agent],
        fitness_scores: dict[str, float],
    ) -> Agent:
        """Select a parent using fitness-proportionate selection.

        Args:
            population: List of candidate agents
            fitness_scores: Map from agent_id to fitness score

        Returns:
            Selected parent agent
        """
        if not population:
            raise ValueError("Cannot select parent from empty population")

        # Build fitness weights
        weights = [fitness_scores.get(str(agent.agent_id), 0.01) for agent in population]

        # Fitness-proportionate selection
        return random.choices(population, weights=weights, k=1)[0]
