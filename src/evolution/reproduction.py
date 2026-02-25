"""Reproduction system: conditions, costs, and offspring creation.

Handles agent reproduction including:
- Reproduction eligibility checks
- Resource costs
- Offspring creation with genetic inheritance
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.simulation.entities import Agent


class ReproductionSystem:
    """Manages agent reproduction.

    Reproduction requires:
    - Minimum age (ticks alive)
    - Minimum fitness threshold
    - Population below cap
    - Sufficient energy and hunger
    """

    def __init__(
        self,
        min_age: int = 100,
        fitness_threshold: float = 0.6,
        max_population: int = 15,
        energy_cost: float = 20.0,
        hunger_cost: float = 10.0,
    ):
        """Initialize reproduction system.

        Args:
            min_age: Minimum ticks_alive before reproduction
            fitness_threshold: Minimum fitness to reproduce
            max_population: Maximum population size
            energy_cost: Energy cost of reproduction
            hunger_cost: Hunger cost of reproduction
        """
        self.min_age = min_age
        self.fitness_threshold = fitness_threshold
        self.max_population = max_population
        self.energy_cost = energy_cost
        self.hunger_cost = hunger_cost

    def can_reproduce(
        self,
        agent: Agent,
        fitness: float,
        population_count: int,
        tick: int,
    ) -> bool:
        """Check if an agent can reproduce.

        Args:
            agent: Agent attempting to reproduce
            fitness: Agent's fitness score
            population_count: Current population size
            tick: Current simulation tick

        Returns:
            True if agent meets all reproduction requirements
        """
        if not agent.alive:
            return False

        # Check age
        if agent.ticks_alive < self.min_age:
            return False

        # Check fitness
        if fitness < self.fitness_threshold:
            return False

        # Check population cap
        if population_count >= self.max_population:
            return False

        # Check resource availability
        if agent.needs.energy < self.energy_cost:
            return False

        return agent.needs.hunger >= self.hunger_cost

    def reproduce(
        self,
        parent: Agent,
        genetics,  # GeneticSystem
        tick: int,
    ) -> dict | None:
        """Create offspring from a parent agent.

        Args:
            parent: Parent agent
            genetics: GeneticSystem instance for trait inheritance
            tick: Current simulation tick

        Returns:
            Dict with offspring data, or None if reproduction fails
        """
        if parent.profile is None or parent.profile.traits is None:
            return None

        # Apply reproduction costs
        parent.needs.energy = max(0.0, parent.needs.energy - self.energy_cost)
        parent.needs.hunger = max(0.0, parent.needs.hunger - self.hunger_cost)

        # Create child traits via inheritance
        child_traits = genetics.inherit(parent.profile.traits)

        # Generate child data
        import random

        from src.agents.identity import AgentID, AgentProfile

        child_id = AgentID()
        child_name = f"{parent.profile.name}-{child_id.value[:4]}"
        child_profile = AgentProfile(
            agent_id=child_id,
            name=child_name,
            archetype=parent.profile.archetype,
            traits=child_traits,
        )

        # Position near parent (with random offset)
        offset_x = random.randint(-2, 2)
        offset_y = random.randint(-2, 2)
        child_x = max(0, min(63, parent.x + offset_x))  # Clamp to world bounds
        child_y = max(0, min(63, parent.y + offset_y))

        return {
            "parent_id": str(parent.agent_id),
            "child_id": str(child_id),
            "child_profile": child_profile,
            "position": (child_x, child_y),
            "birth_tick": tick,
        }

    def calculate_reproduction_probability(
        self,
        agent: Agent,
        fitness: float,
    ) -> float:
        """Calculate probability of reproduction attempt this tick.

        Higher fitness = higher probability.

        Args:
            agent: Agent to evaluate
            fitness: Agent's fitness score

        Returns:
            Probability (0.0-1.0) of reproduction attempt
        """
        if agent.ticks_alive < self.min_age:
            return 0.0

        # Base probability increases with fitness above threshold
        if fitness < self.fitness_threshold:
            return 0.0

        # Scale probability with fitness
        excess_fitness = fitness - self.fitness_threshold
        return min(0.05, excess_fitness * 0.1)  # Max 5% per tick
