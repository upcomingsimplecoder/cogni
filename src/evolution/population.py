"""Population management: births, deaths, and demographics.

Orchestrates population dynamics:
- Monitors population size
- Triggers reproduction when appropriate
- Handles agent death
- Maintains population statistics
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class PopulationManager:
    """Manages population lifecycle and demographics.

    Coordinates reproduction, death, and population control
    to maintain a sustainable population.
    """

    def __init__(
        self,
        genetics,  # GeneticSystem
        reproduction,  # ReproductionSystem
        lineage_tracker=None,  # Optional[LineageTracker]
    ):
        """Initialize population manager.

        Args:
            genetics: GeneticSystem for trait inheritance
            reproduction: ReproductionSystem for reproduction logic
            lineage_tracker: Optional LineageTracker for recording births/deaths
        """
        self.genetics = genetics
        self.reproduction = reproduction
        self.lineage_tracker = lineage_tracker
        self._population_history: list[dict] = []

    def tick(
        self,
        engine,  # SimulationEngine (avoid circular import)
        current_tick: int,
    ) -> dict:
        """Process population dynamics for this tick.

        Handles:
        - Agent deaths
        - Reproduction attempts
        - Population statistics

        Args:
            engine: Simulation engine instance
            current_tick: Current tick number

        Returns:
            Dict with statistics: {"births": [...], "deaths": [...], "population": N}
        """
        from src.simulation.entities import Agent, AgentNeeds

        births = []
        deaths = []

        # Get current agents
        agents = engine.get_agents()

        # Process deaths
        for agent in list(agents):
            if not agent.needs.is_alive():
                agent.die()
                deaths.append(str(agent.agent_id))

                # Track death in lineage system
                if self.lineage_tracker:
                    # Calculate final fitness for lineage tracking
                    fitness = self.genetics.fitness(agent)
                    self.lineage_tracker.record_death(
                        agent_id=str(agent.agent_id),
                        death_tick=current_tick,
                        final_fitness=fitness,
                    )

        # Remove dead agents
        alive_agents = [a for a in agents if a.alive]
        current_population = len(alive_agents)

        # Attempt reproduction for eligible agents
        if current_population < self.reproduction.max_population:
            # Calculate fitness for all agents
            fitness_scores = {
                str(agent.agent_id): self.genetics.fitness(agent) for agent in alive_agents
            }

            for agent in alive_agents:
                fitness = fitness_scores.get(str(agent.agent_id), 0.0)

                # Check if can reproduce
                if not self.reproduction.can_reproduce(
                    agent, fitness, current_population, current_tick
                ):
                    continue

                # Probabilistic reproduction
                import random

                prob = self.reproduction.calculate_reproduction_probability(agent, fitness)
                if random.random() > prob:
                    continue

                # Attempt reproduction
                offspring_data = self.reproduction.reproduce(agent, self.genetics, current_tick)

                if offspring_data is None:
                    continue

                # Create offspring agent
                child_profile = offspring_data["child_profile"]
                child_x, child_y = offspring_data["position"]

                Agent(
                    x=child_x,
                    y=child_y,
                    needs=AgentNeeds(),
                    inventory={},
                    agent_id=child_profile.agent_id,
                    profile=child_profile,
                    alive=True,
                    ticks_alive=0,
                )

                # Add to engine (would need engine method)
                # For now, record birth data
                births.append(
                    {
                        "parent_id": offspring_data["parent_id"],
                        "child_id": offspring_data["child_id"],
                        "position": (child_x, child_y),
                        "tick": current_tick,
                    }
                )

                # Track birth in lineage system
                if self.lineage_tracker:
                    self.lineage_tracker.record_birth(
                        agent_id=offspring_data["child_id"],
                        parent_id=offspring_data["parent_id"],
                        traits=child_profile.traits,
                        birth_tick=current_tick,
                    )

                current_population += 1

                # Stop if we hit population cap
                if current_population >= self.reproduction.max_population:
                    break

        # Record statistics
        stats = {
            "births": births,
            "deaths": deaths,
            "population": current_population,
            "tick": current_tick,
        }

        self._population_history.append(stats)

        return stats

    def get_population_stats(self) -> dict:
        """Get aggregate population statistics.

        Returns:
            Dict with population metrics
        """
        if not self._population_history:
            return {
                "total_births": 0,
                "total_deaths": 0,
                "current_population": 0,
            }

        total_births = sum(len(h["births"]) for h in self._population_history)
        total_deaths = sum(len(h["deaths"]) for h in self._population_history)
        current_pop = self._population_history[-1]["population"]

        return {
            "total_births": total_births,
            "total_deaths": total_deaths,
            "current_population": current_pop,
            "history": self._population_history.copy(),
        }

    def get_population_history(self) -> list[dict]:
        """Get full population history.

        Returns:
            List of population statistics per tick
        """
        return self._population_history.copy()
