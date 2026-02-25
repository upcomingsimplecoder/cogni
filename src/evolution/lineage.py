"""Lineage tracking: family trees and trait drift.

Tracks agent ancestry, reproduction patterns, and trait evolution
across generations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.identity import PersonalityTraits


@dataclass
class LineageNode:
    """A node in the family tree.

    Represents one agent and their place in the lineage.
    """

    agent_id: str
    parent_id: str | None
    birth_tick: int
    death_tick: int | None = None
    traits: dict[str, float] = field(default_factory=dict)
    children: list[str] = field(default_factory=list)
    fitness: float = 0.0

    def add_child(self, child_id: str) -> None:
        """Record a child agent."""
        self.children.append(child_id)

    def is_alive(self, current_tick: int) -> bool:
        """Check if agent is still alive."""
        return self.death_tick is None or self.death_tick > current_tick


class LineageTracker:
    """Tracks family lineages and trait evolution.

    Maintains a family tree of all agents and analyzes
    trait drift, successful lineages, and reproduction patterns.
    """

    def __init__(self):
        self._lineages: dict[str, LineageNode] = {}
        self._roots: list[str] = []  # Original agents (no parents)

    def record_birth(
        self,
        agent_id: str,
        parent_id: str | None,
        traits: PersonalityTraits,
        birth_tick: int,
    ) -> None:
        """Record an agent's birth.

        Args:
            agent_id: New agent's ID
            parent_id: Parent's ID (None for original agents)
            traits: Agent's traits
            birth_tick: Tick when born
        """
        node = LineageNode(
            agent_id=agent_id,
            parent_id=parent_id,
            birth_tick=birth_tick,
            traits=traits.as_dict(),
        )

        self._lineages[agent_id] = node

        # Update parent's children
        if parent_id and parent_id in self._lineages:
            self._lineages[parent_id].add_child(agent_id)
        elif parent_id is None:
            # Root agent
            self._roots.append(agent_id)

    def record_death(
        self,
        agent_id: str,
        death_tick: int,
        final_fitness: float,
    ) -> None:
        """Record an agent's death.

        Args:
            agent_id: Agent's ID
            death_tick: Tick when died
            final_fitness: Agent's final fitness score
        """
        if agent_id in self._lineages:
            self._lineages[agent_id].death_tick = death_tick
            self._lineages[agent_id].fitness = final_fitness

    def get_lineage(self, agent_id: str) -> list[LineageNode]:
        """Get an agent's complete lineage (ancestors).

        Args:
            agent_id: Agent to trace

        Returns:
            List of LineageNode from root ancestor to agent
        """
        lineage = []
        current_id: str | None = agent_id

        while current_id and current_id in self._lineages:
            node = self._lineages[current_id]
            lineage.append(node)
            current_id = node.parent_id

        # Reverse to get root -> agent order
        lineage.reverse()
        return lineage

    def get_descendants(self, agent_id: str) -> list[str]:
        """Get all descendants of an agent.

        Args:
            agent_id: Agent to trace

        Returns:
            List of descendant agent IDs
        """
        if agent_id not in self._lineages:
            return []

        descendants = []
        queue = [agent_id]

        while queue:
            current_id = queue.pop(0)
            node = self._lineages.get(current_id)

            if node is None:
                continue

            for child_id in node.children:
                descendants.append(child_id)
                queue.append(child_id)

        return descendants

    def trait_drift(
        self,
        agent_id: str,
        trait_name: str,
    ) -> list[tuple[int, float]]:
        """Track how a trait changed across a lineage.

        Args:
            agent_id: Agent to trace
            trait_name: Name of trait to track

        Returns:
            List of (tick, trait_value) tuples from root to agent
        """
        lineage = self.get_lineage(agent_id)

        drift = []
        for node in lineage:
            trait_value = node.traits.get(trait_name, 0.5)
            drift.append((node.birth_tick, trait_value))

        return drift

    def most_successful_lineage(self) -> list[LineageNode]:
        """Find the most successful lineage by total descendant count.

        Returns:
            Lineage path of the most successful root agent
        """
        if not self._roots:
            return []

        best_root = None
        best_count = 0

        for root_id in self._roots:
            descendants = self.get_descendants(root_id)
            if len(descendants) > best_count:
                best_count = len(descendants)
                best_root = root_id

        if best_root is None:
            return []

        # Find the longest-lived descendant
        descendants = self.get_descendants(best_root)
        longest_lived = best_root

        max_lifespan = 0
        for desc_id in descendants:
            node = self._lineages.get(desc_id)
            if node and node.death_tick:
                lifespan = node.death_tick - node.birth_tick
                if lifespan > max_lifespan:
                    max_lifespan = lifespan
                    longest_lived = desc_id

        return self.get_lineage(longest_lived)

    def get_generation_stats(self, tick: int) -> dict:
        """Get statistics about current generation.

        Args:
            tick: Current tick

        Returns:
            Dict with generation metrics
        """
        alive_agents = [node for node in self._lineages.values() if node.is_alive(tick)]

        if not alive_agents:
            return {
                "population": 0,
                "avg_fitness": 0.0,
                "trait_diversity": {},
            }

        # Calculate average trait values
        trait_sums: dict[str, float] = {}
        trait_counts: dict[str, int] = {}

        for node in alive_agents:
            for trait_name, value in node.traits.items():
                trait_sums[trait_name] = trait_sums.get(trait_name, 0.0) + value
                trait_counts[trait_name] = trait_counts.get(trait_name, 0) + 1

        avg_traits = {name: trait_sums[name] / trait_counts[name] for name in trait_sums}

        # Calculate trait diversity (variance)
        trait_variance = {}
        for trait_name in trait_sums:
            mean = avg_traits[trait_name]
            variance = sum(
                (node.traits.get(trait_name, 0.5) - mean) ** 2 for node in alive_agents
            ) / len(alive_agents)
            trait_variance[trait_name] = variance

        return {
            "population": len(alive_agents),
            "avg_traits": avg_traits,
            "trait_diversity": trait_variance,
            "total_lineages": len(self._roots),
        }

    def export_tree(self) -> dict:
        """Export the complete family tree.

        Returns:
            Dict representation of the lineage tree
        """
        return {
            "roots": self._roots,
            "nodes": {
                agent_id: {
                    "parent": node.parent_id,
                    "children": node.children,
                    "birth_tick": node.birth_tick,
                    "death_tick": node.death_tick,
                    "traits": node.traits,
                    "fitness": node.fitness,
                }
                for agent_id, node in self._lineages.items()
            },
        }

    def import_tree(self, data: dict) -> None:
        """Import a complete family tree.

        Args:
            data: Dict representation of lineage tree (from export_tree)
        """
        # Clear existing state
        self._lineages.clear()
        self._roots.clear()

        # Handle missing or empty data
        if not data:
            return

        # Set roots
        self._roots = list(data.get("roots", []))

        # Reconstruct nodes
        nodes_data = data.get("nodes", {})
        for agent_id, node_data in nodes_data.items():
            node = LineageNode(
                agent_id=agent_id,
                parent_id=node_data.get("parent"),
                birth_tick=node_data.get("birth_tick", 0),
                death_tick=node_data.get("death_tick"),
                traits=node_data.get("traits", {}),
                children=list(node_data.get("children", [])),
                fitness=node_data.get("fitness", 0.0),
            )
            self._lineages[agent_id] = node
