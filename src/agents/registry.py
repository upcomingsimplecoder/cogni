"""Agent registry: manages agent lifecycle (spawn, death, lookup)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.agents.identity import AgentID, AgentProfile
from src.simulation.entities import Agent

if TYPE_CHECKING:
    pass


class AgentRegistry:
    """Manages agent lifecycle: spawn, death, lookup.

    Single source of truth for all agents in the simulation.
    """

    def __init__(self, engine: Any):
        self._engine = engine
        self._agents: dict[AgentID, Agent] = {}
        self._dead: dict[AgentID, Agent] = {}
        self._awareness_loops: dict[AgentID, Any] = {}  # AwarenessLoop
        self._memories: dict[AgentID, tuple[Any, Any]] = {}  # (EpisodicMemory, SocialMemory)
        self._death_causes: dict[AgentID, str] = {}

    def spawn(
        self,
        profile: AgentProfile,
        x: int,
        y: int,
        color: str = "white",
    ) -> Agent:
        """Spawn a new agent at the given position.

        Validates spawn position is not water and not out of bounds.
        """
        from src.simulation.world import TileType

        tile = self._engine.world.get_tile(x, y)
        if tile is None:
            raise ValueError(f"Cannot spawn agent at ({x}, {y}): out of bounds")
        if tile.type == TileType.WATER:
            raise ValueError(f"Cannot spawn agent at ({x}, {y}): water tile")

        agent = Agent(
            x=x,
            y=y,
            agent_id=profile.agent_id,
            profile=profile,
            color=color,
        )
        self._agents[profile.agent_id] = agent
        self._engine.world.place_agent(profile.agent_id, (x, y))
        return agent

    def kill(self, agent_id: AgentID, cause: str = "unknown") -> None:
        """Remove agent from living registry, move to dead."""
        if agent_id in self._agents:
            agent = self._agents.pop(agent_id)
            agent.die()
            self._dead[agent_id] = agent
            self._death_causes[agent_id] = cause
            self._engine.world.remove_agent(agent_id, (agent.x, agent.y))

    def get(self, agent_id: AgentID | object) -> Agent | None:
        """Look up a living agent by ID."""
        from src.agents.identity import AgentID as AgentIDClass

        if isinstance(agent_id, AgentIDClass):
            return self._agents.get(agent_id)
        return None

    def living_agents(self) -> list[Agent]:
        """All living agents, in spawn order."""
        return list(self._agents.values())

    def all_agents(self) -> list[Agent]:
        """All agents including dead (for history/metrics)."""
        return list(self._agents.values()) + list(self._dead.values())

    def register_awareness_loop(self, agent_id: AgentID, loop: Any) -> None:
        """Register an awareness loop for an agent."""
        self._awareness_loops[agent_id] = loop

    def get_awareness_loop(self, agent_id: AgentID | object) -> Any | None:
        """Get the awareness loop for an agent."""
        from src.agents.identity import AgentID as AgentIDClass

        if isinstance(agent_id, AgentIDClass):
            return self._awareness_loops.get(agent_id)
        return None

    def register_memory(self, agent_id: AgentID, episodic: Any, social: Any) -> None:
        """Register memory systems for an agent."""
        self._memories[agent_id] = (episodic, social)

    def get_memory(self, agent_id: AgentID | object) -> tuple[Any, Any] | None:
        """Get (episodic, social) memory for an agent. Returns None if not registered."""
        from src.agents.identity import AgentID as AgentIDClass

        if isinstance(agent_id, AgentIDClass):
            return self._memories.get(agent_id)
        return None

    def death_cause(self, agent_id: AgentID) -> str:
        """Get the cause of death for a dead agent."""
        return self._death_causes.get(agent_id, "unknown")

    @property
    def count_living(self) -> int:
        return len(self._agents)

    @property
    def count_dead(self) -> int:
        return len(self._dead)
