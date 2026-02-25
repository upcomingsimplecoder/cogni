"""Agent archetypes: predefined personality profiles for spawning."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from src.agents.identity import AgentID, AgentProfile, PersonalityTraits

if TYPE_CHECKING:
    pass


ARCHETYPES: dict[str, dict[str, Any]] = {
    "gatherer": {
        "description": "Focuses on resource collection and hoarding",
        "traits": PersonalityTraits(
            cooperation_tendency=0.3,
            curiosity=0.2,
            risk_tolerance=0.3,
            resource_sharing=0.2,
            aggression=0.1,
            sociability=0.3,
        ),
        "color": "green",
        "symbol": "G",
    },
    "explorer": {
        "description": "Roams widely, discovers resources, shares information",
        "traits": PersonalityTraits(
            cooperation_tendency=0.5,
            curiosity=0.9,
            risk_tolerance=0.7,
            resource_sharing=0.4,
            aggression=0.2,
            sociability=0.5,
        ),
        "color": "cyan",
        "symbol": "E",
    },
    "diplomat": {
        "description": "Seeks out other agents, negotiates, builds relationships",
        "traits": PersonalityTraits(
            cooperation_tendency=0.9,
            curiosity=0.4,
            risk_tolerance=0.4,
            resource_sharing=0.8,
            aggression=0.05,
            sociability=0.9,
        ),
        "color": "yellow",
        "symbol": "D",
    },
    "aggressor": {
        "description": "Territorial, takes resources from others, low cooperation",
        "traits": PersonalityTraits(
            cooperation_tendency=0.1,
            curiosity=0.3,
            risk_tolerance=0.8,
            resource_sharing=0.05,
            aggression=0.9,
            sociability=0.4,
        ),
        "color": "red",
        "symbol": "A",
    },
    "survivalist": {
        "description": "Balanced, cautious, self-sufficient",
        "traits": PersonalityTraits(
            cooperation_tendency=0.5,
            curiosity=0.5,
            risk_tolerance=0.5,
            resource_sharing=0.3,
            aggression=0.3,
            sociability=0.4,
        ),
        "color": "white",
        "symbol": "S",
    },
}


def find_valid_spawn(world: Any, rng: random.Random | None = None) -> tuple[int, int]:
    """Find a random valid (non-water) spawn position."""
    from src.simulation.world import TileType

    _rng = rng or random.Random()
    attempts = 0
    while attempts < 1000:
        x = _rng.randint(5, world.width - 6)
        y = _rng.randint(5, world.height - 6)
        tile = world.get_tile(x, y)
        if tile is not None and tile.type != TileType.WATER:
            return (x, y)
        attempts += 1
    # Fallback: scan for any non-water tile
    for y in range(world.height):
        for x in range(world.width):
            tile = world.get_tile(x, y)
            if tile is not None and tile.type != TileType.WATER:
                return (x, y)
    raise RuntimeError("No valid spawn position found")


def spawn_archetype(
    archetype_name: str,
    agent_name: str,
    engine: Any,
    x: int | None = None,
    y: int | None = None,
    strategy_override: Any | None = None,
    architecture: str | None = None,
) -> Any:
    """Spawn an agent with the given archetype.

    Creates Agent, sets up awareness loop with PersonalityStrategy (or override),
    registers episodic and social memory.

    Args:
        archetype_name: Name of the archetype (e.g., "gatherer", "explorer")
        agent_name: Display name for the agent
        engine: The simulation engine
        x: X position (auto-finds if None)
        y: Y position (auto-finds if None)
        strategy_override: Optional DecisionStrategy (e.g. LLMStrategy).
                          Defaults to PersonalityStrategy if None.
        architecture: Optional cognitive architecture name (e.g., "reactive", "cautious").
                     If provided, uses build_awareness_loop() instead of manual construction.
                     Defaults to None for backward compatibility.

    Returns the spawned Agent.
    """
    from src.awareness.loop import AwarenessLoop
    from src.awareness.reflection import ReflectionModule
    from src.awareness.sensation import SensationModule
    from src.cognition.strategies.personality import PersonalityStrategy
    from src.memory.episodic import EpisodicMemory
    from src.memory.social import SocialMemory

    arch = ARCHETYPES[archetype_name]

    profile = AgentProfile(
        agent_id=AgentID(),
        name=agent_name,
        archetype=archetype_name,
        traits=arch["traits"].copy(),  # Deep copy — don't share traits between agents
    )

    # Find spawn position if not specified
    if x is None or y is None:
        x, y = find_valid_spawn(engine.world)

    # Spawn via registry
    agent = engine.registry.spawn(profile, x, y, color=arch["color"])

    # Give starting inventory
    agent.add_item("water", 2)
    agent.add_item("berries", 1)

    # Create awareness loop
    if architecture is not None:
        # Use architecture-based construction
        from src.awareness.architecture import build_awareness_loop

        awareness = build_awareness_loop(agent, architecture, strategy_override)
    else:
        # Backward compatible: manual construction
        strategy = strategy_override if strategy_override is not None else PersonalityStrategy()
        awareness = AwarenessLoop(
            agent=agent,
            strategy=strategy,
            sensation_module=SensationModule(),
            reflection_module=ReflectionModule(),
        )
    engine.registry.register_awareness_loop(profile.agent_id, awareness)

    # Create memory systems
    episodic = EpisodicMemory()
    social = SocialMemory()
    engine.registry.register_memory(profile.agent_id, episodic, social)

    return agent
