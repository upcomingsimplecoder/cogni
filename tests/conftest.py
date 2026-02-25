"""Shared test fixtures for the cogniarch test suite."""

from __future__ import annotations

import pytest

from src.agents.identity import AgentID, AgentProfile, PersonalityTraits
from src.agents.registry import AgentRegistry
from src.communication.channel import MessageBus
from src.config import SimulationConfig
from src.memory.episodic import EpisodicMemory
from src.memory.social import SocialMemory
from src.simulation.engine import SimulationEngine
from src.simulation.entities import Agent
from src.simulation.world import TileType, World


@pytest.fixture
def config() -> SimulationConfig:
    """Default config with small world for fast tests."""
    return SimulationConfig(
        world_width=16,
        world_height=16,
        seed=42,
        max_ticks=100,
        num_agents=3,
        agent_archetypes=["gatherer", "explorer", "diplomat"],
    )


@pytest.fixture
def world(config: SimulationConfig) -> World:
    """16x16 world with seed 42."""
    return World(config.world_width, config.world_height, config.seed)


@pytest.fixture
def traits() -> PersonalityTraits:
    """Default (balanced) personality traits."""
    return PersonalityTraits()


@pytest.fixture
def agent_id() -> AgentID:
    """A deterministic agent ID for testing."""
    return AgentID("test0001")


@pytest.fixture
def profile(agent_id: AgentID, traits: PersonalityTraits) -> AgentProfile:
    """A test agent profile."""
    return AgentProfile(
        agent_id=agent_id,
        name="test_agent",
        archetype="survivalist",
        traits=traits,
    )


@pytest.fixture
def agent(profile: AgentProfile) -> Agent:
    """A test agent at (8,8) with default needs."""
    return Agent(
        x=8,
        y=8,
        agent_id=profile.agent_id,
        profile=profile,
    )


@pytest.fixture
def engine(config: SimulationConfig) -> SimulationEngine:
    """A fresh simulation engine with small world."""
    return SimulationEngine(config)


@pytest.fixture
def registry(engine: SimulationEngine) -> AgentRegistry:
    """The engine's agent registry."""
    return engine.registry


@pytest.fixture
def message_bus() -> MessageBus:
    """A fresh message bus."""
    return MessageBus()


@pytest.fixture
def episodic_memory() -> EpisodicMemory:
    """A fresh episodic memory."""
    return EpisodicMemory()


@pytest.fixture
def social_memory() -> SocialMemory:
    """A fresh social memory."""
    return SocialMemory()


def find_non_water_pos(world: World) -> tuple[int, int]:
    """Find any non-water tile position in the world."""
    for y in range(world.height):
        for x in range(world.width):
            tile = world.get_tile(x, y)
            if tile is not None and tile.type != TileType.WATER:
                return (x, y)
    raise RuntimeError("No non-water tile in test world")


def spawn_test_agent(
    engine: SimulationEngine,
    name: str = "test",
    archetype: str = "survivalist",
    x: int | None = None,
    y: int | None = None,
    traits: PersonalityTraits | None = None,
) -> Agent:
    """Spawn a test agent on the engine with full awareness loop + memory."""
    from src.agents.archetypes import find_valid_spawn, spawn_archetype

    if x is None or y is None:
        x, y = find_valid_spawn(engine.world)

    return spawn_archetype(
        archetype_name=archetype,
        agent_name=name,
        engine=engine,
        x=x,
        y=y,
    )
