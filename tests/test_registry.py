"""Tests for AgentRegistry: agent lifecycle management."""

from __future__ import annotations

import pytest

from src.agents.identity import AgentID, AgentProfile, PersonalityTraits
from src.simulation.engine import SimulationEngine
from src.simulation.world import TileType


def find_non_water(world) -> tuple[int, int]:
    """Find a non-water tile position."""
    for y in range(world.height):
        for x in range(world.width):
            tile = world.get_tile(x, y)
            if tile and tile.type != TileType.WATER:
                return x, y
    raise RuntimeError("No non-water tile found in test world")


def find_water(world) -> tuple[int, int]:
    """Find a water tile position."""
    for y in range(world.height):
        for x in range(world.width):
            tile = world.get_tile(x, y)
            if tile and tile.type == TileType.WATER:
                return x, y
    raise RuntimeError("No water tile found in test world")


def test_spawn_creates_agent_at_valid_position(engine: SimulationEngine):
    """spawn() creates agent at valid position."""
    x, y = find_non_water(engine.world)
    profile = AgentProfile(
        agent_id=AgentID("test001"),
        name="TestAgent",
        archetype="survivalist",
        traits=PersonalityTraits(),
    )

    agent = engine.registry.spawn(profile, x, y)

    assert agent.x == x
    assert agent.y == y
    assert agent.agent_id == profile.agent_id
    assert agent.profile == profile
    assert agent.alive is True


def test_spawn_raises_valueerror_for_water_tile(engine: SimulationEngine):
    """spawn() raises ValueError for water tile."""
    x, y = find_water(engine.world)
    profile = AgentProfile(
        agent_id=AgentID("test002"),
        name="WaterTest",
        archetype="survivalist",
        traits=PersonalityTraits(),
    )

    with pytest.raises(ValueError, match="water tile"):
        engine.registry.spawn(profile, x, y)


def test_spawn_raises_valueerror_for_out_of_bounds(engine: SimulationEngine):
    """spawn() raises ValueError for out-of-bounds position."""
    profile = AgentProfile(
        agent_id=AgentID("test003"),
        name="OutOfBounds",
        archetype="survivalist",
        traits=PersonalityTraits(),
    )

    with pytest.raises(ValueError, match="out of bounds"):
        engine.registry.spawn(profile, -1, 5)

    with pytest.raises(ValueError, match="out of bounds"):
        engine.registry.spawn(profile, 5, -1)

    with pytest.raises(ValueError, match="out of bounds"):
        engine.registry.spawn(profile, 999, 5)

    with pytest.raises(ValueError, match="out of bounds"):
        engine.registry.spawn(profile, 5, 999)


def test_spawn_places_agent_on_world_tile(engine: SimulationEngine):
    """spawn() places agent on world tile (check occupants)."""
    x, y = find_non_water(engine.world)
    profile = AgentProfile(
        agent_id=AgentID("test004"),
        name="OccupantTest",
        archetype="survivalist",
        traits=PersonalityTraits(),
    )

    _agent = engine.registry.spawn(profile, x, y)

    tile = engine.world.get_tile(x, y)
    assert tile is not None
    assert profile.agent_id in tile.occupants


def test_kill_moves_agent_from_living_to_dead(engine: SimulationEngine):
    """kill() moves agent from living to dead."""
    x, y = find_non_water(engine.world)
    profile = AgentProfile(
        agent_id=AgentID("test005"),
        name="KillTest",
        archetype="survivalist",
        traits=PersonalityTraits(),
    )

    agent = engine.registry.spawn(profile, x, y)
    assert profile.agent_id in [a.agent_id for a in engine.registry.living_agents()]

    engine.registry.kill(profile.agent_id, "test_death")

    assert profile.agent_id not in [a.agent_id for a in engine.registry.living_agents()]
    assert agent.alive is False


def test_kill_removes_from_world_occupants(engine: SimulationEngine):
    """kill() removes agent from world occupants."""
    x, y = find_non_water(engine.world)
    profile = AgentProfile(
        agent_id=AgentID("test006"),
        name="RemoveTest",
        archetype="survivalist",
        traits=PersonalityTraits(),
    )

    _agent = engine.registry.spawn(profile, x, y)
    tile = engine.world.get_tile(x, y)
    assert profile.agent_id in tile.occupants

    engine.registry.kill(profile.agent_id, "test_removal")

    assert profile.agent_id not in tile.occupants


def test_kill_records_death_cause(engine: SimulationEngine):
    """kill() records death cause."""
    x, y = find_non_water(engine.world)
    profile = AgentProfile(
        agent_id=AgentID("test007"),
        name="DeathCauseTest",
        archetype="survivalist",
        traits=PersonalityTraits(),
    )

    _agent = engine.registry.spawn(profile, x, y)
    cause = "starvation"

    engine.registry.kill(profile.agent_id, cause)

    assert engine.registry.death_cause(profile.agent_id) == cause


def test_get_returns_living_agent(engine: SimulationEngine):
    """get() returns living agent."""
    x, y = find_non_water(engine.world)
    profile = AgentProfile(
        agent_id=AgentID("test008"),
        name="GetTest",
        archetype="survivalist",
        traits=PersonalityTraits(),
    )

    spawned_agent = engine.registry.spawn(profile, x, y)
    retrieved_agent = engine.registry.get(profile.agent_id)

    assert retrieved_agent is spawned_agent
    assert retrieved_agent.agent_id == profile.agent_id


def test_get_returns_none_for_dead_agent(engine: SimulationEngine):
    """get() returns None for dead agent."""
    x, y = find_non_water(engine.world)
    profile = AgentProfile(
        agent_id=AgentID("test009"),
        name="DeadTest",
        archetype="survivalist",
        traits=PersonalityTraits(),
    )

    _agent = engine.registry.spawn(profile, x, y)
    engine.registry.kill(profile.agent_id)

    assert engine.registry.get(profile.agent_id) is None


def test_get_returns_none_for_unknown_id(engine: SimulationEngine):
    """get() returns None for unknown ID."""
    unknown_id = AgentID("unknown999")

    assert engine.registry.get(unknown_id) is None


def test_living_agents_returns_only_living(engine: SimulationEngine):
    """living_agents() returns only living agents."""
    positions = [find_non_water(engine.world) for _ in range(3)]

    profile1 = AgentProfile(
        agent_id=AgentID("test010"),
        name="Living1",
        archetype="survivalist",
        traits=PersonalityTraits(),
    )
    profile2 = AgentProfile(
        agent_id=AgentID("test011"),
        name="Living2",
        archetype="survivalist",
        traits=PersonalityTraits(),
    )
    profile3 = AgentProfile(
        agent_id=AgentID("test012"),
        name="ToBeKilled",
        archetype="survivalist",
        traits=PersonalityTraits(),
    )

    _agent1 = engine.registry.spawn(profile1, *positions[0])
    _agent2 = engine.registry.spawn(profile2, *positions[1])
    _agent3 = engine.registry.spawn(profile3, *positions[2])

    engine.registry.kill(profile3.agent_id)

    living = engine.registry.living_agents()
    living_ids = [a.agent_id for a in living]

    assert len(living) == 2
    assert profile1.agent_id in living_ids
    assert profile2.agent_id in living_ids
    assert profile3.agent_id not in living_ids


def test_all_agents_returns_living_and_dead(engine: SimulationEngine):
    """all_agents() returns living + dead."""
    positions = [find_non_water(engine.world) for _ in range(2)]

    profile1 = AgentProfile(
        agent_id=AgentID("test013"),
        name="AllLiving",
        archetype="survivalist",
        traits=PersonalityTraits(),
    )
    profile2 = AgentProfile(
        agent_id=AgentID("test014"),
        name="AllDead",
        archetype="survivalist",
        traits=PersonalityTraits(),
    )

    _agent1 = engine.registry.spawn(profile1, *positions[0])
    _agent2 = engine.registry.spawn(profile2, *positions[1])

    engine.registry.kill(profile2.agent_id)

    all_agents = engine.registry.all_agents()
    all_ids = [a.agent_id for a in all_agents]

    assert len(all_agents) == 2
    assert profile1.agent_id in all_ids
    assert profile2.agent_id in all_ids


def test_count_living_and_count_dead_properties(engine: SimulationEngine):
    """count_living and count_dead properties."""
    positions = [find_non_water(engine.world) for _ in range(3)]

    profiles = [
        AgentProfile(
            agent_id=AgentID(f"test{i:03d}"),
            name=f"CountTest{i}",
            archetype="survivalist",
            traits=PersonalityTraits(),
        )
        for i in range(15, 18)
    ]

    assert engine.registry.count_living == 0
    assert engine.registry.count_dead == 0

    for profile, pos in zip(profiles, positions, strict=False):
        engine.registry.spawn(profile, *pos)

    assert engine.registry.count_living == 3
    assert engine.registry.count_dead == 0

    engine.registry.kill(profiles[0].agent_id)

    assert engine.registry.count_living == 2
    assert engine.registry.count_dead == 1

    engine.registry.kill(profiles[1].agent_id)

    assert engine.registry.count_living == 1
    assert engine.registry.count_dead == 2


def test_register_and_get_awareness_loop_roundtrip(engine: SimulationEngine):
    """register_awareness_loop / get_awareness_loop round-trip."""
    agent_id = AgentID("test018")
    mock_loop = {"type": "awareness_loop", "data": "test"}

    engine.registry.register_awareness_loop(agent_id, mock_loop)
    retrieved = engine.registry.get_awareness_loop(agent_id)

    assert retrieved == mock_loop


def test_register_and_get_memory_roundtrip(engine: SimulationEngine):
    """register_memory / get_memory round-trip."""
    agent_id = AgentID("test019")
    mock_episodic = {"type": "episodic", "events": []}
    mock_social = {"type": "social", "relationships": {}}

    engine.registry.register_memory(agent_id, mock_episodic, mock_social)
    retrieved = engine.registry.get_memory(agent_id)

    assert retrieved is not None
    assert retrieved[0] == mock_episodic
    assert retrieved[1] == mock_social


def test_get_awareness_loop_returns_none_for_unregistered(engine: SimulationEngine):
    """get_awareness_loop returns None for unregistered agent."""
    unknown_id = AgentID("unknown_loop")

    assert engine.registry.get_awareness_loop(unknown_id) is None


def test_get_memory_returns_none_for_unregistered(engine: SimulationEngine):
    """get_memory returns None for unregistered agent."""
    unknown_id = AgentID("unknown_memory")

    assert engine.registry.get_memory(unknown_id) is None
