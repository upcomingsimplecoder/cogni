"""Tests for agents module: identity, archetypes, registry, and evolution.

Covers:
- AgentID: equality, hash, uniqueness, repr, custom value
- PersonalityTraits: defaults, as_dict, shift_trait, copy
- AgentProfile: construction with defaults and explicit traits
- ARCHETYPES: structure, trait ranges, required keys
- TraitEvolution: process_outcome for positive/negative events, clamping, learning rate
- AgentRegistry: spawn, kill, get, living_agents, count_living, count_dead,
  register_awareness_loop, register_memory
"""

from __future__ import annotations

import random

import pytest

from src.agents.archetypes import ARCHETYPES, find_valid_spawn, spawn_archetype
from src.agents.evolution import SIGNAL_MAP, TraitEvolution
from src.agents.identity import AgentID, AgentProfile, PersonalityTraits
from src.simulation.world import TileType

# ==============================================================================
# AgentID Tests
# ==============================================================================


class TestAgentID:
    """Test AgentID equality, hashing, uniqueness, and representation."""

    def test_equality_same_value(self):
        """Two AgentIDs with same value are equal."""
        id1 = AgentID("abc123")
        id2 = AgentID("abc123")
        assert id1 == id2

    def test_equality_different_value(self):
        """Two AgentIDs with different values are not equal."""
        id1 = AgentID("abc123")
        id2 = AgentID("xyz789")
        assert id1 != id2

    def test_equality_with_non_agentid(self):
        """AgentID is not equal to non-AgentID objects."""
        id1 = AgentID("abc123")
        assert id1 != "abc123"
        assert id1 != 123
        assert id1 is not None

    def test_hash_consistency(self):
        """AgentIDs with same value have same hash."""
        id1 = AgentID("abc123")
        id2 = AgentID("abc123")
        assert hash(id1) == hash(id2)

    def test_hash_usable_in_dict(self):
        """AgentIDs can be used as dictionary keys."""
        id1 = AgentID("key1")
        id2 = AgentID("key2")
        id3 = AgentID("key1")  # Same as id1

        d = {id1: "value1", id2: "value2"}
        assert d[id1] == "value1"
        assert d[id3] == "value1"  # Should find same key
        assert d[id2] == "value2"

    def test_hash_usable_in_set(self):
        """AgentIDs can be added to sets."""
        id1 = AgentID("a")
        id2 = AgentID("b")
        id3 = AgentID("a")  # Duplicate

        s = {id1, id2, id3}
        assert len(s) == 2  # Only 2 unique IDs

    def test_uniqueness_default_values(self):
        """AgentIDs created without value are unique."""
        id1 = AgentID()
        id2 = AgentID()
        id3 = AgentID()

        # All different
        assert id1 != id2
        assert id1 != id3
        assert id2 != id3

        # All have different values
        assert id1.value != id2.value
        assert id1.value != id3.value
        assert id2.value != id3.value

    def test_default_value_is_8_chars(self):
        """Default AgentID value is 8 characters (truncated UUID)."""
        id1 = AgentID()
        assert len(id1.value) == 8

    def test_custom_value(self):
        """AgentID accepts custom value."""
        id1 = AgentID("custom_id_12345")
        assert id1.value == "custom_id_12345"

    def test_repr(self):
        """__repr__ shows AgentID(value)."""
        id1 = AgentID("test123")
        assert repr(id1) == "AgentID(test123)"

    def test_str(self):
        """__str__ returns just the value."""
        id1 = AgentID("test456")
        assert str(id1) == "test456"


# ==============================================================================
# PersonalityTraits Tests
# ==============================================================================


class TestPersonalityTraits:
    """Test PersonalityTraits defaults, serialization, mutation, and copying."""

    def test_default_values_are_0_5(self):
        """All traits default to 0.5 (balanced)."""
        traits = PersonalityTraits()
        assert traits.cooperation_tendency == 0.5
        assert traits.curiosity == 0.5
        assert traits.risk_tolerance == 0.5
        assert traits.resource_sharing == 0.5
        assert traits.aggression == 0.5
        assert traits.sociability == 0.5

    def test_custom_trait_values(self):
        """Traits can be set to custom values."""
        traits = PersonalityTraits(
            cooperation_tendency=0.8,
            curiosity=0.2,
            risk_tolerance=0.9,
            resource_sharing=0.1,
            aggression=0.0,
            sociability=1.0,
        )
        assert traits.cooperation_tendency == 0.8
        assert traits.curiosity == 0.2
        assert traits.risk_tolerance == 0.9
        assert traits.resource_sharing == 0.1
        assert traits.aggression == 0.0
        assert traits.sociability == 1.0

    def test_as_dict_returns_all_traits(self):
        """as_dict() returns all traits as flat dictionary."""
        traits = PersonalityTraits(
            cooperation_tendency=0.7,
            curiosity=0.3,
            risk_tolerance=0.6,
            resource_sharing=0.4,
            aggression=0.2,
            sociability=0.8,
        )
        d = traits.as_dict()

        assert d == {
            "cooperation_tendency": 0.7,
            "curiosity": 0.3,
            "risk_tolerance": 0.6,
            "resource_sharing": 0.4,
            "aggression": 0.2,
            "sociability": 0.8,
        }

    def test_as_dict_has_all_trait_keys(self):
        """as_dict() includes all 6 trait keys."""
        traits = PersonalityTraits()
        d = traits.as_dict()

        expected_keys = {
            "cooperation_tendency",
            "curiosity",
            "risk_tolerance",
            "resource_sharing",
            "aggression",
            "sociability",
        }
        assert set(d.keys()) == expected_keys

    def test_shift_trait_positive_delta(self):
        """shift_trait() increases trait by delta."""
        traits = PersonalityTraits()
        traits.shift_trait("curiosity", 0.2)
        assert traits.curiosity == 0.7

    def test_shift_trait_negative_delta(self):
        """shift_trait() decreases trait by delta."""
        traits = PersonalityTraits()
        traits.shift_trait("aggression", -0.3)
        assert traits.aggression == 0.2

    def test_shift_trait_clamps_at_1_0(self):
        """shift_trait() clamps trait at 1.0 (upper bound)."""
        traits = PersonalityTraits(risk_tolerance=0.9)
        traits.shift_trait("risk_tolerance", 0.5)  # Would go to 1.4
        assert traits.risk_tolerance == 1.0

    def test_shift_trait_clamps_at_0_0(self):
        """shift_trait() clamps trait at 0.0 (lower bound)."""
        traits = PersonalityTraits(cooperation_tendency=0.1)
        traits.shift_trait("cooperation_tendency", -0.5)  # Would go to -0.4
        assert traits.cooperation_tendency == 0.0

    def test_shift_trait_multiple_times(self):
        """shift_trait() can be called multiple times."""
        traits = PersonalityTraits()
        traits.shift_trait("sociability", 0.1)
        traits.shift_trait("sociability", 0.1)
        traits.shift_trait("sociability", 0.1)
        # 0.5 + 0.1 + 0.1 + 0.1 = 0.8 (using float comparison for precision)
        assert abs(traits.sociability - 0.8) < 0.001

    def test_copy_returns_deep_copy(self):
        """copy() returns deep copy that doesn't share state."""
        original = PersonalityTraits(
            cooperation_tendency=0.8,
            curiosity=0.3,
            risk_tolerance=0.6,
        )
        copied = original.copy()

        # Values match initially
        assert copied.cooperation_tendency == 0.8
        assert copied.curiosity == 0.3
        assert copied.risk_tolerance == 0.6

        # Mutating copy doesn't affect original
        copied.shift_trait("cooperation_tendency", 0.2)
        assert copied.cooperation_tendency == 1.0
        assert original.cooperation_tendency == 0.8  # Unchanged

    def test_copy_preserves_all_traits(self):
        """copy() preserves all trait values."""
        original = PersonalityTraits(
            cooperation_tendency=0.1,
            curiosity=0.2,
            risk_tolerance=0.3,
            resource_sharing=0.4,
            aggression=0.5,
            sociability=0.6,
        )
        copied = original.copy()

        assert copied.as_dict() == original.as_dict()


# ==============================================================================
# AgentProfile Tests
# ==============================================================================


class TestAgentProfile:
    """Test AgentProfile construction with defaults and explicit traits."""

    def test_construction_with_defaults(self):
        """AgentProfile can be constructed with default traits."""
        agent_id = AgentID("test001")
        profile = AgentProfile(
            agent_id=agent_id,
            name="TestAgent",
            archetype="survivalist",
        )

        assert profile.agent_id == agent_id
        assert profile.name == "TestAgent"
        assert profile.archetype == "survivalist"
        assert profile.traits.cooperation_tendency == 0.5  # Default

    def test_construction_with_explicit_traits(self):
        """AgentProfile can be constructed with explicit traits."""
        agent_id = AgentID("test002")
        traits = PersonalityTraits(
            cooperation_tendency=0.9,
            curiosity=0.8,
            risk_tolerance=0.7,
        )
        profile = AgentProfile(
            agent_id=agent_id,
            name="Diplomat",
            archetype="diplomat",
            traits=traits,
        )

        assert profile.agent_id == agent_id
        assert profile.name == "Diplomat"
        assert profile.archetype == "diplomat"
        assert profile.traits.cooperation_tendency == 0.9
        assert profile.traits.curiosity == 0.8
        assert profile.traits.risk_tolerance == 0.7

    def test_traits_not_shared_between_profiles(self):
        """Traits are not shared between profile instances when using the same object."""
        traits = PersonalityTraits(curiosity=0.7)
        profile1 = AgentProfile(
            agent_id=AgentID("p1"),
            name="Agent1",
            archetype="explorer",
            traits=traits,
        )
        profile2 = AgentProfile(
            agent_id=AgentID("p2"),
            name="Agent2",
            archetype="explorer",
            traits=traits,  # Same object reference
        )

        # Since both profiles share the same traits object, mutating one affects the other
        # This is a demonstration of why .copy() should be used
        profile1.traits.shift_trait("curiosity", 0.2)
        assert abs(profile1.traits.curiosity - 0.9) < 0.001  # Float comparison
        assert abs(profile2.traits.curiosity - 0.9) < 0.001  # Both changed - same object!

        # Using copy() prevents this:
        traits_independent = PersonalityTraits(curiosity=0.7)
        profile3 = AgentProfile(
            agent_id=AgentID("p3"),
            name="Agent3",
            archetype="explorer",
            traits=traits_independent.copy(),  # Copy!
        )
        profile1.traits.shift_trait("curiosity", 0.1)
        assert abs(profile3.traits.curiosity - 0.7) < 0.001  # Unchanged


# ==============================================================================
# ARCHETYPES Tests
# ==============================================================================


class TestArchetypes:
    """Test ARCHETYPES dictionary structure and trait ranges."""

    def test_all_five_archetypes_exist(self):
        """ARCHETYPES contains all 5 expected archetypes."""
        expected = {"gatherer", "explorer", "diplomat", "aggressor", "survivalist"}
        assert set(ARCHETYPES.keys()) == expected

    def test_all_archetypes_have_required_keys(self):
        """Each archetype has description, traits, color, symbol."""
        required_keys = {"description", "traits", "color", "symbol"}
        for name, archetype in ARCHETYPES.items():
            assert set(archetype.keys()) == required_keys, f"{name} missing keys"

    def test_all_traits_in_valid_range(self):
        """All archetype traits are in [0, 1] range."""
        for name, archetype in ARCHETYPES.items():
            traits = archetype["traits"]
            trait_dict = traits.as_dict()
            for trait_name, value in trait_dict.items():
                assert 0.0 <= value <= 1.0, f"{name}.{trait_name} = {value} is out of range"

    def test_gatherer_archetype(self):
        """Gatherer has expected traits profile."""
        gatherer = ARCHETYPES["gatherer"]
        assert gatherer["description"] == "Focuses on resource collection and hoarding"
        assert gatherer["color"] == "green"
        assert gatherer["symbol"] == "G"
        traits = gatherer["traits"]
        assert traits.cooperation_tendency == 0.3
        assert traits.curiosity == 0.2
        assert traits.resource_sharing == 0.2
        assert traits.aggression == 0.1

    def test_explorer_archetype(self):
        """Explorer has high curiosity and risk tolerance."""
        explorer = ARCHETYPES["explorer"]
        assert explorer["description"] == "Roams widely, discovers resources, shares information"
        assert explorer["color"] == "cyan"
        assert explorer["symbol"] == "E"
        traits = explorer["traits"]
        assert traits.curiosity == 0.9
        assert traits.risk_tolerance == 0.7

    def test_diplomat_archetype(self):
        """Diplomat has high cooperation and sociability."""
        diplomat = ARCHETYPES["diplomat"]
        assert diplomat["description"] == "Seeks out other agents, negotiates, builds relationships"
        assert diplomat["color"] == "yellow"
        assert diplomat["symbol"] == "D"
        traits = diplomat["traits"]
        assert traits.cooperation_tendency == 0.9
        assert traits.resource_sharing == 0.8
        assert traits.sociability == 0.9
        assert traits.aggression == 0.05

    def test_aggressor_archetype(self):
        """Aggressor has high aggression and risk tolerance."""
        aggressor = ARCHETYPES["aggressor"]
        assert aggressor["description"] == (
            "Territorial, takes resources from others, low cooperation"
        )
        assert aggressor["color"] == "red"
        assert aggressor["symbol"] == "A"
        traits = aggressor["traits"]
        assert traits.cooperation_tendency == 0.1
        assert traits.aggression == 0.9
        assert traits.risk_tolerance == 0.8
        assert traits.resource_sharing == 0.05

    def test_survivalist_archetype(self):
        """Survivalist has balanced traits."""
        survivalist = ARCHETYPES["survivalist"]
        assert survivalist["description"] == "Balanced, cautious, self-sufficient"
        assert survivalist["color"] == "white"
        assert survivalist["symbol"] == "S"
        traits = survivalist["traits"]
        assert traits.cooperation_tendency == 0.5
        assert traits.curiosity == 0.5
        assert traits.risk_tolerance == 0.5

    def test_archetypes_have_different_trait_profiles(self):
        """Archetypes have distinct trait profiles (not all same)."""
        trait_profiles = []
        for _name, archetype in ARCHETYPES.items():
            traits = archetype["traits"]
            # Create tuple of all trait values as signature
            signature = tuple(traits.as_dict().values())
            trait_profiles.append(signature)

        # All archetypes should have unique trait signatures
        assert len(trait_profiles) == len(set(trait_profiles))


# ==============================================================================
# find_valid_spawn Tests
# ==============================================================================


class TestFindValidSpawn:
    """Test find_valid_spawn finds non-water positions."""

    def test_finds_non_water_position(self, world):
        """find_valid_spawn returns a non-water position."""
        x, y = find_valid_spawn(world)
        tile = world.get_tile(x, y)
        assert tile is not None
        assert tile.type != TileType.WATER

    def test_position_within_bounds(self, world):
        """find_valid_spawn returns position within world bounds (with margin)."""
        x, y = find_valid_spawn(world)
        # Checks bounds are in [5, width-6] and [5, height-6]
        assert 5 <= x < world.width - 5
        assert 5 <= y < world.height - 5

    def test_with_custom_rng_seed(self, world):
        """find_valid_spawn with custom RNG produces deterministic results."""
        rng1 = random.Random(12345)
        rng2 = random.Random(12345)

        pos1 = find_valid_spawn(world, rng1)
        pos2 = find_valid_spawn(world, rng2)

        assert pos1 == pos2  # Same seed = same result

    def test_raises_if_no_valid_position(self):
        """find_valid_spawn raises RuntimeError if no valid position exists."""
        # Create a tiny all-water world (must be at least 12x12 for margins [5, width-6])
        from src.simulation.world import World

        world = World(12, 12, seed=42)
        # Manually make all tiles water
        for y in range(12):
            for x in range(12):
                tile = world.get_tile(x, y)
                if tile:
                    tile.type = TileType.WATER

        with pytest.raises(RuntimeError, match="No valid spawn position found"):
            find_valid_spawn(world)


# ==============================================================================
# spawn_archetype Tests
# ==============================================================================


class TestSpawnArchetype:
    """Test spawn_archetype creates agents with correct setup."""

    def _find_non_water(self, world):
        """Helper to find a non-water position."""
        for y in range(world.height):
            for x in range(world.width):
                tile = world.get_tile(x, y)
                if tile and tile.type != TileType.WATER:
                    return x, y
        raise RuntimeError("No non-water tile in test world")

    def test_spawns_agent_with_correct_archetype(self, engine):
        """spawn_archetype creates agent with correct archetype."""
        x, y = self._find_non_water(engine.world)
        agent = spawn_archetype(
            archetype_name="gatherer",
            agent_name="TestGatherer",
            engine=engine,
            x=x,
            y=y,
        )

        assert agent.profile.archetype == "gatherer"
        assert agent.profile.name == "TestGatherer"

    def test_spawns_agent_with_archetype_traits(self, engine):
        """spawn_archetype copies archetype traits to agent."""
        x, y = self._find_non_water(engine.world)
        agent = spawn_archetype(
            archetype_name="diplomat",
            agent_name="TestDiplomat",
            engine=engine,
            x=x,
            y=y,
        )

        # Should have diplomat's trait profile
        assert agent.profile.traits.cooperation_tendency == 0.9
        assert agent.profile.traits.sociability == 0.9

    def test_spawns_agent_at_specified_position(self, engine):
        """spawn_archetype places agent at specified x, y."""
        x, y = self._find_non_water(engine.world)
        agent = spawn_archetype(
            archetype_name="explorer",
            agent_name="TestExplorer",
            engine=engine,
            x=x,
            y=y,
        )

        assert agent.x == x
        assert agent.y == y

    def test_auto_finds_position_if_not_specified(self, engine):
        """spawn_archetype auto-finds position if x, y not provided."""
        agent = spawn_archetype(
            archetype_name="survivalist",
            agent_name="AutoPos",
            engine=engine,
        )

        # Should be within valid bounds
        assert 0 <= agent.x < engine.world.width
        assert 0 <= agent.y < engine.world.height

        # Should be non-water
        tile = engine.world.get_tile(agent.x, agent.y)
        assert tile.type != TileType.WATER

    def test_gives_starting_inventory(self, engine):
        """spawn_archetype gives agent starting inventory."""
        x, y = self._find_non_water(engine.world)
        agent = spawn_archetype(
            archetype_name="gatherer",
            agent_name="InventoryTest",
            engine=engine,
            x=x,
            y=y,
        )

        assert agent.inventory.get("water") == 2
        assert agent.inventory.get("berries") == 1

    def test_registers_awareness_loop(self, engine):
        """spawn_archetype registers awareness loop."""
        x, y = self._find_non_water(engine.world)
        agent = spawn_archetype(
            archetype_name="explorer",
            agent_name="AwarenessTest",
            engine=engine,
            x=x,
            y=y,
        )

        loop = engine.registry.get_awareness_loop(agent.profile.agent_id)
        assert loop is not None

    def test_registers_memory_systems(self, engine):
        """spawn_archetype registers episodic and social memory."""
        x, y = self._find_non_water(engine.world)
        agent = spawn_archetype(
            archetype_name="diplomat",
            agent_name="MemoryTest",
            engine=engine,
            x=x,
            y=y,
        )

        memory = engine.registry.get_memory(agent.profile.agent_id)
        assert memory is not None
        episodic, social = memory
        assert episodic is not None
        assert social is not None

    def test_traits_are_copied_not_shared(self, engine):
        """spawn_archetype creates independent trait copies for each agent."""
        positions = [self._find_non_water(engine.world) for _ in range(2)]

        agent1 = spawn_archetype(
            archetype_name="gatherer",
            agent_name="Agent1",
            engine=engine,
            x=positions[0][0],
            y=positions[0][1],
        )
        agent2 = spawn_archetype(
            archetype_name="gatherer",
            agent_name="Agent2",
            engine=engine,
            x=positions[1][0],
            y=positions[1][1],
        )

        # Modify agent1's traits
        agent1.profile.traits.shift_trait("curiosity", 0.3)

        # agent2's traits should be unchanged
        assert agent1.profile.traits.curiosity == 0.5  # 0.2 + 0.3
        assert agent2.profile.traits.curiosity == 0.2  # Original gatherer value


# ==============================================================================
# TraitEvolution Tests
# ==============================================================================


class TestTraitEvolution:
    """Test TraitEvolution processes outcomes and updates traits correctly."""

    def test_positive_outcome_increases_traits(self):
        """process_outcome with positive result increases affected traits."""
        evolution = TraitEvolution(learning_rate=0.1)
        traits = PersonalityTraits()

        changes = evolution.process_outcome(traits, "shared_resource", was_positive=True)

        # shared_resource affects cooperation_tendency, resource_sharing, sociability
        assert traits.cooperation_tendency == 0.6  # 0.5 + 0.1
        assert traits.resource_sharing == 0.6
        assert traits.sociability == 0.6

        # Returns list of changes
        assert len(changes) == 3
        assert ("cooperation_tendency", 0.6) in changes
        assert ("resource_sharing", 0.6) in changes
        assert ("sociability", 0.6) in changes

    def test_negative_outcome_decreases_traits(self):
        """process_outcome with negative result decreases affected traits."""
        evolution = TraitEvolution(learning_rate=0.1)
        traits = PersonalityTraits()

        changes = evolution.process_outcome(traits, "attacked_agent", was_positive=False)

        # attacked_agent affects aggression, risk_tolerance
        assert traits.aggression == 0.4  # 0.5 - 0.1
        assert traits.risk_tolerance == 0.4

        assert len(changes) == 2
        assert ("aggression", 0.4) in changes
        assert ("risk_tolerance", 0.4) in changes

    def test_clamping_at_upper_bound(self):
        """process_outcome clamps traits at 1.0."""
        evolution = TraitEvolution(learning_rate=0.3)
        traits = PersonalityTraits(curiosity=0.9)

        evolution.process_outcome(traits, "explored_new_area", was_positive=True)

        # Would go to 1.2, but clamped at 1.0
        assert traits.curiosity == 1.0

    def test_clamping_at_lower_bound(self):
        """process_outcome clamps traits at 0.0."""
        evolution = TraitEvolution(learning_rate=0.3)
        traits = PersonalityTraits(aggression=0.1)

        evolution.process_outcome(traits, "attacked_agent", was_positive=False)

        # Would go to -0.2, but clamped at 0.0
        assert traits.aggression == 0.0

    def test_unknown_event_type_returns_empty(self):
        """process_outcome with unknown event type returns empty list."""
        evolution = TraitEvolution(learning_rate=0.1)
        traits = PersonalityTraits()
        original_dict = traits.as_dict()

        changes = evolution.process_outcome(traits, "unknown_event_type_xyz", was_positive=True)

        assert changes == []
        # No traits changed
        assert traits.as_dict() == original_dict

    def test_custom_learning_rate(self):
        """TraitEvolution respects custom learning rate."""
        evolution = TraitEvolution(learning_rate=0.05)
        traits = PersonalityTraits()

        evolution.process_outcome(traits, "explored_new_area", was_positive=True)

        # curiosity should increase by 0.05
        assert traits.curiosity == 0.55

    def test_multiple_outcomes_accumulate(self):
        """Multiple process_outcome calls accumulate changes."""
        evolution = TraitEvolution(learning_rate=0.01)
        traits = PersonalityTraits()

        # Simulate 10 positive exploration events
        for _ in range(10):
            evolution.process_outcome(traits, "explored_new_area", was_positive=True)

        # curiosity should have increased by 0.01 * 10 = 0.1
        assert abs(traits.curiosity - 0.6) < 0.001  # Float comparison

    def test_signal_map_coverage(self):
        """SIGNAL_MAP contains expected event types."""
        expected_events = {
            "shared_resource",
            "explored_new_area",
            "attacked_agent",
            "received_help",
            "trade_completed",
            "survived_low_health",
            "was_attacked",
            "found_resource",
            "social_interaction",
            "sent_message",
            "received_message",
            "joined_coalition",
        }
        assert set(SIGNAL_MAP.keys()) == expected_events

    def test_signal_map_traits_are_valid(self):
        """All traits in SIGNAL_MAP are valid PersonalityTraits attributes."""
        valid_traits = {
            "cooperation_tendency",
            "curiosity",
            "risk_tolerance",
            "resource_sharing",
            "aggression",
            "sociability",
        }
        for event_type, affected_traits in SIGNAL_MAP.items():
            for trait_name in affected_traits:
                assert trait_name in valid_traits, (
                    f"Invalid trait '{trait_name}' in SIGNAL_MAP['{event_type}']"
                )

    def test_all_event_types_have_traits(self):
        """All events in SIGNAL_MAP have at least one affected trait."""
        for event_type, affected_traits in SIGNAL_MAP.items():
            assert len(affected_traits) > 0, f"{event_type} has no affected traits"


# ==============================================================================
# AgentRegistry Tests (requires engine fixture)
# ==============================================================================


class TestAgentRegistry:
    """Test AgentRegistry spawn, kill, get, living_agents, and registration."""

    def test_spawn_creates_agent_in_registry(self, engine):
        """spawn() adds agent to registry."""
        profile = AgentProfile(
            agent_id=AgentID("reg001"),
            name="RegistryTest",
            archetype="survivalist",
        )

        agent = engine.registry.spawn(profile, 8, 8)

        assert engine.registry.get(profile.agent_id) == agent
        assert profile.agent_id in [a.agent_id for a in engine.registry.living_agents()]

    def test_spawn_places_agent_on_world(self, engine):
        """spawn() places agent on world tile."""
        profile = AgentProfile(
            agent_id=AgentID("reg002"),
            name="WorldTest",
            archetype="survivalist",
        )

        engine.registry.spawn(profile, 8, 8)

        tile = engine.world.get_tile(8, 8)
        assert profile.agent_id in tile.occupants

    def test_spawn_raises_for_water_tile(self, engine):
        """spawn() raises ValueError for water tile."""
        # Find a water tile
        water_pos = None
        for y in range(engine.world.height):
            for x in range(engine.world.width):
                tile = engine.world.get_tile(x, y)
                if tile and tile.type == TileType.WATER:
                    water_pos = (x, y)
                    break
            if water_pos:
                break

        if water_pos:
            profile = AgentProfile(
                agent_id=AgentID("reg003"),
                name="WaterTest",
                archetype="survivalist",
            )

            with pytest.raises(ValueError, match="water tile"):
                engine.registry.spawn(profile, *water_pos)

    def test_spawn_raises_for_out_of_bounds(self, engine):
        """spawn() raises ValueError for out of bounds position."""
        profile = AgentProfile(
            agent_id=AgentID("reg004"),
            name="BoundsTest",
            archetype="survivalist",
        )

        with pytest.raises(ValueError, match="out of bounds"):
            engine.registry.spawn(profile, -1, 5)

    def test_kill_moves_agent_to_dead(self, engine):
        """kill() moves agent from living to dead."""
        profile = AgentProfile(
            agent_id=AgentID("reg005"),
            name="KillTest",
            archetype="survivalist",
        )

        agent = engine.registry.spawn(profile, 8, 8)
        engine.registry.kill(profile.agent_id, "test_death")

        assert engine.registry.get(profile.agent_id) is None
        assert agent.alive is False

    def test_kill_records_death_cause(self, engine):
        """kill() records death cause."""
        profile = AgentProfile(
            agent_id=AgentID("reg006"),
            name="CauseTest",
            archetype="survivalist",
        )

        engine.registry.spawn(profile, 8, 8)
        engine.registry.kill(profile.agent_id, "starvation")

        assert engine.registry.death_cause(profile.agent_id) == "starvation"

    def test_kill_removes_from_world(self, engine):
        """kill() removes agent from world occupants."""
        profile = AgentProfile(
            agent_id=AgentID("reg007"),
            name="RemoveTest",
            archetype="survivalist",
        )

        engine.registry.spawn(profile, 8, 8)
        tile = engine.world.get_tile(8, 8)
        assert profile.agent_id in tile.occupants

        engine.registry.kill(profile.agent_id, "test_removal")

        assert profile.agent_id not in tile.occupants

    def test_get_returns_living_agent(self, engine):
        """get() returns living agent."""
        profile = AgentProfile(
            agent_id=AgentID("reg008"),
            name="GetTest",
            archetype="survivalist",
        )

        spawned = engine.registry.spawn(profile, 8, 8)
        retrieved = engine.registry.get(profile.agent_id)

        assert retrieved is spawned

    def test_get_returns_none_for_dead_agent(self, engine):
        """get() returns None for dead agent."""
        profile = AgentProfile(
            agent_id=AgentID("reg009"),
            name="DeadGetTest",
            archetype="survivalist",
        )

        engine.registry.spawn(profile, 8, 8)
        engine.registry.kill(profile.agent_id)

        assert engine.registry.get(profile.agent_id) is None

    def test_get_returns_none_for_unknown_id(self, engine):
        """get() returns None for unknown agent ID."""
        unknown_id = AgentID("unknown_id_999")
        assert engine.registry.get(unknown_id) is None

    def test_living_agents_returns_all_living(self, engine):
        """living_agents() returns all living agents."""
        profiles = [
            AgentProfile(
                agent_id=AgentID(f"reg{i:03d}"),
                name=f"Agent{i}",
                archetype="survivalist",
            )
            for i in range(10, 13)
        ]

        for i, profile in enumerate(profiles):
            engine.registry.spawn(profile, 8 + i, 8)

        living = engine.registry.living_agents()
        living_ids = [a.agent_id for a in living]

        assert len(living) == 3
        for profile in profiles:
            assert profile.agent_id in living_ids

    def test_living_agents_excludes_dead(self, engine):
        """living_agents() excludes dead agents."""
        profiles = [
            AgentProfile(
                agent_id=AgentID(f"reg{i:03d}"),
                name=f"Agent{i}",
                archetype="survivalist",
            )
            for i in range(13, 16)
        ]

        for i, profile in enumerate(profiles):
            engine.registry.spawn(profile, 8 + i, 8)

        # Kill one agent
        engine.registry.kill(profiles[1].agent_id)

        living = engine.registry.living_agents()
        living_ids = [a.agent_id for a in living]

        assert len(living) == 2
        assert profiles[0].agent_id in living_ids
        assert profiles[2].agent_id in living_ids
        assert profiles[1].agent_id not in living_ids

    def test_count_living(self, engine):
        """count_living property returns correct count."""
        assert engine.registry.count_living == 0

        profile1 = AgentProfile(
            agent_id=AgentID("cnt001"),
            name="Count1",
            archetype="survivalist",
        )
        profile2 = AgentProfile(
            agent_id=AgentID("cnt002"),
            name="Count2",
            archetype="survivalist",
        )

        engine.registry.spawn(profile1, 8, 8)
        assert engine.registry.count_living == 1

        engine.registry.spawn(profile2, 9, 8)
        assert engine.registry.count_living == 2

        engine.registry.kill(profile1.agent_id)
        assert engine.registry.count_living == 1

    def test_count_dead(self, engine):
        """count_dead property returns correct count."""
        assert engine.registry.count_dead == 0

        profiles = [
            AgentProfile(
                agent_id=AgentID(f"cnt{i:03d}"),
                name=f"Dead{i}",
                archetype="survivalist",
            )
            for i in range(3, 6)
        ]

        for i, profile in enumerate(profiles):
            engine.registry.spawn(profile, 8 + i, 8)

        assert engine.registry.count_dead == 0

        engine.registry.kill(profiles[0].agent_id)
        assert engine.registry.count_dead == 1

        engine.registry.kill(profiles[1].agent_id)
        assert engine.registry.count_dead == 2

        engine.registry.kill(profiles[2].agent_id)
        assert engine.registry.count_dead == 3

    def test_register_awareness_loop(self, engine):
        """register_awareness_loop stores and retrieves loop."""
        agent_id = AgentID("loop001")
        mock_loop = {"type": "test_loop", "data": "test"}

        engine.registry.register_awareness_loop(agent_id, mock_loop)
        retrieved = engine.registry.get_awareness_loop(agent_id)

        assert retrieved == mock_loop

    def test_get_awareness_loop_returns_none_for_unregistered(self, engine):
        """get_awareness_loop returns None for unregistered agent."""
        unknown_id = AgentID("loop_unknown")
        assert engine.registry.get_awareness_loop(unknown_id) is None

    def test_register_memory(self, engine):
        """register_memory stores and retrieves memory systems."""
        agent_id = AgentID("mem001")
        mock_episodic = {"type": "episodic", "events": []}
        mock_social = {"type": "social", "relationships": {}}

        engine.registry.register_memory(agent_id, mock_episodic, mock_social)
        retrieved = engine.registry.get_memory(agent_id)

        assert retrieved is not None
        assert retrieved[0] == mock_episodic
        assert retrieved[1] == mock_social

    def test_get_memory_returns_none_for_unregistered(self, engine):
        """get_memory returns None for unregistered agent."""
        unknown_id = AgentID("mem_unknown")
        assert engine.registry.get_memory(unknown_id) is None

    def test_all_agents_includes_living_and_dead(self, engine):
        """all_agents() returns both living and dead agents."""
        profiles = [
            AgentProfile(
                agent_id=AgentID(f"all{i:03d}"),
                name=f"All{i}",
                archetype="survivalist",
            )
            for i in range(20, 23)
        ]

        for i, profile in enumerate(profiles):
            engine.registry.spawn(profile, 8 + i, 8)

        engine.registry.kill(profiles[1].agent_id)

        all_agents = engine.registry.all_agents()
        all_ids = [a.agent_id for a in all_agents]

        assert len(all_agents) == 3
        for profile in profiles:
            assert profile.agent_id in all_ids
