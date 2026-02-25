"""Tests for personality, archetypes, and trait evolution."""

from __future__ import annotations

from src.agents.archetypes import ARCHETYPES, spawn_archetype
from src.agents.evolution import TraitEvolution
from src.agents.identity import PersonalityTraits
from src.awareness.types import AgentSummary, Reflection, Sensation, TileSummary
from src.cognition.strategies.personality import PersonalityStrategy
from src.config import SimulationConfig
from src.simulation.actions import ActionType
from src.simulation.engine import SimulationEngine


class TestArchetypes:
    """Tests for archetype definitions."""

    def test_archetypes_has_five_entries(self):
        """ARCHETYPES dict has 5 predefined archetypes."""
        assert len(ARCHETYPES) == 5

    def test_archetypes_has_expected_types(self):
        """ARCHETYPES contains gatherer, explorer, diplomat, aggressor, survivalist."""
        assert "gatherer" in ARCHETYPES
        assert "explorer" in ARCHETYPES
        assert "diplomat" in ARCHETYPES
        assert "aggressor" in ARCHETYPES
        assert "survivalist" in ARCHETYPES

    def test_each_archetype_has_traits(self):
        """Each archetype has personality traits defined."""
        for _name, arch in ARCHETYPES.items():
            assert "traits" in arch
            assert isinstance(arch["traits"], PersonalityTraits)

    def test_each_archetype_has_color(self):
        """Each archetype has a color defined."""
        for _name, arch in ARCHETYPES.items():
            assert "color" in arch
            assert isinstance(arch["color"], str)

    def test_each_archetype_has_symbol(self):
        """Each archetype has a symbol defined."""
        for _name, arch in ARCHETYPES.items():
            assert "symbol" in arch
            assert isinstance(arch["symbol"], str)


class TestSpawnArchetype:
    """Tests for spawn_archetype() function."""

    def test_spawn_archetype_creates_agent_with_correct_archetype_name(self):
        """spawn_archetype() creates agent with correct archetype."""
        config = SimulationConfig(world_width=16, world_height=16, seed=42)
        engine = SimulationEngine(config)

        agent = spawn_archetype(
            archetype_name="explorer",
            agent_name="test_explorer",
            engine=engine,
            x=8,
            y=8,
        )

        assert agent.profile.archetype == "explorer"
        assert agent.profile.name == "test_explorer"

    def test_spawn_archetype_registers_awareness_loop(self):
        """spawn_archetype() registers awareness loop for agent."""
        config = SimulationConfig(world_width=16, world_height=16, seed=42)
        engine = SimulationEngine(config)

        agent = spawn_archetype(
            archetype_name="diplomat",
            agent_name="test_diplomat",
            engine=engine,
            x=8,
            y=8,
        )

        awareness_loop = engine.registry.get_awareness_loop(agent.agent_id)
        assert awareness_loop is not None

    def test_spawn_archetype_registers_memory(self):
        """spawn_archetype() registers episodic and social memory."""
        config = SimulationConfig(world_width=16, world_height=16, seed=42)
        engine = SimulationEngine(config)

        agent = spawn_archetype(
            archetype_name="gatherer",
            agent_name="test_gatherer",
            engine=engine,
            x=8,
            y=8,
        )

        memory = engine.registry.get_memory(agent.agent_id)
        assert memory is not None
        episodic, social = memory
        assert episodic is not None
        assert social is not None


class TestPersonalityTraits:
    """Tests for PersonalityTraits operations."""

    def test_copy_creates_independent_copy(self):
        """copy() creates an independent copy of traits."""
        traits1 = PersonalityTraits(
            cooperation_tendency=0.8,
            curiosity=0.6,
            risk_tolerance=0.4,
        )
        traits2 = traits1.copy()

        # Should be equal initially
        assert traits2.cooperation_tendency == 0.8
        assert traits2.curiosity == 0.6
        assert traits2.risk_tolerance == 0.4

        # Modifying traits2 should not affect traits1
        traits2.cooperation_tendency = 0.2
        assert traits1.cooperation_tendency == 0.8
        assert traits2.cooperation_tendency == 0.2

    def test_shift_trait_clamps_to_zero(self):
        """shift_trait() clamps values to minimum of 0.0."""
        traits = PersonalityTraits(curiosity=0.3)
        traits.shift_trait("curiosity", -0.5)

        # Should clamp to 0.0, not go negative
        assert traits.curiosity == 0.0

    def test_shift_trait_clamps_to_one(self):
        """shift_trait() clamps values to maximum of 1.0."""
        traits = PersonalityTraits(aggression=0.9)
        traits.shift_trait("aggression", 0.3)

        # Should clamp to 1.0, not exceed
        assert traits.aggression == 1.0

    def test_shift_trait_positive_delta(self):
        """shift_trait() increases trait value with positive delta."""
        traits = PersonalityTraits(sociability=0.5)
        traits.shift_trait("sociability", 0.2)

        assert traits.sociability == 0.7

    def test_shift_trait_negative_delta(self):
        """shift_trait() decreases trait value with negative delta."""
        traits = PersonalityTraits(resource_sharing=0.6)
        traits.shift_trait("resource_sharing", -0.1)

        assert traits.resource_sharing == 0.5

    def test_as_dict_returns_all_six_traits(self):
        """as_dict() returns dictionary with all 6 traits."""
        traits = PersonalityTraits()
        trait_dict = traits.as_dict()

        assert len(trait_dict) == 6
        assert "cooperation_tendency" in trait_dict
        assert "curiosity" in trait_dict
        assert "risk_tolerance" in trait_dict
        assert "resource_sharing" in trait_dict
        assert "aggression" in trait_dict
        assert "sociability" in trait_dict


class TestTraitEvolution:
    """Tests for TraitEvolution outcome processing."""

    def test_process_outcome_shifts_traits_on_known_events(self):
        """process_outcome() shifts traits for recognized event types."""
        traits = PersonalityTraits(curiosity=0.5)
        evolution = TraitEvolution(learning_rate=0.1)

        changes = evolution.process_outcome(
            traits=traits,
            event_type="explored_new_area",
            was_positive=True,
        )

        # Should return changes
        assert len(changes) > 0
        # Curiosity should have increased
        assert traits.curiosity > 0.5

    def test_process_outcome_returns_empty_for_unknown_events(self):
        """process_outcome() returns empty list for unknown event types."""
        traits = PersonalityTraits(curiosity=0.5)
        evolution = TraitEvolution(learning_rate=0.1)

        changes = evolution.process_outcome(
            traits=traits,
            event_type="unknown_event_type",
            was_positive=True,
        )

        assert len(changes) == 0
        # Traits unchanged
        assert traits.curiosity == 0.5

    def test_process_outcome_positive_events_shift_traits_up(self):
        """Positive outcomes increase relevant traits."""
        traits = PersonalityTraits(cooperation_tendency=0.5, sociability=0.5)
        evolution = TraitEvolution(learning_rate=0.1)

        evolution.process_outcome(
            traits=traits,
            event_type="trade_completed",
            was_positive=True,
        )

        # Both cooperation_tendency and sociability should increase
        assert traits.cooperation_tendency > 0.5
        assert traits.sociability > 0.5

    def test_process_outcome_negative_events_shift_traits_down(self):
        """Negative outcomes decrease relevant traits."""
        traits = PersonalityTraits(cooperation_tendency=0.5, resource_sharing=0.5)
        evolution = TraitEvolution(learning_rate=0.1)

        evolution.process_outcome(
            traits=traits,
            event_type="shared_resource",
            was_positive=False,
        )

        # Both should decrease
        assert traits.cooperation_tendency < 0.5
        assert traits.resource_sharing < 0.5


class TestPersonalityStrategy:
    """Tests for PersonalityStrategy decision making."""

    def test_rest_when_energy_low(self):
        """Strategy chooses rest when energy critically low."""
        strategy = PersonalityStrategy()

        sensation = Sensation(
            tick=0,
            own_needs={"hunger": 50, "thirst": 50, "energy": 5, "health": 100},
            own_position=(8, 8),
            own_inventory={},
            visible_tiles=[],
            visible_agents=[],
            own_traits={"risk_tolerance": 0.5},
        )
        reflection = Reflection(
            last_action_succeeded=True,
            need_trends={
                "hunger": "stable",
                "thirst": "stable",
                "energy": "declining",
                "health": "stable",
            },
        )

        intention = strategy.form_intention(sensation, reflection)

        assert intention.primary_goal == "rest"

    def test_attack_when_aggressive_and_agent_on_same_tile(self):
        """Aggressive agent attacks when another agent is on same tile."""
        strategy = PersonalityStrategy()

        target_agent_id = "target_123"
        sensation = Sensation(
            tick=0,
            own_needs={"hunger": 60, "thirst": 60, "energy": 70, "health": 100},
            own_position=(8, 8),
            own_inventory={},
            visible_tiles=[
                TileSummary(x=8, y=8, tile_type="grass", resources=[], occupants=[target_agent_id])
            ],
            visible_agents=[
                AgentSummary(
                    agent_id=target_agent_id,
                    position=(8, 8),
                    apparent_health="healthy",
                    is_carrying_items=False,
                )
            ],
            own_traits={"aggression": 0.9, "sociability": 0.3, "risk_tolerance": 0.5},
        )
        reflection = Reflection(
            last_action_succeeded=True,
            need_trends={
                "hunger": "stable",
                "thirst": "stable",
                "energy": "stable",
                "health": "stable",
            },
        )

        intention = strategy.form_intention(sensation, reflection)

        assert intention.primary_goal == "attack"
        assert intention.target_agent_id == target_agent_id

    def test_explore_when_curious_with_no_urgent_needs(self):
        """Curious agent explores when needs are satisfied."""
        strategy = PersonalityStrategy()

        sensation = Sensation(
            tick=0,
            own_needs={"hunger": 80, "thirst": 80, "energy": 80, "health": 100},
            own_position=(8, 8),
            own_inventory={},
            visible_tiles=[TileSummary(x=8, y=8, tile_type="grass", resources=[], occupants=[])],
            visible_agents=[],
            own_traits={"curiosity": 0.9, "risk_tolerance": 0.5},
        )
        reflection = Reflection(
            last_action_succeeded=True,
            need_trends={
                "hunger": "stable",
                "thirst": "stable",
                "energy": "stable",
                "health": "stable",
            },
        )

        intention = strategy.form_intention(sensation, reflection)

        assert intention.primary_goal == "explore"

    def test_express_generates_attack_action_for_attack_goal(self):
        """express() generates ATTACK action (not FIGHT) for attack goal."""
        strategy = PersonalityStrategy()

        target_agent_id = "target_456"
        sensation = Sensation(
            tick=0,
            own_needs={"hunger": 60, "thirst": 60, "energy": 70, "health": 100},
            own_position=(8, 8),
            own_inventory={},
            visible_tiles=[],
            visible_agents=[],
            own_traits={},
        )
        reflection = Reflection(
            last_action_succeeded=True,
            need_trends={
                "hunger": "stable",
                "thirst": "stable",
                "energy": "stable",
                "health": "stable",
            },
        )

        from src.awareness.types import Intention

        intention = Intention(
            primary_goal="attack",
            target_agent_id=target_agent_id,
            target_position=(8, 8),
        )

        expression = strategy.express(sensation, reflection, intention)

        assert expression.action.type == ActionType.ATTACK
        assert expression.action.target_agent_id == target_agent_id

    def test_express_generates_give_action_for_share_resources_goal(self):
        """express() generates GIVE action for share_resources goal."""
        strategy = PersonalityStrategy()

        target_agent_id = "target_789"
        sensation = Sensation(
            tick=0,
            own_needs={"hunger": 70, "thirst": 70, "energy": 70, "health": 100},
            own_position=(8, 8),
            own_inventory={"berries": 5},
            visible_tiles=[],
            visible_agents=[],
            own_traits={},
        )
        reflection = Reflection(
            last_action_succeeded=True,
            need_trends={
                "hunger": "stable",
                "thirst": "stable",
                "energy": "stable",
                "health": "stable",
            },
        )

        from src.awareness.types import Intention

        intention = Intention(
            primary_goal="share_resources",
            target_agent_id=target_agent_id,
            target_position=(9, 8),
        )

        expression = strategy.express(sensation, reflection, intention)

        assert expression.action.type == ActionType.GIVE
        assert expression.action.target_agent_id == target_agent_id
        assert expression.action.target == "berries"
        assert expression.action.quantity == 1
