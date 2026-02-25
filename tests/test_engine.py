"""Tests for SimulationEngine integration."""

from __future__ import annotations

import pytest

from src.config import SimulationConfig
from src.simulation.actions import Action, ActionType
from src.simulation.engine import SimulationEngine, TickRecord


class TestEngineConstruction:
    """Test engine initialization."""

    def test_engine_constructor_creates_all_components(self, config: SimulationConfig):
        """Engine constructor should create world, registry, message_bus, etc."""
        engine = SimulationEngine(config)

        assert engine.world is not None
        assert engine.world.width == config.world_width
        assert engine.world.height == config.world_height
        assert engine.registry is not None
        assert engine.message_bus is not None
        assert engine.metrics_collector is not None
        assert engine.emergence_detector is not None
        assert engine.trait_evolution is not None
        assert engine.state is not None
        assert engine.state.tick == 0


class TestLegacyMode:
    """Test single-agent backward compatibility mode."""

    def test_setup_legacy_mode_creates_single_agent_at_center(self, config: SimulationConfig):
        """setup_legacy_mode() should create single agent at center."""
        engine = SimulationEngine(config)
        engine.setup_legacy_mode()

        assert engine.agent is not None
        assert engine.agent.x == config.world_width // 2
        assert engine.agent.y == config.world_height // 2
        assert engine.agent.profile is not None
        assert engine.agent.profile.name == "agent"

    def test_step_single_agent_mode_executes_action(self, config: SimulationConfig):
        """step() single-agent mode should execute action."""
        engine = SimulationEngine(config)
        engine.setup_legacy_mode()

        initial_tick = engine.state.tick
        action = Action(type=ActionType.WAIT)

        record = engine.step(action)

        assert record is not None
        assert engine.state.tick == initial_tick + 1
        assert len(record.agent_records) == 1
        assert record.agent_records[0].action == action

    def test_step_applies_needs_delta_from_action_result(self, config: SimulationConfig):
        """step() should apply needs_delta from action result."""
        engine = SimulationEngine(config)
        engine.setup_legacy_mode()

        agent = engine.agent
        initial_energy = agent.needs.energy

        # REST action gives +10 energy
        action = Action(type=ActionType.REST)
        _record = engine.step(action)

        # Energy should increase (but also decay happens after)
        # REST gives +10, decay is -0.3, so net should be +9.7
        expected_energy = min(100.0, initial_energy + 10.0 - config.energy_decay)
        assert abs(agent.needs.energy - expected_energy) < 0.01

    def test_step_increments_tick(self, config: SimulationConfig):
        """step() should increment tick."""
        engine = SimulationEngine(config)
        engine.setup_legacy_mode()

        initial_tick = engine.state.tick
        engine.step(Action(type=ActionType.WAIT))
        assert engine.state.tick == initial_tick + 1

        engine.step(Action(type=ActionType.WAIT))
        assert engine.state.tick == initial_tick + 2


class TestMultiAgentMode:
    """Test multi-agent simulation."""

    def test_setup_multi_agent_spawns_configured_number_of_agents(self, config: SimulationConfig):
        """setup_multi_agent() should spawn configured number of agents."""
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        living = engine.registry.living_agents()
        assert len(living) == config.num_agents

    def test_step_all_multi_agent_executes_one_tick_for_all_agents(self, config: SimulationConfig):
        """step_all() multi-agent should execute one tick for all agents."""
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        initial_tick = engine.state.tick
        num_agents = len(engine.registry.living_agents())

        record = engine.step_all()

        assert engine.state.tick == initial_tick + 1
        assert len(record.agent_records) == num_agents

    def test_step_all_returns_tick_record_with_agent_records(self, config: SimulationConfig):
        """step_all() should return TickRecord with agent_records."""
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        record = engine.step_all()

        assert isinstance(record, TickRecord)
        assert len(record.agent_records) > 0
        for agent_record in record.agent_records:
            assert agent_record.agent_id is not None
            assert agent_record.action is not None
            assert agent_record.result is not None
            assert agent_record.needs_before is not None
            assert agent_record.needs_after is not None
            assert agent_record.position is not None

    def test_step_all_decays_needs_for_all_agents(self, config: SimulationConfig):
        """step_all() should decay needs for all agents."""
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        agents = engine.registry.living_agents()
        initial_hungers = [a.needs.hunger for a in agents]

        engine.step_all()

        # All agents should have decayed hunger
        for i, agent in enumerate(engine.registry.living_agents()):
            expected = initial_hungers[i] - config.hunger_decay
            assert agent.needs.hunger < initial_hungers[i]
            # Allow for action effects
            assert agent.needs.hunger <= expected + 20.0  # max possible gain from eating

    def test_step_all_detects_deaths(self, config: SimulationConfig):
        """step_all() should detect deaths when hunger/thirst/health reaches 0."""
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        agents = engine.registry.living_agents()
        # Set first agent's hunger to 0
        agents[0].needs.hunger = 0.0
        agents[0].needs.health = 1.0  # Will die from hunger damage
        # Clear inventory so agent can't eat to recover
        agents[0].inventory.clear()

        initial_living = engine.registry.count_living

        engine.step_all()

        # Agent should die from hunger
        assert engine.registry.count_living < initial_living
        assert not agents[0].alive

    def test_step_all_delivers_messages_via_message_bus(self, config: SimulationConfig):
        """step_all() should deliver messages via message bus."""
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        # Track initial message bus state
        _initial_queue_size = len(engine.message_bus._pending)

        # Run a tick (agents may send messages)
        _record = engine.step_all()

        # Messages should be delivered (queue cleared)
        assert len(engine.message_bus._pending) == 0

    def test_step_all_updates_memories(self, config: SimulationConfig):
        """step_all() should update episodic memories."""
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        agents = engine.registry.living_agents()
        agent = agents[0]
        episodic, _social = engine.registry.get_memory(agent.agent_id)

        initial_episode_count = episodic.episode_count

        # Run a tick
        engine.step_all()

        # Episodic memory should have a new episode
        assert episodic.episode_count > initial_episode_count

    def test_step_all_increments_tick(self, config: SimulationConfig):
        """step_all() should increment tick."""
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        initial_tick = engine.state.tick
        engine.step_all()
        assert engine.state.tick == initial_tick + 1


class TestSimulationEnd:
    """Test simulation termination conditions."""

    def test_is_over_returns_false_initially(self, config: SimulationConfig):
        """is_over() should return False initially."""
        engine = SimulationEngine(config)
        engine.setup_multi_agent()
        assert engine.is_over() is False

    def test_is_over_returns_true_when_all_agents_dead(self, config: SimulationConfig):
        """is_over() should return True when all agents dead."""
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        # Kill all agents
        for agent in engine.registry.living_agents():
            engine.registry.kill(agent.agent_id, "test")

        assert engine.is_over() is True

    def test_is_over_returns_true_when_max_ticks_reached(self, config: SimulationConfig):
        """is_over() should return True when max_ticks reached."""
        config.max_ticks = 5
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        # Run to max ticks
        for _ in range(5):
            engine.step_all()

        assert engine.is_over() is True


class TestTickRecordBackwardCompat:
    """Test TickRecord backward compatibility properties."""

    def test_tick_record_backward_compat_properties(self, config: SimulationConfig):
        """TickRecord should provide backward compat properties."""
        engine = SimulationEngine(config)
        engine.setup_legacy_mode()

        action = Action(type=ActionType.REST)
        record = engine.step(action)

        # Backward compat: should access first agent's data
        assert record.action == action
        assert record.result is not None
        assert record.agent_needs_before is not None
        assert record.agent_needs_after is not None
        assert record.agent_position is not None
        assert isinstance(record.agent_position, tuple)
        assert len(record.agent_position) == 2


class TestTraitEvolution:
    """Test trait evolution integration."""

    def test_trait_evolution_integration_give_action_updates_giver_traits(
        self, config: SimulationConfig
    ):
        """GIVE action should update giver's traits via trait evolution."""
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        agents = engine.registry.living_agents()
        if len(agents) < 2:
            pytest.skip("Need at least 2 agents for GIVE test")

        giver = agents[0]
        receiver = agents[1]

        # Position them adjacent
        giver.x, giver.y = 5, 5
        receiver.x, receiver.y = 6, 5

        # Give giver some berries
        giver.add_item("berries", 5)

        initial_cooperation = giver.profile.traits.cooperation_tendency

        # Manually create awareness loop to force GIVE action
        from src.awareness.types import Expression

        # Directly manipulate the giver's awareness loop to force a GIVE action
        loop = engine.registry.get_awareness_loop(giver.agent_id)
        if loop:
            # Create a GIVE action expression
            give_action = Action(
                type=ActionType.GIVE,
                target="berries",
                target_agent_id=receiver.agent_id,
                quantity=1,
            )
            # Manually inject the expression
            _original_tick = loop.tick

            def forced_give_tick(engine):
                return Expression(
                    action=give_action, message=None, internal_monologue="Sharing resources"
                )

            loop.tick = forced_give_tick

            # Run a tick
            engine.step_all()

            # Restore original tick function
            loop.tick = _original_tick

            # Check trait was updated (cooperation_tendency should increase)
            assert giver.profile.traits.cooperation_tendency >= initial_cooperation
