"""Tests for the SRIE awareness loop."""

from __future__ import annotations

from src.awareness.reflection import ReflectionModule
from src.awareness.sensation import SensationModule
from src.config import SimulationConfig
from src.simulation.engine import SimulationEngine
from tests.conftest import spawn_test_agent


class TestSensationModule:
    """Tests for SensationModule perception."""

    def test_perceive_returns_sensation_with_correct_own_needs(self):
        """perceive() returns Sensation with agent's current needs."""
        config = SimulationConfig(world_width=16, world_height=16, seed=42)
        engine = SimulationEngine(config)
        agent = spawn_test_agent(engine, name="test", archetype="survivalist", x=8, y=8)

        # Modify needs to known values
        agent.needs.hunger = 45.0
        agent.needs.thirst = 67.0
        agent.needs.energy = 23.0
        agent.needs.health = 88.0

        sensation_module = SensationModule()
        sensation = sensation_module.perceive(agent, engine)

        assert sensation.own_needs["hunger"] == 45.0
        assert sensation.own_needs["thirst"] == 67.0
        assert sensation.own_needs["energy"] == 23.0
        assert sensation.own_needs["health"] == 88.0

    def test_perceive_includes_visible_tiles(self):
        """perceive() includes visible tiles within vision radius."""
        config = SimulationConfig(world_width=16, world_height=16, seed=42, vision_radius=2)
        engine = SimulationEngine(config)
        agent = spawn_test_agent(engine, name="test", archetype="survivalist", x=8, y=8)

        sensation_module = SensationModule()
        sensation = sensation_module.perceive(agent, engine)

        # Should have multiple visible tiles (at least the tile agent is on)
        assert len(sensation.visible_tiles) > 0

        # Verify tiles are actually within vision radius
        for tile in sensation.visible_tiles:
            dist_sq = (tile.x - agent.x) ** 2 + (tile.y - agent.y) ** 2
            assert dist_sq <= config.vision_radius**2 + 1  # +1 for floating point tolerance

    def test_perceive_includes_visible_agents_on_tiles(self):
        """perceive() includes visible agents on tiles."""
        config = SimulationConfig(world_width=16, world_height=16, seed=42, vision_radius=3)
        engine = SimulationEngine(config)
        agent1 = spawn_test_agent(engine, name="agent1", archetype="survivalist", x=8, y=8)
        agent2 = spawn_test_agent(engine, name="agent2", archetype="explorer", x=9, y=8)

        sensation_module = SensationModule()
        sensation = sensation_module.perceive(agent1, engine)

        # agent1 should see agent2
        assert len(sensation.visible_agents) > 0
        visible_ids = [a.agent_id for a in sensation.visible_agents]
        assert agent2.agent_id in visible_ids

    def test_perceive_drains_mailbox_messages(self):
        """perceive() drains incoming messages from mailbox."""
        config = SimulationConfig(world_width=16, world_height=16, seed=42)
        engine = SimulationEngine(config)
        agent = spawn_test_agent(engine, name="test", archetype="survivalist", x=8, y=8)

        # Send a message to the agent
        from src.agents.identity import AgentID
        from src.communication.protocol import MessageType, create_message

        sender = AgentID("sender_123")
        msg = create_message(
            tick=0,
            sender_id=sender,
            receiver_id=agent.agent_id,
            message_type=MessageType.INFORM,
            content="Test message",
        )
        mailbox = engine.message_bus.get_or_create_mailbox(agent.agent_id)
        mailbox.receive(msg)

        sensation_module = SensationModule()
        sensation = sensation_module.perceive(agent, engine)

        # Should have received the message
        assert len(sensation.incoming_messages) == 1
        assert sensation.incoming_messages[0].content == "Test message"

        # Mailbox should be drained
        assert len(mailbox.drain_inbox()) == 0

    def test_perceive_includes_own_traits_from_profile(self):
        """perceive() includes agent's personality traits."""
        config = SimulationConfig(world_width=16, world_height=16, seed=42)
        engine = SimulationEngine(config)

        # Use diplomat archetype which has known traits
        agent = spawn_test_agent(engine, name="test", archetype="diplomat", x=8, y=8)

        sensation_module = SensationModule()
        sensation = sensation_module.perceive(agent, engine)

        # Diplomat has high cooperation_tendency (0.9)
        assert sensation.own_traits["cooperation_tendency"] == 0.9
        assert sensation.own_traits["curiosity"] == 0.4
        assert sensation.own_traits["aggression"] == 0.05
        assert sensation.own_traits["sociability"] == 0.9


class TestReflectionModule:
    """Tests for ReflectionModule evaluation."""

    def test_evaluate_returns_reflection(self):
        """evaluate() returns a Reflection object."""
        config = SimulationConfig(world_width=16, world_height=16, seed=42)
        engine = SimulationEngine(config)
        agent = spawn_test_agent(engine, name="test", archetype="survivalist", x=8, y=8)

        sensation_module = SensationModule()
        sensation = sensation_module.perceive(agent, engine)

        reflection_module = ReflectionModule()
        reflection = reflection_module.evaluate(agent, engine, sensation)

        assert reflection is not None
        assert hasattr(reflection, "last_action_succeeded")
        assert hasattr(reflection, "need_trends")

    def test_evaluate_last_action_succeeded_is_true_with_no_history(self):
        """evaluate() assumes success when no history exists."""
        config = SimulationConfig(world_width=16, world_height=16, seed=42)
        engine = SimulationEngine(config)
        agent = spawn_test_agent(engine, name="test", archetype="survivalist", x=8, y=8)

        # Engine has no history yet
        assert len(engine.state.history) == 0

        sensation_module = SensationModule()
        sensation = sensation_module.perceive(agent, engine)

        reflection_module = ReflectionModule()
        reflection = reflection_module.evaluate(agent, engine, sensation)

        assert reflection.last_action_succeeded is True

    def test_evaluate_computes_need_trends_all_stable_initially(self):
        """evaluate() returns 'stable' trends when insufficient history."""
        config = SimulationConfig(world_width=16, world_height=16, seed=42)
        engine = SimulationEngine(config)
        agent = spawn_test_agent(engine, name="test", archetype="survivalist", x=8, y=8)

        sensation_module = SensationModule()
        sensation = sensation_module.perceive(agent, engine)

        reflection_module = ReflectionModule()
        reflection = reflection_module.evaluate(agent, engine, sensation)

        assert "hunger" in reflection.need_trends
        assert "thirst" in reflection.need_trends
        assert "energy" in reflection.need_trends
        assert "health" in reflection.need_trends

        # All should be stable initially
        assert reflection.need_trends["hunger"] == "stable"
        assert reflection.need_trends["thirst"] == "stable"
        assert reflection.need_trends["energy"] == "stable"
        assert reflection.need_trends["health"] == "stable"

    def test_evaluate_assesses_threat_level(self):
        """evaluate() computes threat level from 0-1."""
        config = SimulationConfig(world_width=16, world_height=16, seed=42)
        engine = SimulationEngine(config)
        agent = spawn_test_agent(engine, name="test", archetype="survivalist", x=8, y=8)

        sensation_module = SensationModule()
        sensation = sensation_module.perceive(agent, engine)

        reflection_module = ReflectionModule()
        reflection = reflection_module.evaluate(agent, engine, sensation)

        assert 0.0 <= reflection.threat_level <= 1.0

    def test_evaluate_assesses_opportunity_from_resources(self):
        """evaluate() computes opportunity score from nearby resources."""
        config = SimulationConfig(world_width=16, world_height=16, seed=42)
        engine = SimulationEngine(config)
        agent = spawn_test_agent(engine, name="test", archetype="survivalist", x=8, y=8)

        sensation_module = SensationModule()
        sensation = sensation_module.perceive(agent, engine)

        reflection_module = ReflectionModule()
        reflection = reflection_module.evaluate(agent, engine, sensation)

        assert 0.0 <= reflection.opportunity_score <= 1.0


class TestAwarenessLoop:
    """Tests for AwarenessLoop SRIE pipeline."""

    def test_tick_runs_full_srie_pipeline(self):
        """tick() runs all four SRIE stages."""
        config = SimulationConfig(world_width=16, world_height=16, seed=42)
        engine = SimulationEngine(config)
        agent = spawn_test_agent(engine, name="test", archetype="survivalist", x=8, y=8)

        # Get the awareness loop registered by spawn_test_agent
        awareness_loop = engine.registry.get_awareness_loop(agent.agent_id)

        # Run one tick
        expression = awareness_loop.tick(engine)

        # Should return an Expression
        assert expression is not None
        assert hasattr(expression, "action")

    def test_tick_returns_expression_with_action(self):
        """tick() returns Expression containing an action."""
        config = SimulationConfig(world_width=16, world_height=16, seed=42)
        engine = SimulationEngine(config)
        agent = spawn_test_agent(engine, name="test", archetype="survivalist", x=8, y=8)

        awareness_loop = engine.registry.get_awareness_loop(agent.agent_id)
        expression = awareness_loop.tick(engine)

        assert expression.action is not None
        from src.simulation.actions import ActionType

        assert isinstance(expression.action.type, ActionType)

    def test_tick_fills_sender_id_on_messages(self):
        """tick() populates sender_id on outgoing messages."""
        config = SimulationConfig(world_width=16, world_height=16, seed=42)
        engine = SimulationEngine(config)

        # Create a very social agent to encourage message sending
        from src.agents.identity import PersonalityTraits

        social_traits = PersonalityTraits(
            sociability=0.95,
            cooperation_tendency=0.9,
        )
        agent1 = spawn_test_agent(
            engine, name="social", archetype="diplomat", x=8, y=8, traits=social_traits
        )
        _agent2 = spawn_test_agent(engine, name="target", archetype="explorer", x=9, y=8)

        awareness_loop = engine.registry.get_awareness_loop(agent1.agent_id)

        # Run multiple ticks to increase chance of message generation
        expression = None
        for _ in range(10):
            expression = awareness_loop.tick(engine)
            if expression.message is not None:
                break

        # If a message was generated, verify sender_id is filled
        if expression and expression.message is not None:
            assert expression.message.sender_id == agent1.agent_id

    def test_tick_stores_last_sensation(self):
        """tick() stores last_sensation for debugging."""
        config = SimulationConfig(world_width=16, world_height=16, seed=42)
        engine = SimulationEngine(config)
        agent = spawn_test_agent(engine, name="test", archetype="survivalist", x=8, y=8)

        awareness_loop = engine.registry.get_awareness_loop(agent.agent_id)
        awareness_loop.tick(engine)

        assert awareness_loop.last_sensation is not None
        assert hasattr(awareness_loop.last_sensation, "own_needs")

    def test_tick_stores_last_reflection(self):
        """tick() stores last_reflection for debugging."""
        config = SimulationConfig(world_width=16, world_height=16, seed=42)
        engine = SimulationEngine(config)
        agent = spawn_test_agent(engine, name="test", archetype="survivalist", x=8, y=8)

        awareness_loop = engine.registry.get_awareness_loop(agent.agent_id)
        awareness_loop.tick(engine)

        assert awareness_loop.last_reflection is not None
        assert hasattr(awareness_loop.last_reflection, "need_trends")

    def test_tick_stores_last_intention(self):
        """tick() stores last_intention for debugging."""
        config = SimulationConfig(world_width=16, world_height=16, seed=42)
        engine = SimulationEngine(config)
        agent = spawn_test_agent(engine, name="test", archetype="survivalist", x=8, y=8)

        awareness_loop = engine.registry.get_awareness_loop(agent.agent_id)
        awareness_loop.tick(engine)

        assert awareness_loop.last_intention is not None
        assert hasattr(awareness_loop.last_intention, "primary_goal")
