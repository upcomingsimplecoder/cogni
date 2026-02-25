"""Tests for cognitive architecture variants and pluggable SRIE components."""

from __future__ import annotations

import pytest

from src.awareness.architecture import (
    CognitiveArchitecture,
    build_awareness_loop,
    get_architectures,
)
from src.awareness.deliberation import (
    ConsensusDeliberation,
    NullDeliberation,
    ThresholdDeliberation,
)
from src.awareness.protocols import DeliberationStrategy, EvaluationStrategy, PerceptionStrategy
from src.awareness.reflection import ReflectionModule
from src.awareness.reflection_variants import (
    OptimisticReflection,
    PessimisticReflection,
    SocialReflection,
)
from src.awareness.sensation import SensationModule
from src.awareness.types import Reflection
from src.config import SimulationConfig
from src.simulation.engine import SimulationEngine
from tests.conftest import spawn_test_agent


class TestProtocols:
    """Tests for protocol compliance."""

    def test_sensation_module_implements_perception_protocol(self):
        """SensationModule satisfies PerceptionStrategy protocol."""
        module = SensationModule()
        assert isinstance(module, PerceptionStrategy)

    def test_reflection_module_implements_evaluation_protocol(self):
        """ReflectionModule satisfies EvaluationStrategy protocol."""
        module = ReflectionModule()
        assert isinstance(module, EvaluationStrategy)

    def test_null_deliberation_implements_protocol(self):
        """NullDeliberation satisfies DeliberationStrategy protocol."""
        delib = NullDeliberation()
        assert isinstance(delib, DeliberationStrategy)


class TestOptimisticReflection:
    """Tests for OptimisticReflection evaluation variant."""

    def test_reduces_threat_level(self):
        """OptimisticReflection reduces threat_level by 40%."""
        config = SimulationConfig(world_width=16, world_height=16, seed=42)
        engine = SimulationEngine(config)
        agent = spawn_test_agent(engine, name="test", archetype="survivalist", x=8, y=8)

        # Set up a high-threat scenario (low health)
        agent.needs.health = 20.0

        sensation_module = SensationModule()
        sensation = sensation_module.perceive(agent, engine)

        # Get base reflection
        base_module = ReflectionModule()
        base_reflection = base_module.evaluate(agent, engine, sensation)

        # Get optimistic reflection
        optimistic = OptimisticReflection()
        opt_reflection = optimistic.evaluate(agent, engine, sensation)

        # Should have lower threat (0.6x multiplier)
        assert opt_reflection.threat_level <= base_reflection.threat_level
        assert opt_reflection.threat_level == pytest.approx(
            base_reflection.threat_level * 0.6, abs=0.01
        )

    def test_increases_opportunity_score(self):
        """OptimisticReflection increases opportunity_score by 40%."""
        config = SimulationConfig(world_width=16, world_height=16, seed=42)
        engine = SimulationEngine(config)
        agent = spawn_test_agent(engine, name="test", archetype="survivalist", x=8, y=8)

        sensation_module = SensationModule()
        sensation = sensation_module.perceive(agent, engine)

        base_module = ReflectionModule()
        base_reflection = base_module.evaluate(agent, engine, sensation)

        optimistic = OptimisticReflection()
        opt_reflection = optimistic.evaluate(agent, engine, sensation)

        # Should have higher opportunity (1.4x multiplier, capped at 1.0)
        assert opt_reflection.opportunity_score >= base_reflection.opportunity_score
        expected = min(1.0, base_reflection.opportunity_score * 1.4)
        assert opt_reflection.opportunity_score == pytest.approx(expected, abs=0.01)

    def test_respects_bounds(self):
        """OptimisticReflection keeps threat and opportunity in [0, 1]."""
        config = SimulationConfig(world_width=16, world_height=16, seed=42)
        engine = SimulationEngine(config)
        agent = spawn_test_agent(engine, name="test", archetype="survivalist", x=8, y=8)

        sensation_module = SensationModule()
        sensation = sensation_module.perceive(agent, engine)

        optimistic = OptimisticReflection()
        reflection = optimistic.evaluate(agent, engine, sensation)

        assert 0.0 <= reflection.threat_level <= 1.0
        assert 0.0 <= reflection.opportunity_score <= 1.0


class TestPessimisticReflection:
    """Tests for PessimisticReflection evaluation variant."""

    def test_increases_threat_level(self):
        """PessimisticReflection increases threat_level by 50%."""
        config = SimulationConfig(world_width=16, world_height=16, seed=42)
        engine = SimulationEngine(config)
        agent = spawn_test_agent(engine, name="test", archetype="survivalist", x=8, y=8)

        # Set up some threat
        agent.needs.health = 70.0

        sensation_module = SensationModule()
        sensation = sensation_module.perceive(agent, engine)

        base_module = ReflectionModule()
        base_reflection = base_module.evaluate(agent, engine, sensation)

        pessimistic = PessimisticReflection()
        pess_reflection = pessimistic.evaluate(agent, engine, sensation)

        # Should have higher threat (1.5x multiplier, capped at 1.0)
        assert pess_reflection.threat_level >= base_reflection.threat_level
        expected = min(1.0, base_reflection.threat_level * 1.5)
        assert pess_reflection.threat_level == pytest.approx(expected, abs=0.01)

    def test_reduces_opportunity_score(self):
        """PessimisticReflection reduces opportunity_score by 40%."""
        config = SimulationConfig(world_width=16, world_height=16, seed=42)
        engine = SimulationEngine(config)
        agent = spawn_test_agent(engine, name="test", archetype="survivalist", x=8, y=8)

        sensation_module = SensationModule()
        sensation = sensation_module.perceive(agent, engine)

        base_module = ReflectionModule()
        base_reflection = base_module.evaluate(agent, engine, sensation)

        pessimistic = PessimisticReflection()
        pess_reflection = pessimistic.evaluate(agent, engine, sensation)

        # Should have lower opportunity (0.6x multiplier)
        assert pess_reflection.opportunity_score <= base_reflection.opportunity_score
        assert pess_reflection.opportunity_score == pytest.approx(
            base_reflection.opportunity_score * 0.6, abs=0.01
        )


class TestSocialReflection:
    """Tests for SocialReflection evaluation variant."""

    def test_responds_to_visible_agents(self):
        """SocialReflection considers visible agents in evaluation."""
        config = SimulationConfig(world_width=16, world_height=16, seed=42, vision_radius=3)
        engine = SimulationEngine(config)
        agent1 = spawn_test_agent(engine, name="agent1", archetype="diplomat", x=8, y=8)
        _agent2 = spawn_test_agent(engine, name="agent2", archetype="explorer", x=9, y=8)

        sensation_module = SensationModule()
        sensation = sensation_module.perceive(agent1, engine)

        # Should see agent2
        assert len(sensation.visible_agents) > 0

        social = SocialReflection()
        reflection = social.evaluate(agent1, engine, sensation)

        # Should produce valid reflection
        assert 0.0 <= reflection.threat_level <= 1.0
        assert 0.0 <= reflection.opportunity_score <= 1.0


class TestNullDeliberation:
    """Tests for NullDeliberation (no System 2)."""

    def test_never_escalates(self):
        """NullDeliberation never triggers System 2."""
        config = SimulationConfig(world_width=16, world_height=16, seed=42)
        engine = SimulationEngine(config)
        agent = spawn_test_agent(engine, name="test", archetype="survivalist", x=8, y=8)

        sensation_module = SensationModule()
        sensation = sensation_module.perceive(agent, engine)

        reflection_module = ReflectionModule()
        reflection = reflection_module.evaluate(agent, engine, sensation)

        delib = NullDeliberation()
        assert delib.should_escalate(sensation, reflection) is False

    def test_returns_reflection_unchanged(self):
        """NullDeliberation returns input reflection as-is."""
        config = SimulationConfig(world_width=16, world_height=16, seed=42)
        engine = SimulationEngine(config)
        agent = spawn_test_agent(engine, name="test", archetype="survivalist", x=8, y=8)

        sensation_module = SensationModule()
        sensation = sensation_module.perceive(agent, engine)

        reflection_module = ReflectionModule()
        original = reflection_module.evaluate(agent, engine, sensation)

        delib = NullDeliberation()
        result = delib.deliberate(agent, sensation, original)

        assert result is original


class TestThresholdDeliberation:
    """Tests for ThresholdDeliberation (System 1/2 dual-process)."""

    def test_escalates_on_high_threat(self):
        """ThresholdDeliberation escalates when threat > threshold."""
        config = SimulationConfig(world_width=16, world_height=16, seed=42)
        engine = SimulationEngine(config)
        agent = spawn_test_agent(engine, name="test", archetype="survivalist", x=8, y=8)

        # Create high threat
        agent.needs.health = 10.0

        sensation_module = SensationModule()
        sensation = sensation_module.perceive(agent, engine)

        reflection_module = ReflectionModule()
        reflection = reflection_module.evaluate(agent, engine, sensation)

        # Default threshold is 0.7
        delib = ThresholdDeliberation(threat_threshold=0.5)

        # Should escalate if threat > 0.5
        if reflection.threat_level > 0.5:
            assert delib.should_escalate(sensation, reflection) is True

    def test_escalates_on_multiple_critical_needs(self):
        """ThresholdDeliberation escalates when 2+ needs are critical."""
        config = SimulationConfig(world_width=16, world_height=16, seed=42)
        engine = SimulationEngine(config)
        agent = spawn_test_agent(engine, name="test", archetype="survivalist", x=8, y=8)

        # Set multiple needs to critical
        agent.needs.hunger = 15.0
        agent.needs.thirst = 18.0

        sensation_module = SensationModule()
        sensation = sensation_module.perceive(agent, engine)

        reflection_module = ReflectionModule()
        reflection = reflection_module.evaluate(agent, engine, sensation)

        delib = ThresholdDeliberation(critical_need_threshold=20.0)
        assert delib.should_escalate(sensation, reflection) is True

    def test_deliberate_returns_valid_reflection(self):
        """ThresholdDeliberation.deliberate() returns valid Reflection."""
        config = SimulationConfig(world_width=16, world_height=16, seed=42)
        engine = SimulationEngine(config)
        agent = spawn_test_agent(engine, name="test", archetype="survivalist", x=8, y=8)

        sensation_module = SensationModule()
        sensation = sensation_module.perceive(agent, engine)

        reflection_module = ReflectionModule()
        reflection = reflection_module.evaluate(agent, engine, sensation)

        delib = ThresholdDeliberation()
        refined = delib.deliberate(agent, sensation, reflection)

        assert isinstance(refined, Reflection)
        assert 0.0 <= refined.threat_level <= 1.0
        assert 0.0 <= refined.opportunity_score <= 1.0


class TestConsensusDeliberation:
    """Tests for ConsensusDeliberation (social reasoning)."""

    def test_escalates_when_agents_visible(self):
        """ConsensusDeliberation escalates when other agents are visible."""
        config = SimulationConfig(world_width=16, world_height=16, seed=42, vision_radius=3)
        engine = SimulationEngine(config)
        agent1 = spawn_test_agent(engine, name="agent1", archetype="diplomat", x=8, y=8)
        _agent2 = spawn_test_agent(engine, name="agent2", archetype="explorer", x=9, y=8)

        sensation_module = SensationModule()
        sensation = sensation_module.perceive(agent1, engine)

        reflection_module = ReflectionModule()
        reflection = reflection_module.evaluate(agent1, engine, sensation)

        delib = ConsensusDeliberation()
        # Should escalate if agents visible
        if len(sensation.visible_agents) > 0:
            assert delib.should_escalate(sensation, reflection) is True

    def test_deliberate_considers_agent_health(self):
        """ConsensusDeliberation adjusts threat based on visible agent health."""
        config = SimulationConfig(world_width=16, world_height=16, seed=42, vision_radius=3)
        engine = SimulationEngine(config)
        agent1 = spawn_test_agent(engine, name="agent1", archetype="diplomat", x=8, y=8)
        agent2 = spawn_test_agent(engine, name="agent2", archetype="explorer", x=9, y=8)

        # Make agent2 injured
        agent2.needs.health = 20.0

        sensation_module = SensationModule()
        sensation = sensation_module.perceive(agent1, engine)

        reflection_module = ReflectionModule()
        reflection = reflection_module.evaluate(agent1, engine, sensation)

        delib = ConsensusDeliberation()
        refined = delib.deliberate(agent1, sensation, reflection)

        assert isinstance(refined, Reflection)
        assert 0.0 <= refined.threat_level <= 1.0


class TestArchitectureRegistry:
    """Tests for architecture registry and builder."""

    def test_registry_has_expected_architectures(self):
        """Architecture registry contains all predefined architectures."""
        architectures = get_architectures()

        assert "reactive" in architectures
        assert "cautious" in architectures
        assert "optimistic" in architectures
        assert "social" in architectures
        assert "dual_process" in architectures

    def test_all_architectures_valid(self):
        """All registered architectures have required components."""
        architectures = get_architectures()

        for name, arch in architectures.items():
            assert isinstance(arch, CognitiveArchitecture)
            assert arch.name == name
            assert arch.perception is not None
            assert arch.evaluation is not None
            assert arch.strategy_factory is not None
            assert isinstance(arch.description, str)

    def test_build_awareness_loop_creates_valid_loop(self):
        """build_awareness_loop() creates a functioning AwarenessLoop."""
        config = SimulationConfig(world_width=16, world_height=16, seed=42)
        engine = SimulationEngine(config)
        agent = spawn_test_agent(engine, name="test", archetype="survivalist", x=8, y=8)

        loop = build_awareness_loop(agent, "reactive")

        assert loop is not None
        assert loop.agent == agent
        assert loop.strategy is not None

    def test_build_awareness_loop_with_invalid_name_raises(self):
        """build_awareness_loop() raises ValueError for unknown architecture."""
        config = SimulationConfig(world_width=16, world_height=16, seed=42)
        engine = SimulationEngine(config)
        agent = spawn_test_agent(engine, name="test", archetype="survivalist", x=8, y=8)

        with pytest.raises(ValueError, match="Unknown architecture"):
            build_awareness_loop(agent, "nonexistent_architecture")

    def test_reactive_architecture_has_no_deliberation(self):
        """Reactive architecture has deliberation=None (System 1 only)."""
        architectures = get_architectures()
        reactive = architectures["reactive"]

        assert reactive.deliberation is None

    def test_cautious_architecture_has_pessimistic_evaluation(self):
        """Cautious architecture uses PessimisticReflection."""
        architectures = get_architectures()
        cautious = architectures["cautious"]

        assert isinstance(cautious.evaluation, PessimisticReflection)

    def test_dual_process_architecture_has_deliberation(self):
        """Dual-process architecture includes ThresholdDeliberation."""
        architectures = get_architectures()
        dual = architectures["dual_process"]

        assert dual.deliberation is not None
        assert isinstance(dual.deliberation, ThresholdDeliberation)


class TestAwarenessLoopWithArchitecture:
    """Integration tests for AwarenessLoop with different architectures."""

    def test_reactive_loop_runs_without_deliberation(self):
        """Reactive architecture runs without deliberation step."""
        config = SimulationConfig(world_width=16, world_height=16, seed=42)
        engine = SimulationEngine(config)
        agent = spawn_test_agent(engine, name="test", archetype="survivalist", x=8, y=8)

        loop = build_awareness_loop(agent, "reactive")
        expression = loop.tick(engine)

        assert expression is not None
        assert expression.action is not None

    def test_dual_process_loop_can_escalate(self):
        """Dual-process architecture can escalate to System 2."""
        config = SimulationConfig(world_width=16, world_height=16, seed=42)
        engine = SimulationEngine(config)
        agent = spawn_test_agent(engine, name="test", archetype="survivalist", x=8, y=8)

        # Create high-threat scenario
        agent.needs.health = 10.0
        agent.needs.hunger = 15.0

        loop = build_awareness_loop(agent, "dual_process")
        expression = loop.tick(engine)

        # Should complete successfully even with deliberation
        assert expression is not None
        assert expression.action is not None

    def test_cautious_agent_has_lower_opportunity_score(self):
        """Cautious architecture produces lower opportunity scores."""
        config = SimulationConfig(world_width=16, world_height=16, seed=42)
        engine = SimulationEngine(config)

        # Create two agents in similar conditions
        agent_reactive = spawn_test_agent(
            engine, name="reactive", archetype="survivalist", x=8, y=8
        )
        agent_cautious = spawn_test_agent(
            engine, name="cautious", archetype="survivalist", x=9, y=8
        )

        loop_reactive = build_awareness_loop(agent_reactive, "reactive")
        loop_cautious = build_awareness_loop(agent_cautious, "cautious")

        # Run one tick each
        loop_reactive.tick(engine)
        loop_cautious.tick(engine)

        # Cautious should have lower opportunity (pessimistic evaluation)
        assert (
            loop_cautious.last_reflection.opportunity_score
            <= loop_reactive.last_reflection.opportunity_score
        )

    def test_optimistic_agent_has_lower_threat_level(self):
        """Optimistic architecture produces lower threat levels."""
        config = SimulationConfig(world_width=16, world_height=16, seed=42)
        engine = SimulationEngine(config)

        agent_reactive = spawn_test_agent(
            engine, name="reactive", archetype="survivalist", x=8, y=8
        )
        agent_optimistic = spawn_test_agent(
            engine, name="optimistic", archetype="survivalist", x=9, y=8
        )

        # Create threat
        agent_reactive.needs.health = 30.0
        agent_optimistic.needs.health = 30.0

        loop_reactive = build_awareness_loop(agent_reactive, "reactive")
        loop_optimistic = build_awareness_loop(agent_optimistic, "optimistic")

        loop_reactive.tick(engine)
        loop_optimistic.tick(engine)

        # Optimistic should have lower threat
        assert (
            loop_optimistic.last_reflection.threat_level
            <= loop_reactive.last_reflection.threat_level
        )


class TestBackwardCompatibility:
    """Tests ensuring existing code still works."""

    def test_old_awareness_loop_constructor_works(self):
        """Old AwarenessLoop(agent, strategy, sensation_module, reflection_module) still works."""
        from src.awareness.loop import AwarenessLoop
        from src.cognition.strategies.personality import PersonalityStrategy

        config = SimulationConfig(world_width=16, world_height=16, seed=42)
        engine = SimulationEngine(config)
        agent = spawn_test_agent(engine, name="test", archetype="survivalist", x=8, y=8)

        # Old-style construction
        loop = AwarenessLoop(
            agent=agent,
            strategy=PersonalityStrategy(),
            sensation_module=SensationModule(),
            reflection_module=ReflectionModule(),
        )

        expression = loop.tick(engine)
        assert expression is not None
        assert expression.action is not None

    def test_spawn_archetype_without_architecture_param_works(self):
        """spawn_archetype() without architecture param works (backward compatible)."""
        config = SimulationConfig(world_width=16, world_height=16, seed=42)
        engine = SimulationEngine(config)

        # Old-style call (no architecture param)
        agent = spawn_test_agent(engine, name="test", archetype="survivalist", x=8, y=8)

        assert agent is not None
        loop = engine.registry.get_awareness_loop(agent.agent_id)
        assert loop is not None

        # Should still work
        expression = loop.tick(engine)
        assert expression is not None
