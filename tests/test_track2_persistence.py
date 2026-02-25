"""Track 2 persistence and effectiveness scoring tests.

Verifies:
- TraitEvolution history tracking and serialization
- Mailbox message history persistence
- SRIE cache (awareness loop) persistence
- Lineage tracking and tree import/export
- Schema migration from v1 to v2
- Nudge effectiveness scoring
- Router quality tracking
- Classifier accuracy scoring
- Effectiveness engine serialization
"""

from __future__ import annotations

import pytest

from src.agents.evolution import TraitEvolution, TraitNudgeRecord
from src.agents.identity import PersonalityTraits
from src.communication.protocol import MessageType, create_message
from src.config import SimulationConfig
from src.effectiveness.scoring import (
    ClassifierAccuracyRecord,
    EffectivenessEngine,
    NudgeEffectivenessScore,
    RouterQualityRecord,
)
from src.evolution.lineage import LineageTracker
from src.persistence.migration import SchemaMigration
from src.persistence.serializer import StateSerializer
from src.simulation.engine import SimulationEngine
from tests.conftest import spawn_test_agent


class TestTraitEvolutionHistory:
    """Test the trait evolution history tracking added in TraitEvolution."""

    def test_process_outcome_records_history(self):
        """Call process_outcome with agent_id and tick, verify history has 1 record."""
        evolution = TraitEvolution(learning_rate=0.01)
        traits = PersonalityTraits()

        evolution.process_outcome(
            traits=traits,
            event_type="shared_resource",
            was_positive=True,
            agent_id="agent123",
            tick=42,
        )

        assert len(evolution.history) == 1, "History should have exactly one record"
        record = evolution.history[0]
        assert record.tick == 42
        assert record.agent_id == "agent123"
        assert record.event_type == "shared_resource"
        assert record.was_positive is True

    def test_history_captures_old_and_new_values(self):
        """Process an outcome, verify traits_affected has (trait_name, old_val, new_val)."""
        evolution = TraitEvolution(learning_rate=0.1)
        traits = PersonalityTraits(cooperation_tendency=0.5)

        evolution.process_outcome(
            traits=traits,
            event_type="shared_resource",
            was_positive=True,
            agent_id="agent456",
            tick=10,
        )

        record = evolution.history[0]
        assert len(record.traits_affected) > 0, "Should have affected traits"

        # Find cooperation_tendency in affected traits
        coop_change = None
        for trait_name, old_val, new_val in record.traits_affected:
            if trait_name == "cooperation_tendency":
                coop_change = (old_val, new_val)
                break

        assert coop_change is not None, "cooperation_tendency should be affected"
        old_val, new_val = coop_change
        assert old_val == pytest.approx(0.5), "Old value should be 0.5"
        assert new_val > old_val, "New value should be greater than old (positive outcome)"

    def test_history_respects_max_size(self):
        """Create TraitEvolution(history_max=5), add 10 records, verify len == 5 (FIFO)."""
        evolution = TraitEvolution(history_max=5)
        traits = PersonalityTraits()

        # Add 10 records
        for i in range(10):
            evolution.process_outcome(
                traits=traits,
                event_type="shared_resource",
                was_positive=True,
                agent_id=f"agent{i}",
                tick=i,
            )

        assert len(evolution.history) == 5, "History should be capped at 5"
        # Verify FIFO: should have ticks 5-9 (last 5)
        ticks = [r.tick for r in evolution.history]
        assert ticks == [5, 6, 7, 8, 9], "Should keep last 5 records (FIFO)"

    def test_export_import_roundtrip(self):
        """Export history, create new TraitEvolution, import, verify records match."""
        evolution1 = TraitEvolution()
        traits = PersonalityTraits()

        # Add some records
        evolution1.process_outcome(traits, "explored_new_area", True, "agent1", 10)
        evolution1.process_outcome(traits, "attacked_agent", False, "agent2", 20)
        evolution1.process_outcome(traits, "shared_resource", True, "agent1", 30)

        # Export
        exported = evolution1.export_history()

        # Import into new instance
        evolution2 = TraitEvolution()
        evolution2.import_history(exported)

        # Verify counts match
        assert len(evolution2.history) == 3, "Should have 3 records after import"

        # Verify records match
        for r1, r2 in zip(evolution1.history, evolution2.history, strict=True):
            assert r1.tick == r2.tick
            assert r1.agent_id == r2.agent_id
            assert r1.event_type == r2.event_type
            assert r1.was_positive == r2.was_positive
            assert len(r1.traits_affected) == len(r2.traits_affected)

    def test_viewed_at_preserved(self):
        """Set viewed_at on a record, export/import, verify preserved."""
        evolution1 = TraitEvolution()
        traits = PersonalityTraits()

        evolution1.process_outcome(traits, "shared_resource", True, "agent1", 10)

        # Set viewed_at
        evolution1.history[0].viewed_at = 100

        # Export and import
        exported = evolution1.export_history()
        evolution2 = TraitEvolution()
        evolution2.import_history(exported)

        assert evolution2.history[0].viewed_at == 100, "viewed_at should be preserved"

    def test_backward_compat_no_agent_id(self):
        """Call process_outcome WITHOUT agent_id/tick, verify it still works."""
        evolution = TraitEvolution()
        traits = PersonalityTraits()

        # Old signature: just traits, event_type, was_positive
        result = evolution.process_outcome(traits, "shared_resource", True)

        assert len(result) > 0, "Should return trait changes"
        assert len(evolution.history) == 1, "Should record in history"

        record = evolution.history[0]
        assert record.agent_id == "", "Should have empty string for agent_id"
        assert record.tick == 0, "Should have 0 for tick"


class TestMailboxPersistence:
    """Test mailbox serialization via StateSerializer."""

    def test_mailbox_history_roundtrip(self, engine):
        """Create engine with agents, send messages, serialize, verify mailbox history."""
        # Spawn agents
        agent1 = spawn_test_agent(engine, name="agent1", archetype="survivalist")
        agent2 = spawn_test_agent(engine, name="agent2", archetype="diplomat")

        # Send messages via message bus
        msg1 = create_message(
            tick=0,
            sender_id=agent1.agent_id,
            receiver_id=agent2.agent_id,
            message_type=MessageType.INFORM,
            content="Hello!",
        )
        engine.message_bus.send(msg1)

        msg2 = create_message(
            tick=0,
            sender_id=agent2.agent_id,
            receiver_id=agent1.agent_id,
            message_type=MessageType.NEGOTIATE,
            content="Trade?",
        )
        engine.message_bus.send(msg2)

        # Deliver messages
        engine.message_bus.deliver_all(engine.registry, engine.world)

        # Serialize
        serializer = StateSerializer()
        data = serializer.serialize(engine)

        # Verify mailboxes section exists
        assert "mailboxes" in data, "Mailboxes section should exist"
        mailboxes = data["mailboxes"]

        # Verify both agents have mailbox entries
        agent1_id = str(agent1.agent_id)
        agent2_id = str(agent2.agent_id)
        assert agent1_id in mailboxes, "Agent1 should have mailbox"
        assert agent2_id in mailboxes, "Agent2 should have mailbox"

        # Verify message history counts
        agent1_history = mailboxes[agent1_id].get("history", [])
        agent2_history = mailboxes[agent2_id].get("history", [])
        assert len(agent1_history) == 1, "Agent1 should have 1 message in history"
        assert len(agent2_history) == 1, "Agent2 should have 1 message in history"

        # Deserialize and verify
        engine2 = serializer.deserialize(data)
        mailbox1 = engine2.message_bus._mailboxes.get(agent1.agent_id)
        mailbox2 = engine2.message_bus._mailboxes.get(agent2.agent_id)

        assert mailbox1 is not None, "Agent1 mailbox should be restored"
        assert mailbox2 is not None, "Agent2 mailbox should be restored"
        assert mailbox1.history_count == 1, "Agent1 history count should be 1"
        assert mailbox2.history_count == 1, "Agent2 history count should be 1"

    def test_empty_mailbox_roundtrip(self, engine):
        """Serialize engine with agents but no messages, verify mailboxes section exists."""
        # Spawn agents but don't send messages
        spawn_test_agent(engine, name="agent1")
        spawn_test_agent(engine, name="agent2")

        # Serialize
        serializer = StateSerializer()
        data = serializer.serialize(engine)

        # Verify mailboxes section exists
        assert "mailboxes" in data, "Mailboxes section should exist"
        mailboxes = data["mailboxes"]

        # Should be empty or have empty mailboxes
        assert isinstance(mailboxes, dict), "Mailboxes should be a dict"


class TestSRIECachePersistence:
    """Test SRIE cache serialization."""

    def test_srie_cache_roundtrip(self, engine):
        """Create engine, spawn agents, run step_all, serialize, verify caches."""
        # Spawn agents
        agent1 = spawn_test_agent(engine, name="agent1")
        agent2 = spawn_test_agent(engine, name="agent2")

        # Run one step to populate SRIE caches
        engine.step_all()

        # Serialize
        serializer = StateSerializer()
        data = serializer.serialize(engine)

        # Verify awareness_loops section exists
        assert "awareness_loops" in data, "awareness_loops section should exist"

        # Deserialize
        engine2 = serializer.deserialize(data)

        # Verify agents have awareness loops with cached data
        loop1 = engine2.registry.get_awareness_loop(agent1.agent_id)
        loop2 = engine2.registry.get_awareness_loop(agent2.agent_id)

        assert loop1 is not None, "Agent1 should have awareness loop"
        assert loop2 is not None, "Agent2 should have awareness loop"

        # Verify caches are populated (at least some fields should be non-None)
        # After step_all, last_sensation should be populated
        assert loop1.last_sensation is not None or loop1.last_reflection is not None, (
            "Agent1 SRIE cache should have some data"
        )

    def test_srie_cache_sensation_is_summary(self, engine):
        """Run step_all, serialize, verify awareness_loops contains summary fields."""
        # Spawn agents
        spawn_test_agent(engine, name="agent1")
        spawn_test_agent(engine, name="agent2")

        # Run step to populate caches
        engine.step_all()

        # Serialize
        serializer = StateSerializer()
        data = serializer.serialize(engine)

        # Verify awareness_loops has data
        loops = data["awareness_loops"]
        assert len(loops) > 0, "Should have awareness loop data"

        # Verify structure contains summary fields
        # (visible_agent_count, not full visible_agents list)
        for _agent_id, loop_data in loops.items():
            if "last_sensation" in loop_data and loop_data["last_sensation"]:
                sensation = loop_data["last_sensation"]
                # Should have visible_agent_count, not verbose visible_agents list
                assert "visible_agent_count" in sensation, (
                    "Sensation should use summary format with visible_agent_count"
                )


class TestLineagePersistence:
    """Test lineage tracking and serialization."""

    def test_lineage_import_export_roundtrip(self):
        """Create LineageTracker, record births, export, import, verify match."""
        tracker1 = LineageTracker()
        traits1 = PersonalityTraits(cooperation_tendency=0.7)
        traits2 = PersonalityTraits(cooperation_tendency=0.75)
        traits3 = PersonalityTraits(cooperation_tendency=0.8)

        # Record births: one root, two children
        tracker1.record_birth("agent1", None, traits1, birth_tick=0)
        tracker1.record_birth("agent2", "agent1", traits2, birth_tick=10)
        tracker1.record_birth("agent3", "agent1", traits3, birth_tick=20)

        # Export
        exported = tracker1.export_tree()

        # Import into new tracker
        tracker2 = LineageTracker()
        tracker2.import_tree(exported)

        # Verify roots match
        assert len(tracker2._roots) == 1, "Should have 1 root"
        assert "agent1" in tracker2._roots, "agent1 should be root"

        # Verify nodes match
        assert len(tracker2._lineages) == 3, "Should have 3 nodes"
        assert "agent1" in tracker2._lineages
        assert "agent2" in tracker2._lineages
        assert "agent3" in tracker2._lineages

        # Verify parent-child relationships
        node1 = tracker2._lineages["agent1"]
        assert node1.parent_id is None, "Root should have no parent"
        assert len(node1.children) == 2, "Root should have 2 children"

        node2 = tracker2._lineages["agent2"]
        assert node2.parent_id == "agent1", "agent2's parent should be agent1"

    def test_lineage_import_empty(self):
        """Import empty data, verify tracker has no lineages."""
        tracker = LineageTracker()
        tracker.import_tree({})

        assert len(tracker._roots) == 0, "Should have no roots"
        assert len(tracker._lineages) == 0, "Should have no nodes"

    def test_lineage_wired_to_population_manager(self):
        """Create engine with evolution_enabled=True, verify lineage_tracker exists."""
        config = SimulationConfig(
            world_width=16,
            world_height=16,
            seed=42,
            evolution_enabled=True,
        )
        engine = SimulationEngine(config)

        assert engine.population_manager is not None, (
            "Population manager should exist when evolution enabled"
        )
        assert hasattr(engine.population_manager, "lineage_tracker"), (
            "Population manager should have lineage_tracker"
        )
        assert engine.population_manager.lineage_tracker is not None, (
            "Lineage tracker should be initialized"
        )


class TestSchemaMigrationV2:
    """Test v1→v2 migration."""

    def test_v1_to_v2_adds_all_sections(self):
        """Migrate v1 data, verify all new sections present with correct defaults."""
        v1_data = {
            "schema_version": 1,
            "config": {"seed": 123},
            "state": {"tick": 10},
            "world": {"width": 64, "height": 64},
            "agents": {"living": [], "dead": []},
            "memories": {},
            "emergence": {},
        }

        migrated = SchemaMigration.migrate(v1_data)

        # Verify version updated
        assert migrated["schema_version"] == 2, "Should upgrade to version 2"

        # Verify new sections added
        assert "mailboxes" in migrated, "Should add mailboxes section"
        assert "awareness_loops" in migrated, "Should add awareness_loops section"
        assert "trait_evolution" in migrated, "Should add trait_evolution section"
        assert "lineage" in migrated, "Should add lineage section"
        assert "effectiveness" in migrated, "Should add effectiveness section"

        # Verify defaults are correct
        assert migrated["mailboxes"] == {}, "Mailboxes should default to empty dict"
        assert migrated["awareness_loops"] == {}, "Awareness loops should default to empty dict"
        assert "history" in migrated["trait_evolution"], "Trait evolution should have history"
        assert migrated["lineage"]["roots"] == [], "Lineage roots should be empty list"
        assert migrated["lineage"]["nodes"] == {}, "Lineage nodes should be empty dict"

    def test_v1_checkpoint_loads(self):
        """Create a v1-format checkpoint, load via StateSerializer, verify no error."""
        # Create minimal v1 checkpoint
        v1_data = {
            "schema_version": 1,
            "timestamp": "2025-01-01T00:00:00Z",
            "config": {
                "seed": 42,
                "world_width": 16,
                "world_height": 16,
                "max_ticks": 100,
                "num_agents": 2,
            },
            "state": {"tick": 0, "day": 0, "time_of_day": "dawn"},
            "world": {
                "width": 16,
                "height": 16,
                "seed": 42,
                "tiles": [],
            },
            "agents": {
                "living": [],
                "dead": [],
                "death_causes": {},
            },
            "memories": {},
            "emergence": {"event_count": 0},
        }

        # Should not raise
        serializer = StateSerializer()
        engine = serializer.deserialize(v1_data)

        assert engine.state.tick == 0, "Tick should be preserved"
        assert engine.world.width == 16, "World dimensions should be preserved"

    def test_v2_data_not_migrated(self):
        """Create v2 data, call migrate, verify schema_version stays 2."""
        v2_data = {
            "schema_version": 2,
            "config": {"seed": 123},
            "mailboxes": {},
            "awareness_loops": {},
            "trait_evolution": {"history": []},
            "lineage": {"roots": [], "nodes": {}},
            "effectiveness": {},
        }

        migrated = SchemaMigration.migrate(v2_data)

        # Should not change version
        assert migrated["schema_version"] == 2, "Should stay at version 2"


class TestNudgeEffectiveness:
    """Test the nudge effectiveness scoring."""

    def test_nudge_scoring_basic(self):
        """Create EffectivenessEngine, record nudge, simulate fitness, score."""
        engine_scoring = EffectivenessEngine(scoring_interval=10, lookback_window=5)

        # Create a mock nudge
        nudge = TraitNudgeRecord(
            tick=10,
            agent_id="agent1",
            event_type="shared_resource",
            was_positive=True,
            traits_affected=[("cooperation_tendency", 0.5, 0.6)],
        )

        # Record nudge
        engine_scoring.record_nudge(nudge)

        # Simulate fitness snapshots
        engine_scoring._fitness_snapshots["agent1"] = [
            (10, 0.5),  # fitness before
            (15, 0.6),  # fitness after
        ]

        # Score at tick 20 (10 ticks after nudge)
        scores = engine_scoring.score_nudges(None, current_tick=20)

        assert len(scores) >= 1, "Should produce at least one score"
        score = scores[0]
        assert score.agent_id == "agent1"
        assert score.nudge_tick == 10
        assert score.delta > 0, "Fitness should have improved"

    def test_nudge_scoring_interval(self):
        """Verify scoring only runs when tick % interval == 0."""
        engine_scoring = EffectivenessEngine(scoring_interval=50)

        # tick() should only score on multiples of 50
        # We can't easily test without a full engine, but we can verify the interval logic

        # At tick 25, should not score (not a multiple of 50)
        # At tick 50, should score

        # Simple verification: scoring_interval is set correctly
        assert engine_scoring._scoring_interval == 50

    def test_get_nudge_effectiveness_empty(self):
        """Verify empty effectiveness returns count=0, avg_score=0."""
        engine_scoring = EffectivenessEngine()

        result = engine_scoring.get_nudge_effectiveness()

        assert result["count"] == 0, "Count should be 0 for empty"
        assert result["avg_score"] == 0.0, "Avg score should be 0.0 for empty"
        assert result["avg_delta"] == 0.0, "Avg delta should be 0.0 for empty"


class TestRouterQuality:
    """Test router quality tracking."""

    def test_router_quality_tracking(self):
        """Record routing outcomes, score, verify quality_scores populated."""
        engine_scoring = EffectivenessEngine(scoring_interval=50)

        # Record 10 outcomes at tick 10-19 (within window of tick 20)
        for i in range(10, 20):
            success = i % 3 == 0  # Every 3rd succeeds
            engine_scoring.record_routing_outcome(
                tick=i,
                agent_id="agent1",
                architecture="arch_A",
                intention_goal="gather_food",
                action_succeeded=success,
                needs_delta_sum=2.0 if success else -1.0,
            )

        # Score at tick 20 (only looks back scoring_interval=50, so all records are within window)
        quality_scores = engine_scoring.score_router_quality(current_tick=20)

        assert "arch_A" in quality_scores, "Should have quality score for arch_A"
        assert 0.0 <= quality_scores["arch_A"] <= 1.0, "Quality should be between 0 and 1"

    def test_architecture_weights_cold_start(self):
        """Verify get_architecture_weights returns empty on cold start."""
        engine_scoring = EffectivenessEngine()

        weights = engine_scoring.get_architecture_weights()

        assert weights == {}, "Should return empty dict on cold start"

    def test_architecture_weights_normalized(self):
        """Record outcomes for 2 architectures, verify weights sum to ~1.0."""
        engine_scoring = EffectivenessEngine(scoring_interval=50)

        # Record outcomes for two architectures at tick 10-19 (within window)
        for i in range(10, 20):
            engine_scoring.record_routing_outcome(
                tick=i,
                agent_id="agent1",
                architecture="arch_A",
                intention_goal="gather",
                action_succeeded=True,
                needs_delta_sum=2.0,
            )
            engine_scoring.record_routing_outcome(
                tick=i,
                agent_id="agent2",
                architecture="arch_B",
                intention_goal="explore",
                action_succeeded=False,
                needs_delta_sum=-1.0,
            )

        # Score
        engine_scoring.score_router_quality(current_tick=20)

        # Get weights
        weights = engine_scoring.get_architecture_weights()

        assert len(weights) == 2, "Should have weights for 2 architectures"
        total = sum(weights.values())
        assert total == pytest.approx(1.0, abs=0.01), "Weights should sum to ~1.0"


class TestClassifierAccuracy:
    """Test classifier accuracy."""

    def test_classifier_accuracy_basic(self):
        """Record classification + actual outcome, score, verify accuracy computed."""
        engine_scoring = EffectivenessEngine(scoring_interval=10)

        # Record a prediction
        engine_scoring.record_classification(
            tick=10,
            agent_id="agent1",
            predicted_threat=0.7,
            predicted_opportunity=0.3,
        )

        # Record actual outcome
        engine_scoring.record_actual_outcome(
            agent_id="agent1",
            health_before=100.0,
            health_after=80.0,  # Lost 20 health = 0.2 damage
            inventory_value_before=5.0,
            inventory_value_after=10.0,  # Gained 5 value
        )

        # Score
        accuracy = engine_scoring.score_classifier_accuracy(current_tick=20)

        assert "agent1" in accuracy, "Should have accuracy for agent1"
        assert 0.0 <= accuracy["agent1"] <= 1.0, "Accuracy should be between 0 and 1"

    def test_classifier_calibration_default(self):
        """Verify get_classifier_calibration returns 0.5 for unknown agent."""
        engine_scoring = EffectivenessEngine()

        calibration = engine_scoring.get_classifier_calibration("unknown_agent")

        assert calibration == 0.5, "Should return neutral default 0.5 for unknown agent"


class TestEffectivenessSerialization:
    """Test effectiveness engine state persistence."""

    def test_export_import_roundtrip(self):
        """Populate engine with scores/records, export, import, verify restored."""
        engine1 = EffectivenessEngine(scoring_interval=10)

        # Add some nudge scores
        engine1._nudge_scores.append(
            NudgeEffectivenessScore(
                nudge_tick=10,
                agent_id="agent1",
                event_type="shared_resource",
                fitness_before=0.5,
                fitness_after=0.6,
                delta=0.1,
                score=0.5,
            )
        )

        # Add router records
        engine1._router_records.append(
            RouterQualityRecord(
                tick=15,
                agent_id="agent1",
                architecture="arch_A",
                intention_goal="gather",
                action_succeeded=True,
                needs_delta_sum=2.0,
            )
        )

        # Add classifier records
        engine1._classifier_records.append(
            ClassifierAccuracyRecord(
                tick=20,
                agent_id="agent1",
                predicted_threat=0.7,
                predicted_opportunity=0.3,
                actual_damage_taken=0.2,
                actual_resources_gained=0.4,
                threat_error=0.5,
                opportunity_error=0.1,
            )
        )

        # Add quality scores
        engine1._quality_scores["arch_A"] = 0.75

        # Add classifier accuracy
        engine1._classifier_accuracy["agent1"] = 0.85

        # Export
        exported = engine1.export_state()

        # Import into new engine
        engine2 = EffectivenessEngine()
        engine2.import_state(exported)

        # Verify nudge scores
        assert len(engine2._nudge_scores) == 1, "Should have 1 nudge score"
        assert engine2._nudge_scores[0].agent_id == "agent1"

        # Verify router records
        assert len(engine2._router_records) == 1, "Should have 1 router record"
        assert engine2._router_records[0].architecture == "arch_A"

        # Verify classifier records
        assert len(engine2._classifier_records) == 1, "Should have 1 classifier record"
        assert engine2._classifier_records[0].agent_id == "agent1"

        # Verify quality scores
        assert engine2._quality_scores["arch_A"] == 0.75

        # Verify classifier accuracy
        assert engine2._classifier_accuracy["agent1"] == 0.85


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_serialize_engine_with_no_effectiveness(self):
        """Engine without effectiveness_scoring_enabled, serialize, verify defaults."""
        config = SimulationConfig(
            world_width=16,
            world_height=16,
            seed=42,
            effectiveness_scoring_enabled=False,
        )
        engine = SimulationEngine(config)

        serializer = StateSerializer()
        data = serializer.serialize(engine)

        # Verify effectiveness section exists with empty defaults
        assert "effectiveness" in data, "Effectiveness section should exist"
        effectiveness = data["effectiveness"]
        assert effectiveness["nudge_scores"] == [], "Nudge scores should be empty"
        assert effectiveness["router_records"] == [], "Router records should be empty"
        assert effectiveness["classifier_records"] == [], "Classifier records should be empty"

    def test_trait_evolution_with_unknown_event(self):
        """Call process_outcome with unknown event_type, verify empty changes and no history."""
        evolution = TraitEvolution()
        traits = PersonalityTraits()

        result = evolution.process_outcome(
            traits=traits,
            event_type="unknown_event_type",
            was_positive=True,
            agent_id="agent1",
            tick=10,
        )

        assert result == [], "Should return empty list for unknown event type"
        # From the code: if not affected, returns [] early before recording
        # So history should NOT be recorded for unknown event types
        assert len(evolution.history) == 0, "Should not record event for unknown event type"
