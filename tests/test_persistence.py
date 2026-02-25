"""Tests for persistence: serialization, checkpointing, and migration.

Verifies:
- Round-trip serialization preserves all state
- Agent fields (needs, inventory, traits, position)
- Memory fields (episodes, relationships)
- Continuity across save/load
- Config branching
- Checkpoint lifecycle (save, load, auto, prune)
- Atomic writes
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agents.identity import AgentID, AgentProfile, PersonalityTraits
from src.config import SimulationConfig
from src.persistence.checkpoint import CheckpointManager
from src.persistence.migration import SchemaMigration
from src.persistence.serializer import StateSerializer
from src.simulation.engine import SimulationEngine


class TestStateSerializer:
    """Test StateSerializer round-trip and field preservation."""

    def test_roundtrip_empty_engine(self):
        """Serialize and deserialize an empty engine."""
        config = SimulationConfig(seed=123, world_width=64, world_height=64)
        engine1 = SimulationEngine(config)

        serializer = StateSerializer()
        data = serializer.serialize(engine1)
        engine2 = serializer.deserialize(data)

        # Verify basic state
        assert engine2.state.tick == engine1.state.tick
        assert engine2.world.width == engine1.world.width
        assert engine2.world.height == engine1.world.height
        assert engine2.world.seed == engine1.world.seed

    def test_roundtrip_with_agents(self):
        """Serialize and deserialize engine with multiple agents."""
        config = SimulationConfig(seed=456, num_agents=3, world_width=64, world_height=64)
        engine1 = SimulationEngine(config)
        engine1.setup_multi_agent()

        # Run a few ticks to generate state
        for _ in range(5):
            engine1.step_all()

        serializer = StateSerializer()
        data = serializer.serialize(engine1)
        engine2 = serializer.deserialize(data)

        # Verify agent count
        assert len(engine2.registry.living_agents()) == len(engine1.registry.living_agents())

        # Verify agent fields
        agents1 = {str(a.agent_id): a for a in engine1.registry.living_agents()}
        agents2 = {str(a.agent_id): a for a in engine2.registry.living_agents()}

        for agent_id_str in agents1:
            a1 = agents1[agent_id_str]
            a2 = agents2[agent_id_str]

            assert a2.x == a1.x
            assert a2.y == a1.y
            assert a2.needs.hunger == pytest.approx(a1.needs.hunger)
            assert a2.needs.thirst == pytest.approx(a1.needs.thirst)
            assert a2.needs.energy == pytest.approx(a1.needs.energy)
            assert a2.needs.health == pytest.approx(a1.needs.health)
            assert a2.inventory == a1.inventory
            assert a2.ticks_alive == a1.ticks_alive
            assert a2.alive == a1.alive

    def test_agent_traits_preserved(self):
        """Verify personality traits are preserved."""
        config = SimulationConfig(seed=789, world_width=64, world_height=64)
        engine1 = SimulationEngine(config)

        # Spawn agent with custom traits
        traits = PersonalityTraits(
            cooperation_tendency=0.8,
            curiosity=0.3,
            risk_tolerance=0.6,
            resource_sharing=0.9,
            aggression=0.1,
            sociability=0.7,
        )
        profile = AgentProfile(
            agent_id=AgentID(),
            name="test_agent",
            archetype="diplomat",
            traits=traits,
        )
        engine1.registry.spawn(profile, 16, 16)

        serializer = StateSerializer()
        data = serializer.serialize(engine1)
        engine2 = serializer.deserialize(data)

        agent2 = engine2.registry.living_agents()[0]
        assert agent2.profile.traits.cooperation_tendency == 0.8
        assert agent2.profile.traits.curiosity == 0.3
        assert agent2.profile.traits.risk_tolerance == 0.6
        assert agent2.profile.traits.resource_sharing == 0.9
        assert agent2.profile.traits.aggression == 0.1
        assert agent2.profile.traits.sociability == 0.7

    def test_inventory_preserved(self):
        """Verify agent inventory is preserved."""
        config = SimulationConfig(seed=111, world_width=64, world_height=64)
        engine1 = SimulationEngine(config)
        engine1.setup_legacy_mode()

        agent1 = engine1.agent
        agent1.add_item("berries", 5)
        agent1.add_item("wood", 3)

        serializer = StateSerializer()
        data = serializer.serialize(engine1)
        engine2 = serializer.deserialize(data)

        agent2 = engine2.agent
        assert agent2.inventory["berries"] == 5
        assert agent2.inventory["wood"] == 3

    def test_memory_preserved(self):
        """Verify episodic and social memories are preserved."""
        config = SimulationConfig(seed=222, num_agents=2, world_width=64, world_height=64)
        engine1 = SimulationEngine(config)
        engine1.setup_multi_agent()

        # Run ticks to generate memory
        for _ in range(10):
            engine1.step_all()

        # Get an agent's memory
        agent1 = engine1.registry.living_agents()[0]
        mem1 = engine1.registry.get_memory(agent1.agent_id)
        if mem1:
            episodic1, social1 = mem1
            episode_count1 = episodic1.episode_count
            relation_count1 = social1.relationship_count

            # Serialize and deserialize
            serializer = StateSerializer()
            data = serializer.serialize(engine1)
            engine2 = serializer.deserialize(data)

            # Verify memory counts
            agent2 = engine2.registry.get(agent1.agent_id)
            mem2 = engine2.registry.get_memory(agent2.agent_id)
            assert mem2 is not None

            episodic2, social2 = mem2
            assert episodic2.episode_count == episode_count1
            assert social2.relationship_count == relation_count1

    def test_world_tiles_preserved(self):
        """Verify world tiles and resources are preserved."""
        config = SimulationConfig(seed=333, world_width=16, world_height=16)
        engine1 = SimulationEngine(config)

        # Harvest some resources
        tile1 = engine1.world.get_tile(5, 5)
        if tile1 and tile1.resources:
            tile1.resources[0].harvest(1)
            harvested_quantity = tile1.resources[0].quantity

        serializer = StateSerializer()
        data = serializer.serialize(engine1)
        engine2 = serializer.deserialize(data)

        # Verify tile state
        tile2 = engine2.world.get_tile(5, 5)
        assert tile2 is not None
        if tile1 and tile1.resources:
            assert tile2.resources[0].quantity == harvested_quantity
            assert tile2.resources[0].kind == tile1.resources[0].kind

    def test_continuity_after_load(self):
        """Verify simulation continues correctly after load."""
        config = SimulationConfig(seed=444, num_agents=2, world_width=64, world_height=64)
        engine1 = SimulationEngine(config)
        engine1.setup_multi_agent()

        # Run 10 ticks
        for _ in range(10):
            engine1.step_all()

        tick1 = engine1.state.tick
        # Save and load
        serializer = StateSerializer()
        data = serializer.serialize(engine1)
        engine2 = serializer.deserialize(data)

        # Verify tick preserved
        assert engine2.state.tick == tick1

        # Run 5 more ticks on engine2
        for _ in range(5):
            engine2.step_all()

        # Verify tick incremented correctly
        assert engine2.state.tick == tick1 + 5

        # Verify agents still functioning
        assert len(engine2.registry.living_agents()) > 0

    def test_config_override_branching(self):
        """Verify config override creates branched simulation."""
        config = SimulationConfig(seed=555, hunger_decay=0.5, world_width=64, world_height=64)
        engine1 = SimulationEngine(config)
        engine1.setup_legacy_mode()

        serializer = StateSerializer()
        data = serializer.serialize(engine1)

        # Load with different hunger decay
        engine2 = serializer.deserialize(data, config_override={"hunger_decay": 1.0})

        assert engine2.config.hunger_decay == 1.0
        assert engine1.config.hunger_decay == 0.5

    def test_dead_agents_preserved(self):
        """Verify dead agents and death causes are preserved."""
        config = SimulationConfig(
            seed=666,
            hunger_decay=10.0,
            num_agents=2,
            world_width=64,
            world_height=64,
        )
        engine1 = SimulationEngine(config)
        engine1.setup_multi_agent()

        # Run until an agent dies
        for _ in range(50):
            engine1.step_all()
            if engine1.registry.count_dead > 0:
                break

        if engine1.registry.count_dead > 0:
            serializer = StateSerializer()
            data = serializer.serialize(engine1)
            engine2 = serializer.deserialize(data)

            assert engine2.registry.count_dead == engine1.registry.count_dead


class TestCheckpointManager:
    """Test CheckpointManager save, load, auto-checkpoint, and pruning."""

    def test_save_and_load(self, tmp_path):
        """Test basic save and load."""
        config = SimulationConfig(seed=777, world_width=64, world_height=64)
        engine1 = SimulationEngine(config)
        engine1.setup_legacy_mode()

        # Run a few ticks
        for _ in range(5):
            engine1.step_all()

        manager = CheckpointManager(checkpoint_dir=str(tmp_path))
        filepath = manager.save(engine1, label="test")

        # Verify file exists
        assert Path(filepath).exists()

        # Load and verify
        engine2 = manager.load(filepath)
        assert engine2.state.tick == engine1.state.tick

    def test_atomic_write(self, tmp_path):
        """Verify atomic write (temp file + rename)."""
        config = SimulationConfig(seed=888, world_width=64, world_height=64)
        engine = SimulationEngine(config)

        manager = CheckpointManager(checkpoint_dir=str(tmp_path))
        manager.save(engine)

        # Verify no temp files left
        temp_files = list(tmp_path.glob(".*tmp"))
        assert len(temp_files) == 0

    def test_auto_checkpoint(self, tmp_path):
        """Test auto-checkpoint at intervals."""
        config = SimulationConfig(seed=999, world_width=64, world_height=64)
        engine = SimulationEngine(config)
        engine.setup_legacy_mode()

        manager = CheckpointManager(
            checkpoint_dir=str(tmp_path), auto_interval=5, max_checkpoints=3
        )

        # Run 15 ticks — should create 3 checkpoints (at ticks 5, 10, 15)
        for _ in range(15):
            engine.step_all()
            manager.auto_checkpoint(engine)

        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) >= 2  # At least 2 checkpoints

    def test_prune_old_checkpoints(self, tmp_path):
        """Test pruning when exceeding max_checkpoints."""
        config = SimulationConfig(seed=101, world_width=64, world_height=64)
        engine = SimulationEngine(config)
        engine.setup_legacy_mode()

        manager = CheckpointManager(
            checkpoint_dir=str(tmp_path), auto_interval=2, max_checkpoints=3
        )

        # Run 10 ticks — should create 5 checkpoints, then prune to 3
        for _ in range(10):
            engine.step_all()
            manager.auto_checkpoint(engine)

        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) <= 3

    def test_list_checkpoints(self, tmp_path):
        """Test listing checkpoints with metadata."""
        config = SimulationConfig(seed=102, world_width=64, world_height=64)
        engine = SimulationEngine(config)
        engine.setup_legacy_mode()

        manager = CheckpointManager(checkpoint_dir=str(tmp_path))

        # Create a few checkpoints
        for i in range(3):
            for _ in range(5):
                engine.step_all()
            manager.save(engine, label=f"test{i}")

        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 3

        # Verify metadata
        for cp in checkpoints:
            assert "path" in cp
            assert "tick" in cp
            assert "timestamp" in cp
            assert "label" in cp

    def test_latest_checkpoint(self, tmp_path):
        """Test getting latest checkpoint."""
        config = SimulationConfig(seed=103, world_width=64, world_height=64)
        engine = SimulationEngine(config)
        engine.setup_legacy_mode()

        manager = CheckpointManager(checkpoint_dir=str(tmp_path))

        # Create checkpoints at different ticks
        engine.step_all()
        manager.save(engine, label="first")

        for _ in range(5):
            engine.step_all()
        manager.save(engine, label="second")

        latest = manager.latest_checkpoint()
        assert latest is not None
        assert "second" in latest

    def test_no_checkpoints(self, tmp_path):
        """Test behavior with no checkpoints."""
        manager = CheckpointManager(checkpoint_dir=str(tmp_path))

        checkpoints = manager.list_checkpoints()
        assert checkpoints == []

        latest = manager.latest_checkpoint()
        assert latest is None


class TestSchemaMigration:
    """Test schema migration logic."""

    def test_get_version(self):
        """Test extracting schema version."""
        data = {"schema_version": 1}
        assert SchemaMigration.get_version(data) == 1

        data_no_version = {}
        assert SchemaMigration.get_version(data_no_version) == 1

    def test_migrate_v1(self):
        """Test migration from v1 to v2 adds new sections."""
        data = {"schema_version": 1, "other": "data"}
        migrated = SchemaMigration.migrate(data)

        assert migrated["schema_version"] == 2
        assert migrated["other"] == "data"
        # New sections added by v1→v2 migration
        assert "mailboxes" in migrated
        assert "awareness_loops" in migrated
        assert "trait_evolution" in migrated
        assert "lineage" in migrated
        assert "effectiveness" in migrated


class TestIntegration:
    """Integration tests for persistence."""

    def test_full_workflow(self, tmp_path):
        """Test complete save/load/continue workflow."""
        # Setup
        config = SimulationConfig(seed=104, num_agents=3, world_width=64, world_height=64)
        engine1 = SimulationEngine(config)
        engine1.setup_multi_agent()

        # Run simulation
        for _ in range(10):
            engine1.step_all()

        # Save
        manager = CheckpointManager(checkpoint_dir=str(tmp_path))
        filepath = manager.save(engine1, label="integration")

        # Load
        engine2 = manager.load(filepath)

        # Verify state matches
        assert engine2.state.tick == engine1.state.tick
        assert len(engine2.registry.living_agents()) == len(engine1.registry.living_agents())

        # Continue simulation
        for _ in range(5):
            engine2.step_all()

        assert engine2.state.tick == engine1.state.tick + 5

    def test_branching_scenario(self, tmp_path):
        """Test branching: load same state with different configs."""
        # Setup
        config = SimulationConfig(seed=105, hunger_decay=0.5, world_width=64, world_height=64)
        engine1 = SimulationEngine(config)
        engine1.setup_legacy_mode()

        for _ in range(5):
            engine1.step_all()

        # Save
        manager = CheckpointManager(checkpoint_dir=str(tmp_path))
        filepath = manager.save(engine1, label="branch_point")

        # Branch A: normal decay
        engineA = manager.load(filepath)
        for _ in range(5):
            engineA.step_all()

        # Branch B: high decay
        engineB = manager.load(filepath, config_override={"hunger_decay": 5.0})
        for _ in range(5):
            engineB.step_all()

        # Verify branches diverged
        agentA = engineA.agent
        agentB = engineB.agent

        # B should have lower hunger due to higher decay
        assert agentB.needs.hunger < agentA.needs.hunger
