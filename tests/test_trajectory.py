"""Tests for trajectory recording, loading, and exporting.

Verifies:
- Schema: AgentSnapshot, EmergenceSnapshot, RunMetadata, TrajectoryDataset
- Recorder: start_run, record_tick, end_run, file creation
- Loader: from_jsonl, filters, action_distribution
- Exporter: to_csv, to_jsonl, to_agent_summary
- Round-trip: record → write → load → verify identical
"""

from __future__ import annotations

import csv
import json

from src.config import SimulationConfig
from src.simulation.engine import SimulationEngine
from src.trajectory.exporter import TrajectoryExporter
from src.trajectory.loader import TrajectoryLoader
from src.trajectory.recorder import TrajectoryRecorder
from src.trajectory.schema import (
    AgentSnapshot,
    EmergenceSnapshot,
    RunMetadata,
    TrajectoryDataset,
)


class TestAgentSnapshot:
    """Test AgentSnapshot schema."""

    def test_agent_snapshot_creation_with_all_fields(self):
        """AgentSnapshot should accept all defined fields."""
        snapshot = AgentSnapshot(
            tick=5,
            agent_id="agent-001",
            agent_name="Alice",
            archetype="gatherer",
            position=(10, 15),
            alive=True,
            hunger=75.0,
            thirst=60.0,
            energy=85.0,
            health=95.0,
            traits={"cooperation_tendency": 0.8, "curiosity": 0.6},
            sensation_summary={"visible_agent_count": 2, "resource_count": 3},
            reflection={"threat_level": 0.3, "opportunity_score": 0.7},
            intention={"primary_goal": "gather_food", "confidence": 0.9},
            action_type="gather",
            action_target="berries",
            action_target_agent=None,
            action_succeeded=True,
            needs_delta={"energy": -0.5, "hunger": -1.0},
            inventory={"berries": 5, "water": 2},
            messages_sent=[{"type": "info", "receiver": "agent-002", "content": "hello"}],
            messages_received=[],
            internal_monologue="I should gather more food",
            trait_changes=[],
        )

        assert snapshot.tick == 5
        assert snapshot.agent_id == "agent-001"
        assert snapshot.agent_name == "Alice"
        assert snapshot.archetype == "gatherer"
        assert snapshot.position == (10, 15)
        assert snapshot.alive is True
        assert snapshot.hunger == 75.0
        assert snapshot.thirst == 60.0
        assert snapshot.energy == 85.0
        assert snapshot.health == 95.0
        assert snapshot.traits["cooperation_tendency"] == 0.8
        assert snapshot.action_type == "gather"
        assert snapshot.action_succeeded is True
        assert snapshot.inventory["berries"] == 5

    def test_agent_snapshot_serializes_to_dict(self):
        """AgentSnapshot.to_dict() should produce valid dictionary."""
        snapshot = AgentSnapshot(
            tick=0,
            agent_id="test-id",
            agent_name="test",
            archetype="survivalist",
            position=(0, 0),
            alive=True,
            hunger=50.0,
            thirst=50.0,
            energy=50.0,
            health=50.0,
            traits={},
            sensation_summary={},
            reflection={},
            intention={},
            action_type="wait",
            action_target=None,
            action_target_agent=None,
            action_succeeded=True,
            needs_delta={},
            inventory={},
        )

        data = snapshot.to_dict()

        assert isinstance(data, dict)
        assert data["tick"] == 0
        assert data["agent_id"] == "test-id"
        assert data["agent_name"] == "test"
        assert data["alive"] is True


class TestEmergenceSnapshot:
    """Test EmergenceSnapshot schema."""

    def test_emergence_snapshot_creation(self):
        """EmergenceSnapshot should capture emergence event data."""
        event = EmergenceSnapshot(
            tick=10,
            pattern_type="clustering",
            agents_involved=["agent-001", "agent-002", "agent-003"],
            description="Agents clustered near water source",
            data={"cluster_size": 3, "cluster_center": (15, 20)},
        )

        assert event.tick == 10
        assert event.pattern_type == "clustering"
        assert len(event.agents_involved) == 3
        assert "water" in event.description
        assert event.data["cluster_size"] == 3

    def test_emergence_snapshot_serializes_to_dict(self):
        """EmergenceSnapshot.to_dict() should produce valid dictionary."""
        event = EmergenceSnapshot(
            tick=5,
            pattern_type="cooperation",
            agents_involved=["a1", "a2"],
            description="test event",
        )

        data = event.to_dict()

        assert isinstance(data, dict)
        assert data["tick"] == 5
        assert data["pattern_type"] == "cooperation"


class TestRunMetadata:
    """Test RunMetadata schema."""

    def test_run_metadata_captures_config(self):
        """RunMetadata should store simulation configuration."""
        metadata = RunMetadata(
            run_id="test-run-123",
            timestamp="2026-02-22T10:00:00Z",
            seed=42,
            config={"world_width": 32, "world_height": 32, "max_ticks": 100},
            num_agents=5,
            max_ticks=100,
            actual_ticks=0,
            agents=[
                {"id": "agent-001", "name": "Alice", "archetype": "gatherer"},
                {"id": "agent-002", "name": "Bob", "archetype": "explorer"},
            ],
            architecture="default",
            final_state={},
        )

        assert metadata.run_id == "test-run-123"
        assert metadata.seed == 42
        assert metadata.num_agents == 5
        assert metadata.max_ticks == 100
        assert len(metadata.agents) == 2
        assert metadata.config["world_width"] == 32

    def test_run_metadata_serializes_to_dict(self):
        """RunMetadata.to_dict() should produce valid dictionary."""
        metadata = RunMetadata(
            run_id="test",
            timestamp="2026-01-01T00:00:00Z",
            seed=1,
            config={},
            num_agents=1,
            max_ticks=10,
        )

        data = metadata.to_dict()

        assert isinstance(data, dict)
        assert data["run_id"] == "test"
        assert data["seed"] == 1


class TestTrajectoryDataset:
    """Test TrajectoryDataset schema."""

    def test_trajectory_dataset_holds_snapshots_and_events(self):
        """TrajectoryDataset should hold metadata, snapshots, and events."""
        metadata = RunMetadata(
            run_id="test",
            timestamp="2026-01-01T00:00:00Z",
            seed=42,
            config={},
            num_agents=1,
            max_ticks=10,
        )

        snapshot = AgentSnapshot(
            tick=0,
            agent_id="agent-001",
            agent_name="test",
            archetype="survivalist",
            position=(0, 0),
            alive=True,
            hunger=50.0,
            thirst=50.0,
            energy=50.0,
            health=50.0,
            traits={},
            sensation_summary={},
            reflection={},
            intention={},
            action_type="wait",
            action_target=None,
            action_target_agent=None,
            action_succeeded=True,
            needs_delta={},
            inventory={},
        )

        event = EmergenceSnapshot(
            tick=0,
            pattern_type="test",
            agents_involved=["agent-001"],
            description="test event",
        )

        dataset = TrajectoryDataset(
            metadata=metadata,
            agent_snapshots=[snapshot],
            emergence_events=[event],
        )

        assert dataset.metadata.run_id == "test"
        assert len(dataset.agent_snapshots) == 1
        assert len(dataset.emergence_events) == 1
        assert dataset.agent_snapshots[0].agent_id == "agent-001"

    def test_trajectory_dataset_to_dict_serializes_nested_objects(self):
        """TrajectoryDataset.to_dict() should serialize all nested objects."""
        metadata = RunMetadata(
            run_id="test",
            timestamp="2026-01-01T00:00:00Z",
            seed=1,
            config={},
            num_agents=0,
            max_ticks=0,
        )
        dataset = TrajectoryDataset(metadata=metadata)

        data = dataset.to_dict()

        assert isinstance(data, dict)
        assert "metadata" in data
        assert "agent_snapshots" in data
        assert "emergence_events" in data


class TestTrajectoryRecorder:
    """Test TrajectoryRecorder."""

    def test_start_run_creates_output_directory(self, config: SimulationConfig, tmp_path):
        """start_run() should create output directory."""
        recorder = TrajectoryRecorder(output_dir=str(tmp_path / "trajectories"))
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        recorder.start_run(engine)

        assert recorder.output_dir.exists()
        assert recorder.output_dir.is_dir()

    def test_start_run_creates_jsonl_file(self, config: SimulationConfig, tmp_path):
        """start_run() should create trajectory.jsonl file."""
        recorder = TrajectoryRecorder(output_dir=str(tmp_path / "trajectories"))
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        recorder.start_run(engine)

        jsonl_path = recorder.output_dir / "trajectory.jsonl"
        assert jsonl_path.exists()
        assert jsonl_path.is_file()

    def test_record_tick_writes_one_line_per_agent(self, config: SimulationConfig, tmp_path):
        """record_tick() should write one snapshot per agent."""
        config.num_agents = 3
        recorder = TrajectoryRecorder(output_dir=str(tmp_path / "trajectories"))
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        recorder.start_run(engine)
        tick_record = engine.step_all()
        recorder.record_tick(engine, tick_record)
        recorder.end_run(engine)

        # Read JSONL and count agent_snapshot lines
        jsonl_path = recorder.output_dir / "trajectory.jsonl"
        with open(jsonl_path) as f:
            lines = f.readlines()

        agent_snapshot_lines = [
            line
            for line in lines
            if '"type": "agent_snapshot"' in line or '"type":"agent_snapshot"' in line
        ]
        assert len(agent_snapshot_lines) == 3

    def test_record_tick_captures_srie_state(self, config: SimulationConfig, tmp_path):
        """record_tick() should capture sensation/reflection/intention state."""
        recorder = TrajectoryRecorder(output_dir=str(tmp_path / "trajectories"))
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        recorder.start_run(engine)
        tick_record = engine.step_all()
        recorder.record_tick(engine, tick_record)
        recorder.end_run(engine)

        # Read snapshots from JSONL
        jsonl_path = recorder.output_dir / "trajectory.jsonl"
        with open(jsonl_path) as f:
            for line in f:
                data = json.loads(line)
                if data.get("type") == "agent_snapshot":
                    # Verify SRIE fields exist
                    assert "sensation_summary" in data
                    assert "reflection" in data
                    assert "intention" in data
                    assert isinstance(data["sensation_summary"], dict)
                    assert isinstance(data["reflection"], dict)
                    assert isinstance(data["intention"], dict)
                    break

    def test_end_run_writes_final_metadata(self, config: SimulationConfig, tmp_path):
        """end_run() should write run_complete record with final stats."""
        recorder = TrajectoryRecorder(output_dir=str(tmp_path / "trajectories"))
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        recorder.start_run(engine)
        for _ in range(5):
            tick_record = engine.step_all()
            recorder.record_tick(engine, tick_record)
        recorder.end_run(engine)

        # Read last line
        jsonl_path = recorder.output_dir / "trajectory.jsonl"
        with open(jsonl_path) as f:
            lines = f.readlines()

        last_line = json.loads(lines[-1])
        assert last_line["type"] == "run_complete"
        assert last_line["actual_ticks"] == 5

    def test_end_run_closes_file(self, config: SimulationConfig, tmp_path):
        """end_run() should close JSONL file handle."""
        recorder = TrajectoryRecorder(output_dir=str(tmp_path / "trajectories"))
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        recorder.start_run(engine)
        recorder.end_run(engine)

        # File should be closed (verify by reading it normally)
        jsonl_path = recorder.output_dir / "trajectory.jsonl"
        with open(jsonl_path) as f:
            content = f.read()
        assert content  # Should not raise


class TestTrajectoryLoader:
    """Test TrajectoryLoader."""

    def test_loader_round_trip_jsonl(self, config: SimulationConfig, tmp_path):
        """Record → write → load → verify identical data."""
        config.num_agents = 2
        recorder = TrajectoryRecorder(output_dir=str(tmp_path / "trajectories"))
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        # Record 10 ticks
        recorder.start_run(engine)
        for _ in range(10):
            tick_record = engine.step_all()
            recorder.record_tick(engine, tick_record)
        recorder.end_run(engine)

        # Load back
        jsonl_path = recorder.output_dir / "trajectory.jsonl"
        dataset = TrajectoryLoader.from_jsonl(str(jsonl_path))

        # Verify metadata
        assert dataset.metadata.seed == config.seed
        assert dataset.metadata.num_agents == 2
        assert dataset.metadata.actual_ticks == 10

        # Verify snapshots (2 agents × 10 ticks = 20 snapshots)
        assert len(dataset.agent_snapshots) == 20

    def test_filter_agent_returns_correct_subset(self, config: SimulationConfig, tmp_path):
        """filter_agent() should return only snapshots for specified agent."""
        config.num_agents = 3
        recorder = TrajectoryRecorder(output_dir=str(tmp_path / "trajectories"))
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        recorder.start_run(engine)
        for _ in range(5):
            tick_record = engine.step_all()
            recorder.record_tick(engine, tick_record)
        recorder.end_run(engine)

        # Load and filter
        jsonl_path = recorder.output_dir / "trajectory.jsonl"
        dataset = TrajectoryLoader.from_jsonl(str(jsonl_path))

        # Get first agent ID
        first_agent_id = dataset.agent_snapshots[0].agent_id

        filtered = TrajectoryLoader.filter_agent(dataset, first_agent_id)

        # Should have 5 snapshots (one per tick)
        assert len(filtered) == 5
        # All should be for the same agent
        assert all(s.agent_id == first_agent_id for s in filtered)

    def test_action_distribution_counts_correctly(self, config: SimulationConfig, tmp_path):
        """action_distribution() should count actions by type."""
        recorder = TrajectoryRecorder(output_dir=str(tmp_path / "trajectories"))
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        recorder.start_run(engine)
        for _ in range(20):
            tick_record = engine.step_all()
            recorder.record_tick(engine, tick_record)
        recorder.end_run(engine)

        # Load and compute distribution
        jsonl_path = recorder.output_dir / "trajectory.jsonl"
        dataset = TrajectoryLoader.from_jsonl(str(jsonl_path))

        distribution = TrajectoryLoader.action_distribution(dataset)

        # Should have counts for various action types
        assert isinstance(distribution, dict)
        assert sum(distribution.values()) == len(dataset.agent_snapshots)

    def test_filter_ticks_returns_correct_range(self, config: SimulationConfig, tmp_path):
        """filter_ticks() should return snapshots in specified range."""
        recorder = TrajectoryRecorder(output_dir=str(tmp_path / "trajectories"))
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        recorder.start_run(engine)
        for _ in range(10):
            tick_record = engine.step_all()
            recorder.record_tick(engine, tick_record)
        recorder.end_run(engine)

        # Load and filter ticks 3-7
        jsonl_path = recorder.output_dir / "trajectory.jsonl"
        dataset = TrajectoryLoader.from_jsonl(str(jsonl_path))

        filtered = TrajectoryLoader.filter_ticks(dataset, start=3, end=7)

        # Should only have snapshots from ticks 3-7 (inclusive)
        assert all(3 <= s.tick <= 7 for s in filtered.agent_snapshots)
        # With 3 agents, should be 3 * 5 = 15 snapshots
        assert len(filtered.agent_snapshots) == 15


class TestTrajectoryExporter:
    """Test TrajectoryExporter."""

    def test_export_to_csv_creates_flat_table(self, config: SimulationConfig, tmp_path):
        """to_csv() should create flat CSV file with one row per agent per tick."""
        recorder = TrajectoryRecorder(output_dir=str(tmp_path / "trajectories"))
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        recorder.start_run(engine)
        for _ in range(5):
            tick_record = engine.step_all()
            recorder.record_tick(engine, tick_record)
        recorder.end_run(engine)

        # Load and export
        jsonl_path = recorder.output_dir / "trajectory.jsonl"
        dataset = TrajectoryLoader.from_jsonl(str(jsonl_path))

        csv_path = tmp_path / "trajectory.csv"
        TrajectoryExporter.to_csv(dataset, str(csv_path))

        assert csv_path.exists()

        # Read CSV and verify structure
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Should have 3 agents × 5 ticks = 15 rows
        assert len(rows) == 15

    def test_export_csv_columns_match_schema(self, config: SimulationConfig, tmp_path):
        """CSV should have expected columns from schema."""
        recorder = TrajectoryRecorder(output_dir=str(tmp_path / "trajectories"))
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        recorder.start_run(engine)
        tick_record = engine.step_all()
        recorder.record_tick(engine, tick_record)
        recorder.end_run(engine)

        # Load and export
        jsonl_path = recorder.output_dir / "trajectory.jsonl"
        dataset = TrajectoryLoader.from_jsonl(str(jsonl_path))

        csv_path = tmp_path / "trajectory.csv"
        TrajectoryExporter.to_csv(dataset, str(csv_path))

        # Read CSV headers
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            headers = next(reader)

        # Verify expected columns exist
        expected_columns = [
            "tick",
            "agent_id",
            "agent_name",
            "archetype",
            "x",
            "y",
            "alive",
            "hunger",
            "thirst",
            "energy",
            "health",
            "action_type",
            "action_succeeded",
        ]
        for col in expected_columns:
            assert col in headers

    def test_export_to_agent_summary_creates_aggregated_stats(
        self, config: SimulationConfig, tmp_path
    ):
        """to_agent_summary() should create per-agent aggregated stats."""
        recorder = TrajectoryRecorder(output_dir=str(tmp_path / "trajectories"))
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        recorder.start_run(engine)
        for _ in range(10):
            tick_record = engine.step_all()
            recorder.record_tick(engine, tick_record)
        recorder.end_run(engine)

        # Load and export
        jsonl_path = recorder.output_dir / "trajectory.jsonl"
        dataset = TrajectoryLoader.from_jsonl(str(jsonl_path))

        summary_path = tmp_path / "agent_summary.csv"
        TrajectoryExporter.to_agent_summary(dataset, str(summary_path))

        assert summary_path.exists()

        # Read CSV and verify structure
        with open(summary_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Should have 3 rows (one per agent)
        assert len(rows) == 3
        # Verify expected columns
        assert "agent_id" in rows[0]
        assert "survival_ticks" in rows[0]
        assert "avg_hunger" in rows[0]


class TestIntegration:
    """Integration tests for trajectory system."""

    def test_50_tick_simulation_produces_valid_trajectory(self, config: SimulationConfig, tmp_path):
        """Full 50-tick simulation should produce complete valid trajectory."""
        config.num_agents = 5
        config.max_ticks = 50

        recorder = TrajectoryRecorder(output_dir=str(tmp_path / "trajectories"))
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        recorder.start_run(engine)
        for _ in range(50):
            tick_record = engine.step_all()
            recorder.record_tick(engine, tick_record)
        recorder.end_run(engine)

        # Load and verify
        jsonl_path = recorder.output_dir / "trajectory.jsonl"
        dataset = TrajectoryLoader.from_jsonl(str(jsonl_path))

        assert dataset.metadata.actual_ticks == 50
        # Snapshots depend on survival, but should have at least some
        assert len(dataset.agent_snapshots) > 0
        # All snapshots should be valid (ticks 0-49, but might have tick 50 from final state)
        for snapshot in dataset.agent_snapshots:
            assert 0 <= snapshot.tick <= 50  # <= because engine might record final state
            assert snapshot.agent_id
            assert snapshot.agent_name

    def test_empty_simulation_produces_minimal_valid_trajectory(
        self, config: SimulationConfig, tmp_path
    ):
        """Simulation with 0 ticks should produce valid minimal trajectory."""
        config.num_agents = 1

        recorder = TrajectoryRecorder(output_dir=str(tmp_path / "trajectories"))
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        recorder.start_run(engine)
        recorder.end_run(engine)

        # Load and verify
        jsonl_path = recorder.output_dir / "trajectory.jsonl"
        dataset = TrajectoryLoader.from_jsonl(str(jsonl_path))

        assert dataset.metadata.actual_ticks == 0
        assert len(dataset.agent_snapshots) == 0

    def test_single_tick_simulation(self, config: SimulationConfig, tmp_path):
        """Simulation with single tick should work correctly."""
        config.num_agents = 2

        recorder = TrajectoryRecorder(output_dir=str(tmp_path / "trajectories"))
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        recorder.start_run(engine)
        tick_record = engine.step_all()
        recorder.record_tick(engine, tick_record)
        recorder.end_run(engine)

        # Load and verify
        jsonl_path = recorder.output_dir / "trajectory.jsonl"
        dataset = TrajectoryLoader.from_jsonl(str(jsonl_path))

        assert dataset.metadata.actual_ticks == 1
        assert len(dataset.agent_snapshots) == 2
