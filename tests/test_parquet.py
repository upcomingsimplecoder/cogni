"""Tests for Parquet export/import of trajectory data.

Verifies:
- ParquetExporter.export() creates all required files
- ParquetExporter.load() reconstructs TrajectoryDataset correctly
- Round-trip preserves scalar fields, nested JSON, nullable fields
- Empty datasets handle gracefully
- Float precision maintained within tolerance
- Schema version included in metadata
"""

from __future__ import annotations

import pytest

# Skip all tests if pyarrow not installed
pa = pytest.importorskip("pyarrow")

from src.trajectory.parquet import ParquetExporter  # noqa: E402
from src.trajectory.schema import (  # noqa: E402
    AgentSnapshot,
    EmergenceSnapshot,
    RunMetadata,
    TrajectoryDataset,
)


@pytest.fixture
def sample_dataset():
    """Create a sample TrajectoryDataset with 2 agents over 3 ticks."""
    metadata = RunMetadata(
        run_id="test-run-001",
        timestamp="2026-02-24T00:00:00Z",
        seed=42,
        config={"world_width": 32, "world_height": 32},
        num_agents=2,
        max_ticks=100,
        actual_ticks=3,
        agents=[
            {
                "id": "agent_a",
                "name": "Alpha",
                "archetype": "explorer",
                "initial_traits": {},
            },
            {
                "id": "agent_b",
                "name": "Beta",
                "archetype": "gatherer",
                "initial_traits": {},
            },
        ],
        architecture="reactive",
        final_state={"agents_alive": 2, "agents_dead": 0},
    )

    snapshots = []
    for tick in range(3):
        for agent_id, name, archetype in [
            ("agent_a", "Alpha", "explorer"),
            ("agent_b", "Beta", "gatherer"),
        ]:
            snapshots.append(
                AgentSnapshot(
                    tick=tick,
                    agent_id=agent_id,
                    agent_name=name,
                    archetype=archetype,
                    position=(10 + tick, 20),
                    alive=True,
                    hunger=50.0 - tick * 0.35,
                    thirst=50.0 - tick * 0.4,
                    energy=50.0 - tick * 0.3,
                    health=100.0,
                    traits={
                        "cooperation_tendency": 0.5,
                        "curiosity": 0.7,
                        "risk_tolerance": 0.3,
                        "resource_sharing": 0.4,
                        "aggression": 0.2,
                        "sociability": 0.6,
                    },
                    sensation_summary={
                        "visible_agent_count": 1,
                        "visible_resource_tiles": 3,
                        "total_resources": 5,
                        "message_count": 0,
                        "time_of_day": "morning",
                    },
                    reflection={
                        "threat_level": 0.1,
                        "opportunity_score": 0.6,
                        "need_trends": {"hunger": "declining"},
                        "last_action_succeeded": True,
                        "interaction_count": 0,
                    },
                    intention={
                        "primary_goal": "explore",
                        "confidence": 0.8,
                        "target_position": (15, 25),
                        "target_agent_id": None,
                    },
                    action_type="move",
                    action_target="(15, 25)",
                    action_target_agent=None,
                    action_succeeded=True,
                    needs_delta={"hunger": -0.35, "thirst": -0.4},
                    inventory={"berries": 2, "wood": 1},
                    messages_sent=(
                        [{"type": "INFORM", "receiver": "agent_b", "content": "food here"}]
                        if tick == 1 and agent_id == "agent_a"
                        else []
                    ),
                    messages_received=[],
                    internal_monologue="Looking for food" if tick == 0 else "",
                    trait_changes=(
                        [
                            {
                                "trait": "curiosity",
                                "old_value": 0.7,
                                "new_value": 0.71,
                                "delta": 0.01,
                            }
                        ]
                        if tick == 2 and agent_id == "agent_a"
                        else []
                    ),
                    tom_models=(
                        {
                            "agent_b": {
                                "estimated_disposition": 0.6,
                                "prediction_accuracy": 0.5,
                            }
                        }
                        if agent_id == "agent_a"
                        else {}
                    ),
                    tom_model_count=1 if agent_id == "agent_a" else 0,
                    social_relationships=(
                        {
                            "agent_b": {
                                "trust": 0.55,
                                "interaction_count": 3,
                                "net_resources_given": 1,
                                "was_attacked_by": False,
                                "was_helped_by": True,
                                "last_interaction_tick": tick,
                            }
                        }
                        if agent_id == "agent_a"
                        else {}
                    ),
                    coalition_id="coalition_1" if tick >= 2 else None,
                    coalition_role="leader" if tick >= 2 and agent_id == "agent_a" else "",
                    coalition_goal="hunt" if tick >= 2 else "",
                    cultural_repertoire=(
                        {"gather:at_food": {"context": "food", "action": "gather"}}
                        if agent_id == "agent_b"
                        else {}
                    ),
                    cultural_learning_style="prestige" if agent_id == "agent_b" else "",
                    cultural_group_id=0 if agent_id == "agent_b" else -1,
                    transmission_events_this_tick=[],
                    metacog_calibration_curve=[],
                    metacog_deliberation_invoked=False,
                    plan_state={},
                    language_symbols=(
                        [
                            {
                                "form": "grp",
                                "meaning": "action:gather",
                                "success_rate": 0.8,
                                "times_used": 5,
                                "strength": 0.7,
                            }
                        ]
                        if tick == 2 and agent_id == "agent_b"
                        else []
                    ),
                )
            )

    events = [
        EmergenceSnapshot(
            tick=1,
            pattern_type="cluster",
            agents_involved=["agent_a", "agent_b"],
            description="2 agents clustered",
            data={"center": [12, 20]},
        ),
        EmergenceSnapshot(
            tick=2,
            pattern_type="specialization",
            agents_involved=["agent_b"],
            description="agent_b specializes in gathering",
            data={"category": "gathering", "ratio": 0.73},
        ),
    ]

    return TrajectoryDataset(metadata=metadata, agent_snapshots=snapshots, emergence_events=events)


@pytest.fixture
def empty_dataset():
    """Create an empty TrajectoryDataset with no snapshots or events."""
    metadata = RunMetadata(
        run_id="empty-run-001",
        timestamp="2026-02-24T00:00:00Z",
        seed=99,
        config={"world_width": 16, "world_height": 16},
        num_agents=0,
        max_ticks=0,
        actual_ticks=0,
        agents=[],
        architecture="test",
        final_state={},
    )

    return TrajectoryDataset(metadata=metadata, agent_snapshots=[], emergence_events=[])


class TestParquetExport:
    """Test Parquet export functionality."""

    def test_parquet_export_creates_files(self, sample_dataset, tmp_path):
        """export() should create required parquet files.

        Creates agent_snapshots.parquet, emergence_events.parquet, metadata.json.
        """
        output_dir = tmp_path / "parquet_output"

        result = ParquetExporter.export(sample_dataset, str(output_dir))

        # Check all files created
        assert "agent_snapshots" in result
        assert "emergence_events" in result
        assert "metadata" in result

        assert result["agent_snapshots"].exists()
        assert result["emergence_events"].exists()
        assert result["metadata"].exists()

        # Check file names
        assert result["agent_snapshots"].name == "agent_snapshots.parquet"
        assert result["emergence_events"].name == "emergence_events.parquet"
        assert result["metadata"].name == "metadata.json"

    def test_parquet_export_creates_directory_if_needed(self, sample_dataset, tmp_path):
        """export() should create output directory if it doesn't exist."""
        output_dir = tmp_path / "nested" / "dir" / "parquet"

        assert not output_dir.exists()

        ParquetExporter.export(sample_dataset, str(output_dir))

        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_parquet_metadata_has_schema_version(self, sample_dataset, tmp_path):
        """metadata.json should contain schema_version field."""
        import json

        output_dir = tmp_path / "parquet_output"
        result = ParquetExporter.export(sample_dataset, str(output_dir))

        metadata_path = result["metadata"]
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert "schema_version" in metadata
        assert metadata["schema_version"] == "1.0.0"


class TestParquetLoad:
    """Test Parquet load functionality."""

    def test_parquet_load_missing_dir_raises(self, tmp_path):
        """load() should raise FileNotFoundError for nonexistent directory."""
        nonexistent_dir = tmp_path / "does_not_exist"

        with pytest.raises(FileNotFoundError):
            ParquetExporter.load(str(nonexistent_dir))

    def test_parquet_load_missing_agent_file_raises(self, sample_dataset, tmp_path):
        """load() should raise FileNotFoundError if agent_snapshots.parquet missing."""
        output_dir = tmp_path / "incomplete"
        output_dir.mkdir()

        # Create only metadata and events files
        import json

        metadata_dict = sample_dataset.metadata.to_dict()
        metadata_dict["schema_version"] = "1.0.0"
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata_dict, f)

        # Create empty emergence events using same approach as export code
        import pyarrow as pa
        import pyarrow.parquet as pq

        schema = ParquetExporter._build_emergence_schema()
        arrays = [pa.array([], type=field.type) for field in schema]
        empty_table = pa.Table.from_arrays(arrays, schema=schema)
        pq.write_table(empty_table, output_dir / "emergence_events.parquet")

        # Missing agent_snapshots.parquet
        with pytest.raises(FileNotFoundError, match="agent_snapshots.parquet"):
            ParquetExporter.load(str(output_dir))


class TestParquetRoundTrip:
    """Test round-trip export → load preserves data."""

    def test_parquet_round_trip(self, sample_dataset, tmp_path):
        """export → load should produce dataset with same counts and IDs."""
        output_dir = tmp_path / "roundtrip"

        # Export
        ParquetExporter.export(sample_dataset, str(output_dir))

        # Load
        loaded = ParquetExporter.load(str(output_dir))

        # Verify metadata
        assert loaded.metadata.run_id == sample_dataset.metadata.run_id
        assert loaded.metadata.seed == sample_dataset.metadata.seed
        assert loaded.metadata.num_agents == sample_dataset.metadata.num_agents
        assert loaded.metadata.actual_ticks == sample_dataset.metadata.actual_ticks

        # Verify snapshot count
        assert len(loaded.agent_snapshots) == len(sample_dataset.agent_snapshots)

        # Verify emergence event count
        assert len(loaded.emergence_events) == len(sample_dataset.emergence_events)

        # Verify agent IDs
        original_ids = {s.agent_id for s in sample_dataset.agent_snapshots}
        loaded_ids = {s.agent_id for s in loaded.agent_snapshots}
        assert loaded_ids == original_ids

        # Verify ticks
        original_ticks = {s.tick for s in sample_dataset.agent_snapshots}
        loaded_ticks = {s.tick for s in loaded.agent_snapshots}
        assert loaded_ticks == original_ticks

    def test_parquet_round_trip_preserves_scalar_fields(self, sample_dataset, tmp_path):
        """Round-trip should preserve tick, agent_id, hunger, traits, etc."""
        output_dir = tmp_path / "roundtrip_scalars"

        # Export and load
        ParquetExporter.export(sample_dataset, str(output_dir))
        loaded = ParquetExporter.load(str(output_dir))

        # Compare first snapshot in detail
        original = sample_dataset.agent_snapshots[0]
        loaded_snap = next(
            s
            for s in loaded.agent_snapshots
            if s.tick == original.tick and s.agent_id == original.agent_id
        )

        assert loaded_snap.tick == original.tick
        assert loaded_snap.agent_id == original.agent_id
        assert loaded_snap.agent_name == original.agent_name
        assert loaded_snap.archetype == original.archetype
        assert loaded_snap.position == original.position
        assert loaded_snap.alive == original.alive
        assert abs(loaded_snap.hunger - original.hunger) < 1e-4
        assert abs(loaded_snap.thirst - original.thirst) < 1e-4
        assert abs(loaded_snap.energy - original.energy) < 1e-4
        assert abs(loaded_snap.health - original.health) < 1e-4
        assert loaded_snap.action_type == original.action_type
        assert loaded_snap.action_succeeded == original.action_succeeded

        # Check traits
        assert (
            abs(
                loaded_snap.traits["cooperation_tendency"] - original.traits["cooperation_tendency"]
            )
            < 1e-4
        )
        assert abs(loaded_snap.traits["curiosity"] - original.traits["curiosity"]) < 1e-4

    def test_parquet_round_trip_preserves_nested_json(self, sample_dataset, tmp_path):
        """Round-trip should preserve tom_models, social_relationships, etc."""
        output_dir = tmp_path / "roundtrip_json"

        # Export and load
        ParquetExporter.export(sample_dataset, str(output_dir))
        loaded = ParquetExporter.load(str(output_dir))

        # Find agent_a snapshot that has ToM models
        original = next(s for s in sample_dataset.agent_snapshots if s.agent_id == "agent_a")
        loaded_snap = next(
            s
            for s in loaded.agent_snapshots
            if s.tick == original.tick and s.agent_id == original.agent_id
        )

        # Verify ToM models preserved
        assert loaded_snap.tom_models == original.tom_models
        assert loaded_snap.tom_model_count == original.tom_model_count

        # Verify social relationships preserved
        assert loaded_snap.social_relationships == original.social_relationships

        # Verify sensation summary
        assert loaded_snap.sensation_summary == original.sensation_summary

        # Verify reflection
        assert loaded_snap.reflection == original.reflection

        # Verify intention (note: tuples in JSON become lists)
        assert loaded_snap.intention["primary_goal"] == original.intention["primary_goal"]
        assert loaded_snap.intention["confidence"] == original.intention["confidence"]
        # target_position tuple becomes list after JSON round-trip
        assert loaded_snap.intention["target_position"] == list(
            original.intention["target_position"]
        )

    def test_parquet_round_trip_empty_dataset(self, empty_dataset, tmp_path):
        """Empty dataset (0 snapshots, 0 events) should round-trip correctly."""
        output_dir = tmp_path / "roundtrip_empty"

        # Export
        ParquetExporter.export(empty_dataset, str(output_dir))

        # Load
        loaded = ParquetExporter.load(str(output_dir))

        # Verify metadata
        assert loaded.metadata.run_id == empty_dataset.metadata.run_id
        assert loaded.metadata.num_agents == 0
        assert loaded.metadata.actual_ticks == 0

        # Verify empty collections
        assert len(loaded.agent_snapshots) == 0
        assert len(loaded.emergence_events) == 0

    def test_parquet_round_trip_nullable_fields(self, tmp_path):
        """action_target=None, coalition_id=None should round-trip correctly."""
        # Create dataset with nullable fields
        metadata = RunMetadata(
            run_id="nullable-test",
            timestamp="2026-02-24T00:00:00Z",
            seed=1,
            config={},
            num_agents=1,
            max_ticks=1,
            actual_ticks=1,
        )

        snapshot = AgentSnapshot(
            tick=0,
            agent_id="agent_nullable",
            agent_name="NullableAgent",
            archetype="test",
            position=(0, 0),
            alive=True,
            hunger=50.0,
            thirst=50.0,
            energy=50.0,
            health=100.0,
            traits={
                "cooperation_tendency": 0.5,
                "curiosity": 0.5,
                "risk_tolerance": 0.5,
                "resource_sharing": 0.5,
                "aggression": 0.5,
                "sociability": 0.5,
            },
            sensation_summary={},
            reflection={},
            intention={},
            action_type="wait",
            action_target=None,  # Nullable
            action_target_agent=None,  # Nullable
            action_succeeded=True,
            needs_delta={},
            inventory={},
            coalition_id=None,  # Nullable
        )

        dataset = TrajectoryDataset(
            metadata=metadata, agent_snapshots=[snapshot], emergence_events=[]
        )

        output_dir = tmp_path / "roundtrip_nullable"

        # Export and load
        ParquetExporter.export(dataset, str(output_dir))
        loaded = ParquetExporter.load(str(output_dir))

        loaded_snap = loaded.agent_snapshots[0]

        # Verify nullable fields preserved
        assert loaded_snap.action_target is None
        assert loaded_snap.action_target_agent is None
        assert loaded_snap.coalition_id is None

    def test_parquet_float_precision(self, sample_dataset, tmp_path):
        """float32 values should be within 1e-4 tolerance after round-trip."""
        output_dir = tmp_path / "roundtrip_precision"

        # Export and load
        ParquetExporter.export(sample_dataset, str(output_dir))
        loaded = ParquetExporter.load(str(output_dir))

        # Check all float fields across all snapshots
        for original, loaded_snap in zip(
            sorted(sample_dataset.agent_snapshots, key=lambda s: (s.tick, s.agent_id)),
            sorted(loaded.agent_snapshots, key=lambda s: (s.tick, s.agent_id)),
            strict=False,
        ):
            assert abs(loaded_snap.hunger - original.hunger) < 1e-4
            assert abs(loaded_snap.thirst - original.thirst) < 1e-4
            assert abs(loaded_snap.energy - original.energy) < 1e-4
            assert abs(loaded_snap.health - original.health) < 1e-4

            for trait_name in [
                "cooperation_tendency",
                "curiosity",
                "risk_tolerance",
                "resource_sharing",
                "aggression",
                "sociability",
            ]:
                assert abs(loaded_snap.traits[trait_name] - original.traits[trait_name]) < 1e-4


class TestEmergenceEvents:
    """Test emergence event round-trip."""

    def test_emergence_events_round_trip(self, sample_dataset, tmp_path):
        """Emergence events should round-trip correctly."""
        output_dir = tmp_path / "emergence_roundtrip"

        # Export and load
        ParquetExporter.export(sample_dataset, str(output_dir))
        loaded = ParquetExporter.load(str(output_dir))

        # Compare events
        assert len(loaded.emergence_events) == len(sample_dataset.emergence_events)

        for original, loaded_event in zip(
            sorted(sample_dataset.emergence_events, key=lambda e: e.tick),
            sorted(loaded.emergence_events, key=lambda e: e.tick),
            strict=False,
        ):
            assert loaded_event.tick == original.tick
            assert loaded_event.pattern_type == original.pattern_type
            assert loaded_event.agents_involved == original.agents_involved
            assert loaded_event.description == original.description
            assert loaded_event.data == original.data


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_large_internal_monologue(self, tmp_path):
        """Large string fields should round-trip correctly."""
        metadata = RunMetadata(
            run_id="large-string-test",
            timestamp="2026-02-24T00:00:00Z",
            seed=1,
            config={},
            num_agents=1,
            max_ticks=1,
            actual_ticks=1,
        )

        # Create snapshot with large internal monologue
        large_monologue = "x" * 10000  # 10KB string

        snapshot = AgentSnapshot(
            tick=0,
            agent_id="agent_large",
            agent_name="LargeAgent",
            archetype="test",
            position=(0, 0),
            alive=True,
            hunger=50.0,
            thirst=50.0,
            energy=50.0,
            health=100.0,
            traits={
                "cooperation_tendency": 0.5,
                "curiosity": 0.5,
                "risk_tolerance": 0.5,
                "resource_sharing": 0.5,
                "aggression": 0.5,
                "sociability": 0.5,
            },
            sensation_summary={},
            reflection={},
            intention={},
            action_type="wait",
            action_target=None,
            action_target_agent=None,
            action_succeeded=True,
            needs_delta={},
            inventory={},
            internal_monologue=large_monologue,
        )

        dataset = TrajectoryDataset(
            metadata=metadata, agent_snapshots=[snapshot], emergence_events=[]
        )

        output_dir = tmp_path / "large_string"

        # Export and load
        ParquetExporter.export(dataset, str(output_dir))
        loaded = ParquetExporter.load(str(output_dir))

        # Verify large string preserved
        assert loaded.agent_snapshots[0].internal_monologue == large_monologue

    def test_complex_nested_json_structures(self, tmp_path):
        """Complex nested JSON in fields should round-trip correctly."""
        metadata = RunMetadata(
            run_id="complex-json-test",
            timestamp="2026-02-24T00:00:00Z",
            seed=1,
            config={},
            num_agents=1,
            max_ticks=1,
            actual_ticks=1,
        )

        # Create snapshot with complex nested structures
        snapshot = AgentSnapshot(
            tick=0,
            agent_id="agent_complex",
            agent_name="ComplexAgent",
            archetype="test",
            position=(0, 0),
            alive=True,
            hunger=50.0,
            thirst=50.0,
            energy=50.0,
            health=100.0,
            traits={
                "cooperation_tendency": 0.5,
                "curiosity": 0.5,
                "risk_tolerance": 0.5,
                "resource_sharing": 0.5,
                "aggression": 0.5,
                "sociability": 0.5,
            },
            sensation_summary={
                "nested": {"deeply": {"nested": {"value": [1, 2, 3], "map": {"key": "value"}}}}
            },
            reflection={},
            intention={},
            action_type="wait",
            action_target=None,
            action_target_agent=None,
            action_succeeded=True,
            needs_delta={},
            inventory={},
            plan_state={
                "goal": "complex_goal",
                "steps": ["step1", "step2", "step3"],
                "metadata": {"nested": True, "levels": 3},
            },
        )

        dataset = TrajectoryDataset(
            metadata=metadata, agent_snapshots=[snapshot], emergence_events=[]
        )

        output_dir = tmp_path / "complex_json"

        # Export and load
        ParquetExporter.export(dataset, str(output_dir))
        loaded = ParquetExporter.load(str(output_dir))

        # Verify complex structures preserved
        loaded_snap = loaded.agent_snapshots[0]
        assert loaded_snap.sensation_summary == snapshot.sensation_summary
        assert loaded_snap.plan_state == snapshot.plan_state
