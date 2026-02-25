"""Tests for trajectory query interface."""

from __future__ import annotations

import pytest

# Optional dependencies for testing
duckdb = pytest.importorskip("duckdb")
pa = pytest.importorskip("pyarrow")


@pytest.fixture
def query_data(tmp_path):
    """Create Parquet data for query testing."""
    from src.trajectory.parquet import ParquetExporter
    from src.trajectory.schema import (
        AgentSnapshot,
        EmergenceSnapshot,
        RunMetadata,
        TrajectoryDataset,
    )

    # Create 2 runs with different characteristics
    for run_idx, (run_id, arch) in enumerate(
        [("run-001", "reactive"), ("run-002", "dual_process")]
    ):
        metadata = RunMetadata(
            run_id=run_id,
            timestamp="2026-02-24T00:00:00Z",
            seed=42 + run_idx,
            config={},
            num_agents=2,
            max_ticks=10,
            actual_ticks=5,
            architecture=arch,
            final_state={"agents_alive": 2, "agents_dead": 0},
        )

        snapshots = []
        for tick in range(5):
            for agent_id, archetype in [("a1", "explorer"), ("a2", "gatherer")]:
                snapshots.append(
                    AgentSnapshot(
                        tick=tick,
                        agent_id=agent_id,
                        agent_name=f"Agent_{agent_id}",
                        archetype=archetype,
                        position=(10, 10),
                        alive=True,
                        hunger=50.0 - tick,
                        thirst=50.0,
                        energy=50.0,
                        health=100.0,
                        traits={
                            "cooperation_tendency": 0.5,
                            "curiosity": 0.7,
                            "risk_tolerance": 0.3,
                            "resource_sharing": 0.4,
                            "aggression": 0.2,
                            "sociability": 0.6,
                        },
                        sensation_summary={},
                        reflection={"threat_level": 0.1, "opportunity_score": 0.5},
                        intention={"primary_goal": "explore", "confidence": 0.8},
                        action_type="move" if tick % 2 == 0 else "gather",
                        action_target=None,
                        action_target_agent=None,
                        action_succeeded=True,
                        needs_delta={},
                        inventory={},
                    )
                )

        events = [
            EmergenceSnapshot(
                tick=2,
                pattern_type="cluster",
                agents_involved=["a1", "a2"],
                description="test cluster",
                data={},
            )
        ]

        dataset = TrajectoryDataset(
            metadata=metadata, agent_snapshots=snapshots, emergence_events=events
        )
        run_dir = str(tmp_path / run_id)
        ParquetExporter.export(dataset, run_dir)

    return tmp_path


def test_sql_basic_query(query_data):
    """Test basic SQL query over snapshots."""
    from src.trajectory.query import TrajectoryQuery

    query = TrajectoryQuery(data_dir=str(query_data))

    # 2 runs × 2 agents × 5 ticks = 20 total snapshots
    result = query.sql("SELECT COUNT(*) as n FROM snapshots")
    assert len(result) == 1
    assert result[0]["n"] == 20

    query.close()


def test_sql_filter_by_archetype(query_data):
    """Test filtering snapshots by archetype."""
    from src.trajectory.query import TrajectoryQuery

    query = TrajectoryQuery(data_dir=str(query_data))

    # Filter for explorer archetype: 2 runs × 1 explorer × 5 ticks = 10
    result = query.sql("SELECT COUNT(*) as n FROM snapshots WHERE archetype = 'explorer'")
    assert result[0]["n"] == 10

    # Filter for gatherer archetype: 2 runs × 1 gatherer × 5 ticks = 10
    result = query.sql("SELECT COUNT(*) as n FROM snapshots WHERE archetype = 'gatherer'")
    assert result[0]["n"] == 10

    query.close()


def test_trait_evolution(query_data):
    """Test trait evolution time series for specific agent."""
    from src.trajectory.query import TrajectoryQuery

    query = TrajectoryQuery(data_dir=str(query_data))

    # Get trait evolution for agent a1 in run-001
    result = query.trait_evolution("a1", run_id="run-001")

    # Should have 5 ticks (0-4)
    assert len(result) == 5

    # Check all traits are present and consistent (use approx for float32 precision)
    for row in result:
        assert "tick" in row
        assert row["cooperation_tendency"] == pytest.approx(0.5, abs=0.01)
        assert row["curiosity"] == pytest.approx(0.7, abs=0.01)
        assert row["risk_tolerance"] == pytest.approx(0.3, abs=0.01)
        assert row["resource_sharing"] == pytest.approx(0.4, abs=0.01)
        assert row["aggression"] == pytest.approx(0.2, abs=0.01)
        assert row["sociability"] == pytest.approx(0.6, abs=0.01)

    # Verify ticks are ordered
    ticks = [row["tick"] for row in result]
    assert ticks == [0, 1, 2, 3, 4]

    query.close()


def test_action_distribution(query_data):
    """Test action distribution counts."""
    from src.trajectory.query import TrajectoryQuery

    query = TrajectoryQuery(data_dir=str(query_data))

    # Get all actions
    result = query.action_distribution()

    # Should have move and gather
    assert "move" in result
    assert "gather" in result

    # Total should be 20 (all snapshots)
    assert result["move"] + result["gather"] == 20

    # Move happens on even ticks (0, 2, 4) = 3 ticks × 2 agents × 2 runs = 12
    assert result["move"] == 12

    # Gather happens on odd ticks (1, 3) = 2 ticks × 2 agents × 2 runs = 8
    assert result["gather"] == 8

    query.close()


def test_action_distribution_filtered(query_data):
    """Test action distribution filtered by archetype."""
    from src.trajectory.query import TrajectoryQuery

    query = TrajectoryQuery(data_dir=str(query_data))

    # Filter by explorer archetype
    result = query.action_distribution(archetype="explorer")

    # Should have 10 total actions for explorer (2 runs × 5 ticks)
    assert result["move"] + result["gather"] == 10

    # Move: 3 ticks × 2 runs = 6
    assert result["move"] == 6

    # Gather: 2 ticks × 2 runs = 4
    assert result["gather"] == 4

    query.close()


def test_survival_curve(query_data):
    """Test survival curve for a run."""
    from src.trajectory.query import TrajectoryQuery

    query = TrajectoryQuery(data_dir=str(query_data))

    # Get survival curve for run-001
    result = query.survival_curve("run-001")

    # Should have 5 ticks
    assert len(result) == 5

    # All agents alive in all ticks (2 agents)
    for row in result:
        assert row["agents_alive"] == 2

    # Verify ticks are ordered
    ticks = [row["tick"] for row in result]
    assert ticks == [0, 1, 2, 3, 4]

    query.close()


def test_cross_run_comparison(query_data):
    """Test cross-run comparison grouped by archetype."""
    from src.trajectory.query import TrajectoryQuery

    query = TrajectoryQuery(data_dir=str(query_data))

    # Compare health across archetypes
    result = query.cross_run_comparison("health", group_by="archetype")

    # Should have 2 groups (explorer, gatherer)
    assert len(result) == 2

    # Find explorer and gatherer rows
    explorer_row = next(r for r in result if r["archetype"] == "explorer")
    gatherer_row = next(r for r in result if r["archetype"] == "gatherer")

    # All health values are 100.0
    assert explorer_row["avg_health"] == 100.0
    assert explorer_row["min_health"] == 100.0
    assert explorer_row["max_health"] == 100.0
    assert explorer_row["sample_count"] == 10  # 2 runs × 5 ticks

    assert gatherer_row["avg_health"] == 100.0
    assert gatherer_row["min_health"] == 100.0
    assert gatherer_row["max_health"] == 100.0
    assert gatherer_row["sample_count"] == 10

    query.close()


def test_no_parquet_data_error(tmp_path):
    """Test error handling when no Parquet files exist."""
    from src.trajectory.query import TrajectoryQuery

    # Create empty directory
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    query = TrajectoryQuery(data_dir=str(empty_dir))

    # Should raise ValueError with helpful message
    with pytest.raises(ValueError, match="No Parquet data found"):
        query.sql("SELECT COUNT(*) FROM snapshots")

    query.close()


def test_trait_evolution_no_run_filter(query_data):
    """Test trait evolution across all runs for an agent ID."""
    from src.trajectory.query import TrajectoryQuery

    query = TrajectoryQuery(data_dir=str(query_data))

    # Get trait evolution for agent a1 across all runs (no run_id filter)
    result = query.trait_evolution("a1")

    # Should have 10 rows (2 runs × 5 ticks)
    assert len(result) == 10

    # All should have same trait values (use approx for float32 precision)
    for row in result:
        assert row["cooperation_tendency"] == pytest.approx(0.5, abs=0.01)
        assert row["curiosity"] == pytest.approx(0.7, abs=0.01)

    query.close()


def test_action_distribution_with_run_filter(query_data):
    """Test action distribution filtered by run ID."""
    from src.trajectory.query import TrajectoryQuery

    query = TrajectoryQuery(data_dir=str(query_data))

    # Filter by run-001
    result = query.action_distribution(run_id="run-001")

    # Should have 10 total actions (2 agents × 5 ticks)
    assert result["move"] + result["gather"] == 10

    # Move: 3 ticks × 2 agents = 6
    assert result["move"] == 6

    # Gather: 2 ticks × 2 agents = 4
    assert result["gather"] == 4

    query.close()
