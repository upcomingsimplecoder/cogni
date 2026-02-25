"""Tests for run catalog."""

from __future__ import annotations

import json

import pytest

duckdb = pytest.importorskip("duckdb")

from src.trajectory.catalog import RunCatalog  # noqa: E402


def create_fake_run(
    base_dir,
    run_id,
    seed=42,
    architecture="reactive",
    num_agents=5,
    actual_ticks=100,
    agents_alive=3,
    agents_dead=2,
):
    """Create a minimal run directory with metadata.json."""
    run_dir = base_dir / "trajectories" / run_id
    run_dir.mkdir(parents=True)
    metadata = {
        "run_id": run_id,
        "timestamp": "2026-02-24T00:00:00Z",
        "seed": seed,
        "config": {"world_width": 32},
        "num_agents": num_agents,
        "max_ticks": 1000,
        "actual_ticks": actual_ticks,
        "agents": [],
        "architecture": architecture,
        "final_state": {"agents_alive": agents_alive, "agents_dead": agents_dead},
        "schema_version": "1.0.0",
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)
    return run_dir


def test_catalog_register_and_get(tmp_path):
    """Test registering a run and retrieving it."""
    # Create fake run
    create_fake_run(
        tmp_path,
        "run-001",
        seed=123,
        architecture="deliberative",
        num_agents=10,
        actual_ticks=500,
    )

    # Create catalog
    catalog = RunCatalog(data_dir=str(tmp_path))

    # Register the run
    run_dir = tmp_path / "trajectories" / "run-001"
    catalog.register_run(str(run_dir))

    # Get the run
    run = catalog.get_run("run-001")

    assert run is not None
    assert run["run_id"] == "run-001"
    assert run["seed"] == 123
    assert run["architecture"] == "deliberative"
    assert run["num_agents"] == 10
    assert run["actual_ticks"] == 500
    assert run["agents_alive"] == 3
    assert run["agents_dead"] == 2
    assert run["has_parquet"] is False

    catalog.close()


def test_catalog_rebuild_finds_runs(tmp_path):
    """Test rebuild scans and indexes all runs."""
    # Create two fake runs
    create_fake_run(tmp_path, "run-001", architecture="reactive")
    create_fake_run(tmp_path, "run-002", architecture="deliberative")

    # Create catalog and rebuild
    catalog = RunCatalog(data_dir=str(tmp_path))
    count = catalog.rebuild()

    assert count == 2

    # Verify both runs are indexed
    run1 = catalog.get_run("run-001")
    run2 = catalog.get_run("run-002")

    assert run1 is not None
    assert run2 is not None
    assert run1["architecture"] == "reactive"
    assert run2["architecture"] == "deliberative"

    catalog.close()


def test_catalog_list_with_filters(tmp_path):
    """Test filtering by architecture."""
    # Create runs with different architectures
    create_fake_run(tmp_path, "run-001", architecture="reactive")
    create_fake_run(tmp_path, "run-002", architecture="deliberative")
    create_fake_run(tmp_path, "run-003", architecture="reactive")

    # Build catalog
    catalog = RunCatalog(data_dir=str(tmp_path))
    catalog.rebuild()

    # Filter by architecture
    reactive_runs = catalog.list_runs(architecture="reactive")
    deliberative_runs = catalog.list_runs(architecture="deliberative")

    assert len(reactive_runs) == 2
    assert len(deliberative_runs) == 1
    assert all(r["architecture"] == "reactive" for r in reactive_runs)
    assert all(r["architecture"] == "deliberative" for r in deliberative_runs)

    catalog.close()


def test_catalog_list_with_min_ticks(tmp_path):
    """Test filtering by minimum ticks."""
    # Create runs with different tick counts
    create_fake_run(tmp_path, "run-001", actual_ticks=50)
    create_fake_run(tmp_path, "run-002", actual_ticks=150)
    create_fake_run(tmp_path, "run-003", actual_ticks=300)

    # Build catalog
    catalog = RunCatalog(data_dir=str(tmp_path))
    catalog.rebuild()

    # Filter by min_ticks
    runs = catalog.list_runs(min_ticks=100)

    assert len(runs) == 2
    assert all(r["actual_ticks"] >= 100 for r in runs)

    catalog.close()


def test_catalog_stats(tmp_path):
    """Test catalog statistics."""
    # Create runs
    create_fake_run(tmp_path, "run-001", actual_ticks=100, architecture="reactive")
    create_fake_run(tmp_path, "run-002", actual_ticks=200, architecture="deliberative")

    # Build catalog
    catalog = RunCatalog(data_dir=str(tmp_path))
    catalog.rebuild()

    # Get stats
    stats = catalog.stats()

    assert stats["total_runs"] == 2
    assert stats["total_ticks"] == 300
    assert set(stats["architectures"]) == {"reactive", "deliberative"}
    assert stats["earliest_run"] == "2026-02-24T00:00:00Z"
    assert stats["latest_run"] == "2026-02-24T00:00:00Z"

    catalog.close()


def test_catalog_get_nonexistent(tmp_path):
    """Test getting a non-existent run returns None."""
    catalog = RunCatalog(data_dir=str(tmp_path))

    run = catalog.get_run("fake-run-id")

    assert run is None

    catalog.close()


def test_catalog_compare_runs(tmp_path):
    """Test comparing multiple runs."""
    # Create runs
    create_fake_run(tmp_path, "run-001", seed=123, architecture="reactive")
    create_fake_run(tmp_path, "run-002", seed=456, architecture="deliberative")

    # Build catalog
    catalog = RunCatalog(data_dir=str(tmp_path))
    catalog.rebuild()

    # Compare runs
    comparison = catalog.compare_runs(["run-001", "run-002"])

    assert "runs" in comparison
    assert len(comparison["runs"]) == 2

    # Verify both runs are present
    run_ids = {r["run_id"] for r in comparison["runs"]}
    assert run_ids == {"run-001", "run-002"}

    catalog.close()


def test_catalog_register_with_jsonl_fallback(tmp_path):
    """Test registering a run from JSONL when metadata.json doesn't exist."""
    # Create run directory without metadata.json
    run_dir = tmp_path / "trajectories" / "run-jsonl"
    run_dir.mkdir(parents=True)

    # Create trajectory.jsonl with metadata as first line
    metadata_record = {
        "type": "metadata",
        "run_id": "run-jsonl",
        "timestamp": "2026-02-24T12:00:00Z",
        "seed": 999,
        "config": {"world_width": 64},
        "num_agents": 8,
        "max_ticks": 500,
        "actual_ticks": 250,
        "agents": [],
        "architecture": "hybrid",
        "final_state": {"agents_alive": 5, "agents_dead": 3},
    }

    with open(run_dir / "trajectory.jsonl", "w") as f:
        f.write(json.dumps(metadata_record) + "\n")

    # Create catalog and register
    catalog = RunCatalog(data_dir=str(tmp_path))
    catalog.register_run(str(run_dir))

    # Verify registration
    run = catalog.get_run("run-jsonl")
    assert run is not None
    assert run["run_id"] == "run-jsonl"
    assert run["seed"] == 999
    assert run["architecture"] == "hybrid"

    catalog.close()


def test_catalog_list_with_multiple_filters(tmp_path):
    """Test combining multiple filters."""
    # Create varied runs
    create_fake_run(tmp_path, "run-001", architecture="reactive", num_agents=5, actual_ticks=100)
    create_fake_run(tmp_path, "run-002", architecture="reactive", num_agents=10, actual_ticks=200)
    create_fake_run(
        tmp_path, "run-003", architecture="deliberative", num_agents=10, actual_ticks=150
    )

    # Build catalog
    catalog = RunCatalog(data_dir=str(tmp_path))
    catalog.rebuild()

    # Filter by architecture AND min_agents AND min_ticks
    runs = catalog.list_runs(architecture="reactive", min_agents=8, min_ticks=150)

    assert len(runs) == 1
    assert runs[0]["run_id"] == "run-002"

    catalog.close()


def test_catalog_limit(tmp_path):
    """Test limit parameter works correctly."""
    # Create 5 runs
    for i in range(5):
        create_fake_run(tmp_path, f"run-{i:03d}")

    # Build catalog
    catalog = RunCatalog(data_dir=str(tmp_path))
    catalog.rebuild()

    # List with limit
    runs = catalog.list_runs(limit=3)

    assert len(runs) == 3

    catalog.close()
