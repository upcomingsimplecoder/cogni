"""Tests for experiment provenance tracking."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.experiments.provenance import (
    ExperimentProvenance,
    _get_autocog_version,
    _get_dependency_versions,
    _get_git_commit,
    _hash_file,
    _is_git_dirty,
    capture_provenance,
)


def test_capture_provenance_returns_all_fields():
    """Test that capture_provenance returns a complete ExperimentProvenance object."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("name: test\n")
        yaml_path = f.name

    try:
        config_resolved = {
            "name": "test",
            "replicates": 5,
            "seed_start": 42,
        }
        seed_range = [42, 43, 44, 45, 46]

        provenance = capture_provenance(
            experiment_id="test-123",
            config_yaml_path=yaml_path,
            config_resolved=config_resolved,
            seed_range=seed_range,
            duration_seconds=10.5,
        )

        # Check all required fields are present
        assert provenance.experiment_id == "test-123"
        assert provenance.timestamp is not None
        assert provenance.git_commit is None or isinstance(provenance.git_commit, str)
        assert isinstance(provenance.git_dirty, bool)
        assert provenance.python_version is not None
        assert provenance.platform_info is not None
        assert provenance.autocog_version is not None
        assert provenance.config_yaml_path == yaml_path
        assert provenance.config_yaml_hash is not None
        assert provenance.config_resolved == config_resolved
        assert provenance.seed_range == seed_range
        assert provenance.duration_seconds == 10.5
        assert isinstance(provenance.dependencies, dict)

    finally:
        Path(yaml_path).unlink(missing_ok=True)


def test_provenance_round_trip_serialization():
    """Test that provenance can be serialized and deserialized."""
    original = ExperimentProvenance(
        experiment_id="test-456",
        timestamp="2024-01-01T00:00:00Z",
        git_commit="abc123",
        git_dirty=False,
        python_version="3.12.1",
        platform_info="Linux-6.1-x86_64",
        autocog_version="0.1.0",
        config_yaml_path="test.yaml",
        config_yaml_hash="def456",
        config_resolved={"name": "test"},
        seed_range=[1, 2, 3],
        duration_seconds=5.5,
        dependencies={"pydantic": "2.0.0"},
    )

    # Serialize
    data = original.to_dict()
    assert isinstance(data, dict)

    # Deserialize
    restored = ExperimentProvenance.from_dict(data)
    assert restored.experiment_id == original.experiment_id
    assert restored.timestamp == original.timestamp
    assert restored.git_commit == original.git_commit
    assert restored.git_dirty == original.git_dirty
    assert restored.config_yaml_hash == original.config_yaml_hash
    assert restored.seed_range == original.seed_range


@patch("subprocess.run")
def test_git_commit_capture(mock_run):
    """Test git commit capture with mocked subprocess."""
    mock_result = Mock()
    mock_result.returncode = 0
    mock_result.stdout = "abc123def456\n"
    mock_run.return_value = mock_result

    commit = _get_git_commit()
    assert commit == "abc123def456"

    mock_run.assert_called_once_with(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )


@patch("subprocess.run")
def test_git_commit_not_in_repo(mock_run):
    """Test git commit capture when not in a repo."""
    mock_result = Mock()
    mock_result.returncode = 128  # Not a git repo
    mock_run.return_value = mock_result

    commit = _get_git_commit()
    assert commit is None


@patch("subprocess.run")
def test_git_dirty_detection_clean(mock_run):
    """Test git dirty detection with clean working tree."""
    mock_result = Mock()
    mock_result.returncode = 0
    mock_result.stdout = ""  # Empty = clean
    mock_run.return_value = mock_result

    is_dirty = _is_git_dirty()
    assert is_dirty is False


@patch("subprocess.run")
def test_git_dirty_detection_dirty(mock_run):
    """Test git dirty detection with uncommitted changes."""
    mock_result = Mock()
    mock_result.returncode = 0
    mock_result.stdout = " M src/file.py\n"  # Modified file
    mock_run.return_value = mock_result

    is_dirty = _is_git_dirty()
    assert is_dirty is True


@patch("subprocess.run")
def test_provenance_handles_missing_git(mock_run):
    """Test provenance capture when git is not available."""
    mock_run.side_effect = FileNotFoundError("git not found")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("name: test\n")
        yaml_path = f.name

    try:
        provenance = capture_provenance(
            experiment_id="test-789",
            config_yaml_path=yaml_path,
            config_resolved={"name": "test"},
            seed_range=[1],
            duration_seconds=1.0,
        )

        # Should not fail, just return None for git fields
        assert provenance.git_commit is None
        assert provenance.git_dirty is False

    finally:
        Path(yaml_path).unlink(missing_ok=True)


def test_hash_file_deterministic():
    """Test that file hashing is deterministic."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("test content\n")
        path = f.name

    try:
        hash1 = _hash_file(path)
        hash2 = _hash_file(path)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex digest length

    finally:
        Path(path).unlink(missing_ok=True)


def test_hash_file_missing():
    """Test file hashing with missing file."""
    hash_val = _hash_file("/nonexistent/file.yaml")
    assert hash_val == "file_not_found"


def test_get_autocog_version():
    """Test AUTOCOG version extraction."""
    version = _get_autocog_version()
    assert version == "0.1.1"  # From pyproject.toml


def test_get_dependency_versions():
    """Test dependency version extraction."""
    deps = _get_dependency_versions()

    assert isinstance(deps, dict)
    assert "pydantic" in deps
    assert "pyyaml" in deps
    assert "rich" in deps
    assert "openai" in deps

    # At least one should be installed (pydantic is required)
    assert deps["pydantic"] != "not_installed"


def test_registry_append_and_list():
    """Test registry append and list operations."""
    from src.experiments.registry import ExperimentRegistry

    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = Path(tmpdir) / "registry.jsonl"
        registry = ExperimentRegistry(str(registry_path))

        # Create provenance
        prov = ExperimentProvenance(
            experiment_id="exp-001",
            timestamp="2024-01-01T00:00:00Z",
            git_commit="abc123",
            git_dirty=False,
            python_version="3.12.1",
            platform_info="Linux",
            autocog_version="0.1.0",
            config_yaml_path="test.yaml",
            config_yaml_hash="hash123",
            config_resolved={"name": "test1"},
            seed_range=[1, 2, 3],
            duration_seconds=5.0,
            dependencies={},
        )

        # Register
        registry.register(prov, "/tmp/results")

        # List
        experiments = registry.list_experiments()
        assert len(experiments) == 1
        assert experiments[0]["provenance"]["experiment_id"] == "exp-001"
        assert experiments[0]["results_dir"] == "/tmp/results"


def test_registry_find_by_config_hash():
    """Test finding experiments by config hash."""
    from src.experiments.registry import ExperimentRegistry

    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = Path(tmpdir) / "registry.jsonl"
        registry = ExperimentRegistry(str(registry_path))

        # Register two experiments with same config hash
        for i in range(2):
            prov = ExperimentProvenance(
                experiment_id=f"exp-{i:03d}",
                timestamp=f"2024-01-0{i + 1}T00:00:00Z",
                git_commit="abc123",
                git_dirty=False,
                python_version="3.12.1",
                platform_info="Linux",
                autocog_version="0.1.0",
                config_yaml_path="test.yaml",
                config_yaml_hash="same_hash",
                config_resolved={"name": f"test{i}"},
                seed_range=[1, 2, 3],
                duration_seconds=5.0,
                dependencies={},
            )
            registry.register(prov, f"/tmp/results{i}")

        # Register one with different hash
        prov_diff = ExperimentProvenance(
            experiment_id="exp-999",
            timestamp="2024-01-03T00:00:00Z",
            git_commit="abc123",
            git_dirty=False,
            python_version="3.12.1",
            platform_info="Linux",
            autocog_version="0.1.0",
            config_yaml_path="test.yaml",
            config_yaml_hash="different_hash",
            config_resolved={"name": "test999"},
            seed_range=[1, 2, 3],
            duration_seconds=5.0,
            dependencies={},
        )
        registry.register(prov_diff, "/tmp/results999")

        # Find by config hash
        found = registry.find_by_config("same_hash")
        assert len(found) == 2
        assert all(exp["provenance"]["config_yaml_hash"] == "same_hash" for exp in found)


def test_registry_compare_experiments():
    """Test comparing two experiments."""
    from src.experiments.registry import ExperimentRegistry

    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = Path(tmpdir) / "registry.jsonl"
        registry = ExperimentRegistry(str(registry_path))

        # Register two experiments
        prov_a = ExperimentProvenance(
            experiment_id="exp-a",
            timestamp="2024-01-01T00:00:00Z",
            git_commit="abc123",
            git_dirty=False,
            python_version="3.12.1",
            platform_info="Linux",
            autocog_version="0.1.0",
            config_yaml_path="test.yaml",
            config_yaml_hash="hash_a",
            config_resolved={"name": "test", "replicates": 5},
            seed_range=[1, 2, 3],
            duration_seconds=5.0,
            dependencies={},
        )
        registry.register(prov_a, "/tmp/results_a")

        prov_b = ExperimentProvenance(
            experiment_id="exp-b",
            timestamp="2024-01-02T00:00:00Z",
            git_commit="def456",
            git_dirty=True,
            python_version="3.12.1",
            platform_info="Linux",
            autocog_version="0.1.0",
            config_yaml_path="test.yaml",
            config_yaml_hash="hash_b",
            config_resolved={"name": "test", "replicates": 10},
            seed_range=[1, 2, 3],
            duration_seconds=10.0,
            dependencies={},
        )
        registry.register(prov_b, "/tmp/results_b")

        # Compare
        comparison = registry.compare("exp-a", "exp-b")

        assert comparison["exp_a"]["provenance"]["experiment_id"] == "exp-a"
        assert comparison["exp_b"]["provenance"]["experiment_id"] == "exp-b"

        # Check config diff
        config_diff = comparison["config_diff"]
        assert "replicates" in config_diff["different_values"]
        assert config_diff["different_values"]["replicates"]["a"] == 5
        assert config_diff["different_values"]["replicates"]["b"] == 10

        # Check metadata diff
        metadata_diff = comparison["metadata_diff"]
        assert "git_commit" in metadata_diff["different_values"]


def test_registry_compare_nonexistent_experiment():
    """Test comparing with nonexistent experiment."""
    from src.experiments.registry import ExperimentRegistry

    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = Path(tmpdir) / "registry.jsonl"
        registry = ExperimentRegistry(str(registry_path))

        with pytest.raises(ValueError, match="not found"):
            registry.compare("nonexistent-a", "nonexistent-b")


def test_lock_file_generation():
    """Test lock file generation during experiment run."""
    from src.experiments.config import ExperimentCondition, ExperimentConfig
    from src.experiments.runner import ExperimentRunner

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a minimal experiment config
        config = ExperimentConfig(
            name="test_exp",
            description="Test experiment",
            base={
                "world_width": 32,  # Larger world to avoid spawn issues
                "world_height": 32,
                "max_ticks": 10,
                "num_agents": 2,
            },
            conditions=[
                ExperimentCondition(
                    name="baseline",
                    overrides={},
                )
            ],
            replicates=1,
            seed_start=42,
            metrics=["agents_alive_at_end"],
            output_dir=tmpdir,
            record_trajectories=False,
        )

        # Create config file for provenance
        config_path = Path(tmpdir) / "test_config.yaml"
        config_path.write_text("name: test_exp\n")

        runner = ExperimentRunner(
            config,
            config_yaml_path=str(config_path),
            registry_path=str(Path(tmpdir) / "registry.jsonl"),
        )

        # Run experiment
        runner.run_all()

        # Check lock file was created
        exp_id = runner.get_experiment_id()
        lock_file = Path(tmpdir) / f"experiment_{exp_id}.lock"
        assert lock_file.exists()

        # Verify lock file content
        content = lock_file.read_text()
        assert f"Experiment ID: {exp_id}" in content
        assert "Timestamp:" in content
        assert "Git Commit:" in content
        assert "Config Hash:" in content
        assert "Seeds:" in content
        assert "Duration:" in content


def test_registry_handles_corrupted_lines():
    """Test registry gracefully handles corrupted JSONL lines."""
    from src.experiments.registry import ExperimentRegistry

    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = Path(tmpdir) / "registry.jsonl"

        # Write some corrupted data
        with open(registry_path, "w") as f:
            f.write('{"valid": "json"}\n')
            f.write("invalid json line\n")  # Corrupted
            f.write('{"another": "valid"}\n')

        registry = ExperimentRegistry(str(registry_path))

        # Should only return valid entries
        experiments = registry.list_experiments()
        # Won't match our filter since we wrote raw JSON, but shouldn't crash
        assert isinstance(experiments, list)
