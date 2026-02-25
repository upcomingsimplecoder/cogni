"""Tests for benchmarking harness."""

import json
import tempfile
from pathlib import Path

from benchmarks.benchmark_suite import BENCHMARK_CONFIGS, BenchmarkRunner
from benchmarks.regression import RegressionDetector


def test_benchmark_runner_runs_single_config():
    """Test that BenchmarkRunner can run a single benchmark configuration."""
    # Use a minimal config for fast testing
    test_config = {
        "test_minimal": {
            "num_agents": 2,
            "max_ticks": 10,
            "default_architecture": "reactive",
        }
    }

    runner = BenchmarkRunner(configs=test_config)
    results = runner.run_all(only="test_minimal")

    assert "test_minimal" in results
    result = results["test_minimal"]

    # Check that result has expected fields
    assert "total_seconds" in result
    assert "ticks" in result
    assert "ms_per_tick" in result
    assert "peak_memory_mb" in result
    assert "perf_summary" in result

    # Check values are reasonable
    assert result["ticks"] > 0
    assert result["total_seconds"] > 0
    assert result["ms_per_tick"] > 0
    assert result["peak_memory_mb"] > 0


def test_benchmark_result_has_expected_fields():
    """Test that benchmark results contain all required fields."""
    test_config = {
        "test_fields": {
            "num_agents": 2,
            "max_ticks": 5,
            "default_architecture": "reactive",
        }
    }

    runner = BenchmarkRunner(configs=test_config)
    result = runner.run_single("test_fields", test_config["test_fields"])

    required_fields = [
        "total_seconds",
        "ticks",
        "ms_per_tick",
        "peak_memory_mb",
        "perf_summary",
    ]

    for field in required_fields:
        assert field in result, f"Missing required field: {field}"

    # Check types
    assert isinstance(result["total_seconds"], (int, float))
    assert isinstance(result["ticks"], int)
    assert isinstance(result["ms_per_tick"], (int, float))
    assert isinstance(result["peak_memory_mb"], (int, float))
    assert isinstance(result["perf_summary"], dict)


def test_regression_detector_detects_regression():
    """Test that RegressionDetector identifies performance regressions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_path = Path(tmpdir) / "baseline.json"

        # Create baseline with good performance
        baseline = {
            "test_benchmark": {
                "total_seconds": 1.0,
                "ticks": 100,
                "ms_per_tick": 10.0,
                "peak_memory_mb": 50.0,
                "perf_summary": {},
            }
        }

        with open(baseline_path, "w") as f:
            json.dump(baseline, f)

        detector = RegressionDetector(baseline_path=str(baseline_path))

        # Current results show 30% slowdown (10ms -> 13ms)
        current_results = {
            "test_benchmark": {
                "total_seconds": 1.3,
                "ticks": 100,
                "ms_per_tick": 13.0,
                "peak_memory_mb": 50.0,
                "perf_summary": {},
            }
        }

        warnings = detector.check(current_results)

        assert len(warnings) == 1
        assert "test_benchmark" in warnings[0]
        assert "+30.0%" in warnings[0]


def test_regression_detector_passes_within_threshold():
    """Test that RegressionDetector doesn't flag changes within threshold."""
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_path = Path(tmpdir) / "baseline.json"

        # Create baseline
        baseline = {
            "test_benchmark": {
                "total_seconds": 1.0,
                "ticks": 100,
                "ms_per_tick": 10.0,
                "peak_memory_mb": 50.0,
                "perf_summary": {},
            }
        }

        with open(baseline_path, "w") as f:
            json.dump(baseline, f)

        detector = RegressionDetector(baseline_path=str(baseline_path))

        # Current results show 15% slowdown (below 20% threshold)
        current_results = {
            "test_benchmark": {
                "total_seconds": 1.15,
                "ticks": 100,
                "ms_per_tick": 11.5,
                "peak_memory_mb": 50.0,
                "perf_summary": {},
            }
        }

        warnings = detector.check(current_results)

        assert len(warnings) == 0


def test_regression_detector_handles_missing_baseline():
    """Test that RegressionDetector handles missing baseline gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_path = Path(tmpdir) / "nonexistent_baseline.json"

        detector = RegressionDetector(baseline_path=str(baseline_path))

        # Should not crash with missing baseline
        current_results = {
            "test_benchmark": {
                "total_seconds": 1.0,
                "ticks": 100,
                "ms_per_tick": 10.0,
                "peak_memory_mb": 50.0,
                "perf_summary": {},
            }
        }

        warnings = detector.check(current_results)

        # No baseline means no warnings
        assert len(warnings) == 0

        # Should be able to save new baseline
        detector.update_baseline(current_results)
        assert baseline_path.exists()

        # Load and verify
        with open(baseline_path) as f:
            saved_baseline = json.load(f)

        assert "test_benchmark" in saved_baseline
        assert saved_baseline["test_benchmark"]["ms_per_tick"] == 10.0


def test_benchmark_configs_are_valid():
    """Test that all predefined BENCHMARK_CONFIGS are valid."""
    # Just check that the configs dict is not empty and has expected structure
    assert len(BENCHMARK_CONFIGS) > 0

    required_configs = [
        "baseline_5_agents_500_ticks",
        "scaling_10_agents",
        "scaling_25_agents",
    ]

    for config_name in required_configs:
        assert config_name in BENCHMARK_CONFIGS

        config = BENCHMARK_CONFIGS[config_name]
        assert "num_agents" in config
        assert "max_ticks" in config
        assert config["num_agents"] > 0
        assert config["max_ticks"] > 0
