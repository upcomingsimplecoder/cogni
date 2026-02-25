"""Tests for experiment runner system."""

from __future__ import annotations

import csv
import json

from src.experiments.analysis import ConditionSummary, ResultAnalyzer
from src.experiments.config import ExperimentCondition, ExperimentConfig
from src.experiments.report import ReportGenerator
from src.experiments.runner import ExperimentRunner, RunResult


class TestExperimentConfig:
    """Tests for ExperimentConfig."""

    def test_from_dict_creates_valid_config(self):
        """Test that from_dict creates a valid configuration."""
        data = {
            "name": "Test Experiment",
            "description": "A test experiment",
            "base": {"world_width": 16, "world_height": 16},
            "conditions": [
                {"name": "condition_a", "overrides": {"num_agents": 2}},
                {"name": "condition_b", "overrides": {"num_agents": 4}},
            ],
            "replicates": 3,
            "seed_start": 100,
            "metrics": ["agents_alive_at_end"],
            "output_dir": "data/test",
            "formats": ["csv"],
        }

        config = ExperimentConfig.from_dict(data)

        assert config.name == "Test Experiment"
        assert config.description == "A test experiment"
        assert config.base == {"world_width": 16, "world_height": 16}
        assert len(config.conditions) == 2
        assert config.conditions[0].name == "condition_a"
        assert config.conditions[0].overrides == {"num_agents": 2}
        assert config.replicates == 3
        assert config.seed_start == 100
        assert config.metrics == ["agents_alive_at_end"]

    def test_from_dict_uses_defaults(self):
        """Test that from_dict uses default values when not specified."""
        data = {
            "name": "Minimal",
            "conditions": [{"name": "default", "overrides": {}}],
        }

        config = ExperimentConfig.from_dict(data)

        assert config.description == ""
        assert config.base == {}
        assert config.replicates == 5
        assert config.seed_start == 42
        assert "agents_alive_at_end" in config.metrics

    def test_expand_conditions(self):
        """Test condition expansion (currently identity)."""
        config = ExperimentConfig(
            name="Test",
            description="",
            base={},
            conditions=[
                ExperimentCondition("a", {}),
                ExperimentCondition("b", {}),
            ],
        )

        expanded = config.expand_conditions()

        assert len(expanded) == 2
        assert expanded[0].name == "a"
        assert expanded[1].name == "b"


class TestExperimentRunner:
    """Tests for ExperimentRunner."""

    def test_run_single_completes_simulation(self):
        """Test that _run_single completes a simulation and returns metrics."""
        config = ExperimentConfig(
            name="Single Run Test",
            description="",
            base={
                "world_width": 16,
                "world_height": 16,
                "max_ticks": 20,
                "num_agents": 2,
            },
            conditions=[ExperimentCondition("test", {})],
            replicates=1,
            metrics=["agents_alive_at_end", "avg_survival_ticks"],
        )

        runner = ExperimentRunner(config)
        result = runner._run_single(config.conditions[0], 0, 42)

        assert result.condition_name == "test"
        assert result.replicate == 0
        assert result.seed == 42
        assert "agents_alive_at_end" in result.metrics
        assert "avg_survival_ticks" in result.metrics
        assert result.duration_seconds > 0

    def test_run_all_executes_all_combinations(self):
        """Test that run_all executes all conditions × replicates."""
        config = ExperimentConfig(
            name="Multi Run Test",
            description="",
            base={
                "world_width": 16,
                "world_height": 16,
                "max_ticks": 10,
                "num_agents": 2,
            },
            conditions=[
                ExperimentCondition("cond_a", {}),
                ExperimentCondition("cond_b", {}),
            ],
            replicates=2,
            metrics=["agents_alive_at_end"],
        )

        runner = ExperimentRunner(config)
        results = runner.run_all()

        # Should have 2 conditions × 2 replicates = 4 results
        assert len(results) == 4

        # Check condition names are correct
        cond_names = [r.condition_name for r in results]
        assert cond_names.count("cond_a") == 2
        assert cond_names.count("cond_b") == 2

        # Check replicate numbers
        for cond in ["cond_a", "cond_b"]:
            cond_results = [r for r in results if r.condition_name == cond]
            replicates = [r.replicate for r in cond_results]
            assert 0 in replicates
            assert 1 in replicates

    def test_metric_extraction_agents_alive(self):
        """Test agents_alive_at_end metric extraction."""
        config = ExperimentConfig(
            name="Metric Test",
            description="",
            base={
                "world_width": 16,
                "world_height": 16,
                "max_ticks": 10,
                "num_agents": 3,
            },
            conditions=[ExperimentCondition("test", {})],
            replicates=1,
            metrics=["agents_alive_at_end"],
        )

        runner = ExperimentRunner(config)
        result = runner._run_single(config.conditions[0], 0, 42)

        # Should have extracted the metric
        assert "agents_alive_at_end" in result.metrics
        # Value should be non-negative
        assert result.metrics["agents_alive_at_end"] >= 0

    def test_metric_extraction_avg_survival(self):
        """Test avg_survival_ticks metric works correctly."""
        config = ExperimentConfig(
            name="Survival Test",
            description="",
            base={
                "world_width": 16,
                "world_height": 16,
                "max_ticks": 15,
                "num_agents": 2,
            },
            conditions=[ExperimentCondition("test", {})],
            replicates=1,
            metrics=["avg_survival_ticks"],
        )

        runner = ExperimentRunner(config)
        result = runner._run_single(config.conditions[0], 0, 42)

        assert "avg_survival_ticks" in result.metrics
        assert result.metrics["avg_survival_ticks"] > 0


class TestResultAnalyzer:
    """Tests for ResultAnalyzer."""

    def test_summarize_computes_correct_mean(self):
        """Test that summarize computes correct mean values."""
        results = [
            RunResult("cond_a", 0, 42, {"metric_x": 10.0}, 1.0),
            RunResult("cond_a", 1, 43, {"metric_x": 20.0}, 1.0),
            RunResult("cond_a", 2, 44, {"metric_x": 30.0}, 1.0),
        ]

        analyzer = ResultAnalyzer()
        summaries = analyzer.summarize(results)

        assert len(summaries) == 1
        assert summaries[0].condition_name == "cond_a"
        assert summaries[0].n == 3
        assert summaries[0].metrics["metric_x"]["mean"] == 20.0

    def test_summarize_computes_correct_std(self):
        """Test that summarize computes correct standard deviation."""
        results = [
            RunResult("cond_a", 0, 42, {"metric_x": 10.0}, 1.0),
            RunResult("cond_a", 1, 43, {"metric_x": 20.0}, 1.0),
            RunResult("cond_a", 2, 44, {"metric_x": 30.0}, 1.0),
        ]

        analyzer = ResultAnalyzer()
        summaries = analyzer.summarize(results)

        # Sample std of [10, 20, 30] = 10.0
        assert abs(summaries[0].metrics["metric_x"]["std"] - 10.0) < 0.01

    def test_summarize_handles_multiple_conditions(self):
        """Test that summarize handles multiple conditions correctly."""
        results = [
            RunResult("cond_a", 0, 42, {"metric_x": 10.0}, 1.0),
            RunResult("cond_a", 1, 43, {"metric_x": 20.0}, 1.0),
            RunResult("cond_b", 0, 42, {"metric_x": 50.0}, 1.0),
            RunResult("cond_b", 1, 43, {"metric_x": 60.0}, 1.0),
        ]

        analyzer = ResultAnalyzer()
        summaries = analyzer.summarize(results)

        assert len(summaries) == 2
        cond_a = next(s for s in summaries if s.condition_name == "cond_a")
        cond_b = next(s for s in summaries if s.condition_name == "cond_b")

        assert cond_a.metrics["metric_x"]["mean"] == 15.0
        assert cond_b.metrics["metric_x"]["mean"] == 55.0

    def test_pairwise_comparison_identifies_differences(self):
        """Test that pairwise_comparison computes differences correctly."""
        results = [
            RunResult("cond_a", 0, 42, {"metric_x": 10.0}, 1.0),
            RunResult("cond_a", 1, 43, {"metric_x": 20.0}, 1.0),
            RunResult("cond_b", 0, 42, {"metric_x": 50.0}, 1.0),
            RunResult("cond_b", 1, 43, {"metric_x": 60.0}, 1.0),
        ]

        analyzer = ResultAnalyzer()
        comparisons = analyzer.pairwise_comparison(results)

        assert len(comparisons) == 1  # Only one pair
        comp = comparisons[0]

        assert comp["condition_a"] == "cond_a"
        assert comp["condition_b"] == "cond_b"
        assert comp["metric"] == "metric_x"
        # mean_b - mean_a = 55 - 15 = 40
        assert comp["mean_diff"] == 40.0
        # Effect size should be large (40 / pooled_std)
        assert abs(comp["effect_size"]) > 0


class TestReportGenerator:
    """Tests for ReportGenerator."""

    def test_to_csv_produces_valid_csv(self, tmp_path):
        """Test that to_csv produces valid CSV output."""
        results = [
            RunResult("cond_a", 0, 42, {"metric_x": 10.0, "metric_y": 5.0}, 1.0),
            RunResult("cond_a", 1, 43, {"metric_x": 20.0, "metric_y": 15.0}, 1.5),
        ]

        generator = ReportGenerator()
        output_file = tmp_path / "results.csv"
        generator.to_csv(results, str(output_file))

        # Read and validate CSV
        assert output_file.exists()
        with open(output_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["condition"] == "cond_a"
        assert float(rows[0]["metric_x"]) == 10.0
        assert float(rows[1]["metric_x"]) == 20.0

    def test_to_markdown_produces_valid_markdown(self, tmp_path):
        """Test that to_markdown produces valid markdown table."""
        config = ExperimentConfig(
            name="Test Experiment",
            description="A test",
            base={},
            conditions=[ExperimentCondition("cond_a", {})],
            metrics=["metric_x"],
        )

        summaries = [
            ConditionSummary(
                condition_name="cond_a",
                n=3,
                metrics={
                    "metric_x": {
                        "mean": 20.0,
                        "std": 10.0,
                        "min": 10.0,
                        "max": 30.0,
                        "ci_95_lower": 10.0,
                        "ci_95_upper": 30.0,
                    }
                },
            )
        ]

        generator = ReportGenerator()
        output_file = tmp_path / "report.md"
        generator.to_markdown(config, summaries, str(output_file))

        # Read and validate markdown
        assert output_file.exists()
        content = output_file.read_text()

        assert "# Test Experiment" in content
        assert "## metric_x" in content
        assert "cond_a" in content
        assert "20.00" in content  # Mean value

    def test_to_json_produces_valid_json(self, tmp_path):
        """Test that to_json produces valid JSON output."""
        summaries = [
            ConditionSummary(
                condition_name="cond_a",
                n=2,
                metrics={
                    "metric_x": {
                        "mean": 15.0,
                        "std": 5.0,
                        "min": 10.0,
                        "max": 20.0,
                        "ci_95_lower": 10.0,
                        "ci_95_upper": 20.0,
                    }
                },
            )
        ]

        generator = ReportGenerator()
        output_file = tmp_path / "summary.json"
        generator.to_json(summaries, str(output_file))

        # Read and validate JSON
        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["condition"] == "cond_a"
        assert data[0]["n"] == 2
        assert data[0]["metrics"]["metric_x"]["mean"] == 15.0


class TestIntegration:
    """Integration tests for full experiment workflow."""

    def test_full_workflow_two_conditions(self, tmp_path):
        """Test full workflow: run 2 conditions × 2 replicates, verify results."""
        config = ExperimentConfig(
            name="Integration Test",
            description="Full workflow test",
            base={
                "world_width": 16,
                "world_height": 16,
                "max_ticks": 20,
                "num_agents": 2,
            },
            conditions=[
                ExperimentCondition("small", {"num_agents": 2}),
                ExperimentCondition("large", {"num_agents": 3}),
            ],
            replicates=2,
            seed_start=100,
            metrics=["agents_alive_at_end", "avg_survival_ticks"],
            output_dir=str(tmp_path / "experiment"),
            formats=["csv", "markdown", "json"],
        )

        # Run experiment
        runner = ExperimentRunner(config)
        results = runner.run_all()

        # Verify all results present
        assert len(results) == 4  # 2 conditions × 2 replicates
        assert all(r.condition_name in ["small", "large"] for r in results)
        assert all(r.replicate in [0, 1] for r in results)

        # Analyze results
        analyzer = ResultAnalyzer()
        summaries = analyzer.summarize(results)
        assert len(summaries) == 2

        # Generate reports
        generator = ReportGenerator()
        generator.to_csv(results, str(tmp_path / "results.csv"))
        generator.to_markdown(config, summaries, str(tmp_path / "report.md"))
        generator.to_json(summaries, str(tmp_path / "summary.json"))

        # Verify files exist
        assert (tmp_path / "results.csv").exists()
        assert (tmp_path / "report.md").exists()
        assert (tmp_path / "summary.json").exists()
