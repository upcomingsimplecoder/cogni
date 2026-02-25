"""Benchmark suite runner for cross-scenario evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.experiments.analysis import ConditionSummary, ResultAnalyzer
from src.experiments.config import ExperimentConfig
from src.experiments.report import ReportGenerator
from src.experiments.runner import ExperimentRunner, RunResult


class BenchmarkRunner:
    """Runs benchmark suite and aggregates cross-scenario results.

    Orchestrates execution of multiple benchmark scenarios and generates
    comprehensive comparison reports.
    """

    def __init__(
        self,
        suite_dir: str,
        architectures: list[str] | None = None,
    ):
        """Initialize benchmark runner.

        Args:
            suite_dir: Path to directory containing benchmark YAML files
            architectures: Optional list of architectures to compare.
                          If provided, overrides the conditions in each YAML.
        """
        self.suite_dir = Path(suite_dir)
        self.architectures = architectures
        self._validate_suite_dir()

    def _validate_suite_dir(self) -> None:
        """Validate that suite directory exists and contains YAML files.

        Raises:
            ValueError: If directory doesn't exist or contains no YAML files
        """
        if not self.suite_dir.exists():
            raise ValueError(f"Suite directory not found: {self.suite_dir}")

        if not self.suite_dir.is_dir():
            raise ValueError(f"Suite path is not a directory: {self.suite_dir}")

        yaml_files = list(self.suite_dir.glob("*.yaml"))
        if not yaml_files:
            raise ValueError(f"No YAML files found in: {self.suite_dir}")

    def run_suite(
        self,
        progress_callback: Any = None,
    ) -> dict[str, list[RunResult]]:
        """Run all scenarios in the suite directory.

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            Dict mapping scenario_name -> list of RunResult
        """
        # Find all YAML files
        yaml_files = sorted(self.suite_dir.glob("*.yaml"))

        print(f"\n{'=' * 70}")
        print("AUTOCOG Benchmark Suite")
        print(f"{'=' * 70}")
        print(f"Suite directory: {self.suite_dir}")
        print(f"Scenarios: {len(yaml_files)}")
        if self.architectures:
            print(f"Architecture override: {', '.join(self.architectures)}")
        print(f"{'=' * 70}\n")

        all_results = {}

        for i, yaml_path in enumerate(yaml_files, 1):
            scenario_name = yaml_path.stem
            print(f"\n[{i}/{len(yaml_files)}] Running scenario: {scenario_name}")
            print(f"{'-' * 70}")

            try:
                results = self.run_scenario(str(yaml_path), progress_callback)
                all_results[scenario_name] = results
                print(f"[OK] Completed {scenario_name}: {len(results)} runs")
            except Exception as e:
                print(f"[FAIL] Error in {scenario_name}: {e}")
                # Continue with next scenario
                continue

        print(f"\n{'=' * 70}")
        print(f"Benchmark suite complete: {len(all_results)}/{len(yaml_files)} scenarios")
        print(f"{'=' * 70}\n")

        return all_results

    def run_scenario(
        self,
        yaml_path: str,
        progress_callback: Any = None,
    ) -> list[RunResult]:
        """Run a single benchmark scenario.

        Args:
            yaml_path: Path to scenario YAML file
            progress_callback: Optional callback for progress updates

        Returns:
            List of RunResult objects
        """
        # Load config
        config = ExperimentConfig.from_yaml(yaml_path)

        # Override architectures if specified
        if self.architectures:
            from src.experiments.config import ExperimentCondition

            config.conditions = [
                ExperimentCondition(name=arch, overrides={"default_architecture": arch})
                for arch in self.architectures
            ]

        # Create runner and execute
        runner = ExperimentRunner(config, config_yaml_path=yaml_path)

        def scenario_progress(condition: str, replicate: int, total: int) -> None:
            print(f"  {condition} [{replicate + 1}/{config.replicates}]", end="\r")
            if progress_callback:
                progress_callback(config.name, condition, replicate, total)

        results = runner.run_all(progress_callback=scenario_progress)
        print()  # Clear progress line

        return results

    def generate_report(
        self,
        results: dict[str, list[RunResult]],
        output_dir: str,
    ) -> str:
        """Generate a comprehensive benchmark report.

        Creates per-scenario summaries and cross-scenario comparison.

        Args:
            results: Dict mapping scenario_name -> list of RunResult
            output_dir: Directory to write report files

        Returns:
            Path to generated comparison report
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("\nGenerating benchmark report...")

        analyzer = ResultAnalyzer()
        generator = ReportGenerator()

        # Analyze each scenario
        all_summaries: dict[str, list[ConditionSummary]] = {}

        for scenario_name, scenario_results in results.items():
            # Generate per-scenario analysis
            summaries = analyzer.summarize(scenario_results)
            all_summaries[scenario_name] = summaries

            # Save individual scenario report
            scenario_report_path = output_path / f"{scenario_name}_report.md"

            # Create minimal config for report (we don't have full config anymore)
            from src.experiments.config import ExperimentConfig

            dummy_config = ExperimentConfig(
                name=scenario_name.replace("_", " ").title(),
                description=f"Benchmark scenario: {scenario_name}",
                base={},
                conditions=[],
                replicates=len(
                    [
                        r
                        for r in scenario_results
                        if r.condition_name == scenario_results[0].condition_name
                    ]
                ),
            )

            generator.to_markdown(dummy_config, summaries, str(scenario_report_path))

            # Save CSV
            csv_path = output_path / f"{scenario_name}_results.csv"
            generator.to_csv(scenario_results, str(csv_path))

        # Generate cross-scenario comparison
        comparison_path = output_path / "benchmark_comparison.md"
        generator.to_comparison_markdown(all_summaries, str(comparison_path))

        # Save summary JSON
        summary_json_path = output_path / "benchmark_summary.json"
        self._save_summary_json(all_summaries, summary_json_path)

        print(f"[OK] Report generated: {comparison_path}")
        print(f"  Individual reports: {output_path}/*_report.md")
        print(f"  CSV files: {output_path}/*_results.csv")
        print(f"  Summary JSON: {summary_json_path}")

        return str(comparison_path)

    def _save_summary_json(
        self,
        all_summaries: dict[str, list[ConditionSummary]],
        output_path: Path,
    ) -> None:
        """Save benchmark summary as JSON.

        Args:
            all_summaries: Dict mapping scenario_name -> list of ConditionSummary
            output_path: Path to output JSON file
        """
        data = {}
        for scenario_name, summaries in all_summaries.items():
            data[scenario_name] = [
                {
                    "condition": s.condition_name,
                    "n": s.n,
                    "metrics": s.metrics,
                }
                for s in summaries
            ]

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
