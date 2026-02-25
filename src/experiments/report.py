"""Report generation for experiment results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.experiments.analysis import ConditionSummary
from src.experiments.config import ExperimentConfig
from src.experiments.runner import RunResult


class ReportGenerator:
    """Generates reports in various formats.

    Supports markdown tables, CSV, and JSON output.
    """

    def to_markdown(
        self,
        config: ExperimentConfig,
        summaries: list[ConditionSummary],
        output_path: str,
    ) -> None:
        """Generate markdown report with tables.

        Args:
            config: Experiment configuration
            summaries: List of condition summaries
            output_path: Path to output markdown file
        """
        lines = []
        lines.append(f"# {config.name}")
        lines.append("")
        lines.append(f"**Description:** {config.description}")
        lines.append("")
        lines.append(f"**Replicates:** {config.replicates}")
        lines.append(f"**Seed Start:** {config.seed_start}")
        lines.append("")

        # Get all metric names
        all_metrics: set[str] = set()
        for summary in summaries:
            all_metrics.update(summary.metrics.keys())
        sorted_metrics = sorted(list(all_metrics))

        # Create a table for each metric
        for metric in sorted_metrics:
            lines.append(f"## {metric}")
            lines.append("")

            # Table header
            lines.append("| Condition | N | Mean | Std | Min | Max | 95% CI |")
            lines.append("|-----------|---|------|-----|-----|-----|--------|")

            # Table rows
            for summary in summaries:
                if metric not in summary.metrics:
                    continue

                stats = summary.metrics[metric]
                mean = stats["mean"]
                std = stats["std"]
                min_val = stats["min"]
                max_val = stats["max"]
                ci_lower = stats["ci_95_lower"]
                ci_upper = stats["ci_95_upper"]

                lines.append(
                    f"| {summary.condition_name} | {summary.n} | "
                    f"{mean:.2f} | {std:.2f} | {min_val:.2f} | {max_val:.2f} | "
                    f"[{ci_lower:.2f}, {ci_upper:.2f}] |"
                )

            lines.append("")

        # Write to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(lines))

    def to_csv(self, results: list[RunResult], output_path: str) -> None:
        """Generate flat CSV with one row per run.

        Args:
            results: List of run results
            output_path: Path to output CSV file
        """
        if not results:
            # Write empty file
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write("")
            return

        # Get all metric names
        all_metrics: set[str] = set()
        for result in results:
            all_metrics.update(result.metrics.keys())
        sorted_metrics = sorted(list(all_metrics))

        # Write CSV
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            fieldnames = [
                "condition",
                "replicate",
                "seed",
                "duration_seconds",
            ]
            fieldnames.extend(sorted_metrics)

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                row = {
                    "condition": result.condition_name,
                    "replicate": result.replicate,
                    "seed": result.seed,
                    "duration_seconds": result.duration_seconds,
                }
                for metric in sorted_metrics:
                    row[metric] = result.metrics.get(metric, 0.0)

                writer.writerow(row)

    def to_json(self, summaries: list[ConditionSummary], output_path: str) -> None:
        """Generate machine-readable JSON.

        Args:
            summaries: List of condition summaries
            output_path: Path to output JSON file
        """
        data = []
        for summary in summaries:
            data.append(
                {
                    "condition": summary.condition_name,
                    "n": summary.n,
                    "metrics": summary.metrics,
                }
            )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def to_comparison_markdown(
        self,
        all_results: dict[str, list[ConditionSummary]],
        output_path: str,
    ) -> None:
        """Generate cross-scenario comparison report.

        Creates a markdown report with:
        - Overview table: scenarios × conditions × key metrics
        - Per-scenario detail sections
        - Architecture ranking (which arch wins most scenarios)
        - Summary findings

        Args:
            all_results: Dict mapping scenario_name -> list of ConditionSummary
            output_path: Path to output markdown file
        """
        lines = []
        lines.append("# AUTOCOG Benchmark Results")
        lines.append("")
        lines.append(f"**Scenarios:** {len(all_results)}")
        lines.append("")

        # Get all unique conditions (architectures)
        all_conditions: set[str] = set()
        for summaries in all_results.values():
            for summary in summaries:
                all_conditions.add(summary.condition_name)
        sorted_conditions = sorted(list(all_conditions))

        lines.append(f"**Architectures:** {', '.join(sorted_conditions)}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Architecture ranking section
        lines.append("## Architecture Rankings")
        lines.append("")
        ranking_data = self._compute_architecture_rankings(all_results, sorted_conditions)

        lines.append("### Wins by Scenario")
        lines.append("")
        lines.append(
            "Architecture wins are determined by the highest mean value across all metrics "
            "per scenario."
        )
        lines.append("")
        lines.append("| Architecture | Wins | Win Rate |")
        lines.append("|--------------|------|----------|")

        total_scenarios = len(all_results)
        for arch in sorted(
            ranking_data["wins"].keys(), key=lambda x: ranking_data["wins"][x], reverse=True
        ):
            wins = ranking_data["wins"][arch]
            win_rate = wins / total_scenarios if total_scenarios > 0 else 0.0
            lines.append(f"| {arch} | {wins} | {win_rate:.1%} |")

        lines.append("")

        # Per-scenario win breakdown
        lines.append("### Scenario-by-Scenario Winners")
        lines.append("")
        lines.append("| Scenario | Winner | Best Metric | Value |")
        lines.append("|----------|--------|-------------|-------|")

        for scenario_name in sorted(all_results.keys()):
            winner_info = ranking_data["scenario_winners"][scenario_name]
            lines.append(
                f"| {scenario_name} | {winner_info['condition']} | "
                f"{winner_info['metric']} | {winner_info['value']:.2f} |"
            )

        lines.append("")
        lines.append("---")
        lines.append("")

        # Per-scenario detailed results
        lines.append("## Detailed Results by Scenario")
        lines.append("")

        for scenario_name in sorted(all_results.keys()):
            summaries = all_results[scenario_name]

            lines.append(f"### {scenario_name.replace('_', ' ').title()}")
            lines.append("")

            # Get all metrics for this scenario
            scenario_metrics: set[str] = set()
            for summary in summaries:
                scenario_metrics.update(summary.metrics.keys())
            sorted_scenario_metrics = sorted(list(scenario_metrics))

            # Create table for each metric
            for metric in sorted_scenario_metrics:
                lines.append(f"**{metric}**")
                lines.append("")
                lines.append("| Condition | N | Mean | Std | 95% CI |")
                lines.append("|-----------|---|------|-----|--------|")

                for summary in summaries:
                    if metric not in summary.metrics:
                        continue

                    stats = summary.metrics[metric]
                    lines.append(
                        f"| {summary.condition_name} | {summary.n} | "
                        f"{stats['mean']:.2f} | {stats['std']:.2f} | "
                        f"[{stats['ci_95_lower']:.2f}, {stats['ci_95_upper']:.2f}] |"
                    )

                lines.append("")

            lines.append("---")
            lines.append("")

        # Write to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(lines))

    def _compute_architecture_rankings(
        self,
        all_results: dict[str, list[ConditionSummary]],
        all_conditions: list[str],
    ) -> dict[str, Any]:
        """Compute architecture rankings across scenarios.

        Args:
            all_results: Dict mapping scenario_name -> list of ConditionSummary
            all_conditions: List of all unique condition names

        Returns:
            Dict with "wins" (arch -> count) and "scenario_winners" (scenario -> winner info)
        """
        wins = {arch: 0 for arch in all_conditions}
        scenario_winners = {}

        for scenario_name, summaries in all_results.items():
            # For each scenario, find the architecture with highest mean across all metrics
            best_condition = None
            best_metric = None
            best_value = float("-inf")

            for summary in summaries:
                # Average all metric means for this condition
                if not summary.metrics:
                    continue

                # Find the single best metric value for this condition
                for metric_name, stats in summary.metrics.items():
                    if stats["mean"] > best_value:
                        best_value = stats["mean"]
                        best_metric = metric_name
                        best_condition = summary.condition_name

            if best_condition:
                wins[best_condition] += 1
                scenario_winners[scenario_name] = {
                    "condition": best_condition,
                    "metric": best_metric,
                    "value": best_value,
                }

        return {
            "wins": wins,
            "scenario_winners": scenario_winners,
        }
