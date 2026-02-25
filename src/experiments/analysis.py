"""Statistical analysis of experiment results."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from src.experiments.runner import RunResult


@dataclass
class ConditionSummary:
    """Summary statistics for a single condition."""

    condition_name: str
    n: int
    metrics: dict[str, dict]  # metric_name → {"mean", "std", "min", "max", "ci_95"}


class ResultAnalyzer:
    """Analyzes experiment results with summary statistics.

    Uses standard library statistics (no scipy dependency).
    """

    def summarize(self, results: list[RunResult]) -> list[ConditionSummary]:
        """Compute summary statistics per condition.

        Args:
            results: List of RunResult objects

        Returns:
            List of ConditionSummary objects
        """
        # Group results by condition
        by_condition: dict[str, list[RunResult]] = defaultdict(list)
        for result in results:
            by_condition[result.condition_name].append(result)

        summaries = []
        for condition_name, cond_results in by_condition.items():
            # Extract all metric names
            all_metrics: set[str] = set()
            for result in cond_results:
                all_metrics.update(result.metrics.keys())

            # Compute statistics for each metric
            metric_stats = {}
            for metric_name in all_metrics:
                values = [r.metrics[metric_name] for r in cond_results]
                metric_stats[metric_name] = self._compute_stats(values)

            summaries.append(
                ConditionSummary(
                    condition_name=condition_name,
                    n=len(cond_results),
                    metrics=metric_stats,
                )
            )

        return summaries

    def pairwise_comparison(self, results: list[RunResult]) -> list[dict]:
        """Compare pairs of conditions with effect sizes.

        Args:
            results: List of RunResult objects

        Returns:
            List of comparison dicts with keys:
            - condition_a: First condition name
            - condition_b: Second condition name
            - metric: Metric name
            - mean_diff: Difference in means (b - a)
            - effect_size: Cohen's d
        """
        # Group by condition
        by_condition: dict[str, list[RunResult]] = defaultdict(list)
        for result in results:
            by_condition[result.condition_name].append(result)

        conditions = list(by_condition.keys())
        comparisons = []

        # Get all metric names
        all_metrics: set[str] = set()
        for result in results:
            all_metrics.update(result.metrics.keys())

        # Compare all pairs
        for i, cond_a in enumerate(conditions):
            for cond_b in conditions[i + 1 :]:
                results_a = by_condition[cond_a]
                results_b = by_condition[cond_b]

                for metric in all_metrics:
                    values_a = [r.metrics[metric] for r in results_a]
                    values_b = [r.metrics[metric] for r in results_b]

                    mean_a = sum(values_a) / len(values_a)
                    mean_b = sum(values_b) / len(values_b)
                    mean_diff = mean_b - mean_a

                    # Cohen's d: (mean_b - mean_a) / pooled_std
                    std_a = self._std(values_a)
                    std_b = self._std(values_b)
                    pooled_std = math.sqrt(
                        ((len(values_a) - 1) * std_a**2 + (len(values_b) - 1) * std_b**2)
                        / (len(values_a) + len(values_b) - 2)
                    )
                    effect_size = mean_diff / pooled_std if pooled_std > 0 else 0.0

                    comparisons.append(
                        {
                            "condition_a": cond_a,
                            "condition_b": cond_b,
                            "metric": metric,
                            "mean_diff": mean_diff,
                            "effect_size": effect_size,
                        }
                    )

        return comparisons

    def _compute_stats(self, values: list[float]) -> dict[str, float]:
        """Compute summary statistics for a list of values.

        Args:
            values: List of numeric values

        Returns:
            Dict with keys: mean, std, min, max, ci_95_lower, ci_95_upper
        """
        n = len(values)
        if n == 0:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "ci_95_lower": 0.0,
                "ci_95_upper": 0.0,
            }

        mean = sum(values) / n
        std = self._std(values)
        ci_lower, ci_upper = self._confidence_interval(values)

        return {
            "mean": mean,
            "std": std,
            "min": min(values),
            "max": max(values),
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
        }

    def _std(self, values: list[float]) -> float:
        """Calculate sample standard deviation.

        Args:
            values: List of numeric values

        Returns:
            Standard deviation
        """
        n = len(values)
        if n < 2:
            return 0.0

        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        return math.sqrt(variance)

    def _confidence_interval(
        self, values: list[float], confidence: float = 0.95
    ) -> tuple[float, float]:
        """Compute confidence interval using t-distribution.

        Args:
            values: List of numeric values
            confidence: Confidence level (default 0.95)

        Returns:
            (lower, upper) confidence interval bounds
        """
        n = len(values)
        if n < 2:
            mean = values[0] if values else 0.0
            return (mean, mean)

        mean = sum(values) / n
        std = self._std(values)

        # Degrees of freedom
        df = n - 1

        # t-value for 95% CI (approximate using normal for large n)
        # For small n, use lookup table
        t_values = {
            1: 12.706,
            2: 4.303,
            3: 3.182,
            4: 2.776,
            5: 2.571,
            6: 2.447,
            7: 2.365,
            8: 2.306,
            9: 2.262,
            10: 2.228,
            15: 2.131,
            20: 2.086,
            30: 2.042,
            40: 2.021,
            50: 2.009,
            100: 1.984,
        }

        if df in t_values:
            t_val = t_values[df]
        elif df < 1:
            t_val = 12.706
        elif df > 100:
            t_val = 1.96  # Normal approximation
        else:
            # Interpolate
            keys = sorted(t_values.keys())
            lower_key = max(k for k in keys if k <= df)
            upper_key = min(k for k in keys if k >= df)
            if lower_key == upper_key:
                t_val = t_values[lower_key]
            else:
                # Linear interpolation
                frac = (df - lower_key) / (upper_key - lower_key)
                t_val = t_values[lower_key] * (1 - frac) + t_values[upper_key] * frac

        margin = t_val * std / math.sqrt(n)
        return (mean - margin, mean + margin)
