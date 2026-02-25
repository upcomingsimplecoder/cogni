"""Regression detection for benchmark results."""

import json
from pathlib import Path


class RegressionDetector:
    """Compare current benchmarks against stored baseline."""

    def __init__(self, baseline_path: str = "benchmarks/baseline.json"):
        self.baseline_path = Path(baseline_path)
        self.baseline = self._load_baseline()

    def _load_baseline(self) -> dict:
        """Load baseline from disk."""
        if not self.baseline_path.exists():
            return {}
        with open(self.baseline_path, "r") as f:
            return json.load(f)

    def check(self, results: dict) -> list[str]:
        """Return list of warnings for regressions > 20%.

        Args:
            results: Dict of benchmark_name -> result_dict with 'ms_per_tick' field

        Returns:
            List of warning strings describing regressions
        """
        warnings = []
        if not self.baseline:
            return warnings

        for name, current_result in results.items():
            if name not in self.baseline:
                continue

            baseline_result = self.baseline[name]
            current_ms = current_result.get("ms_per_tick", 0)
            baseline_ms = baseline_result.get("ms_per_tick", 0)

            if baseline_ms == 0:
                continue

            pct_change = ((current_ms - baseline_ms) / baseline_ms) * 100

            if pct_change > 20:
                warnings.append(
                    f"[WARN] {name}: {pct_change:+.1f}% slower "
                    f"(baseline: {baseline_ms:.2f}ms, current: {current_ms:.2f}ms)"
                )

        return warnings

    def update_baseline(self, results: dict, path: str | None = None) -> None:
        """Save current results as new baseline.

        Args:
            results: Dict of benchmark_name -> result_dict
            path: Optional custom path (defaults to self.baseline_path)
        """
        save_path = Path(path) if path else self.baseline_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
