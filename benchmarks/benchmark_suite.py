"""Benchmark suite for AUTOCOG performance testing."""

import argparse
import time
import tracemalloc
from pathlib import Path

from src.config import SimulationConfig
from src.simulation.engine import SimulationEngine
from benchmarks.regression import RegressionDetector


BENCHMARK_CONFIGS = {
    "baseline_5_agents_500_ticks": {
        "num_agents": 5,
        "max_ticks": 500,
        "default_architecture": "reactive",
    },
    "scaling_10_agents": {
        "num_agents": 10,
        "max_ticks": 500,
        "default_architecture": "reactive",
    },
    "scaling_25_agents": {
        "num_agents": 25,
        "max_ticks": 200,
        "default_architecture": "reactive",
    },
    "architecture_cautious": {
        "num_agents": 5,
        "max_ticks": 500,
        "default_architecture": "cautious",
    },
    "architecture_dual_process": {
        "num_agents": 5,
        "max_ticks": 500,
        "default_architecture": "dual_process",
    },
    "planning_enabled": {
        "num_agents": 5,
        "max_ticks": 500,
        "default_architecture": "planning",
    },
    "theory_of_mind_enabled": {
        "num_agents": 5,
        "max_ticks": 200,
        "theory_of_mind_enabled": True,
    },
}


class BenchmarkRunner:
    """Run benchmark suite and track performance metrics."""

    def __init__(self, configs: dict | None = None):
        self.configs = configs or BENCHMARK_CONFIGS

    def run_single(self, name: str, config: dict) -> dict:
        """Run a single benchmark configuration.

        Args:
            name: Benchmark name
            config: Dict of SimulationConfig overrides

        Returns:
            Dict with timing and memory results
        """
        print(f"Running {name}...")

        # Build config
        sim_config = SimulationConfig(seed=42, **config)

        # Start memory tracking
        tracemalloc.start()

        # Run simulation
        engine = SimulationEngine(sim_config)
        engine.setup_multi_agent()

        start = time.perf_counter()
        ticks = 0

        while not engine.is_over():
            engine.step_all()
            ticks += 1

        end = time.perf_counter()

        # Get memory stats
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        total_seconds = end - start
        ms_per_tick = (total_seconds * 1000) / max(1, ticks)
        peak_memory_mb = peak_mem / (1024 * 1024)

        # Get performance summary from engine
        perf_summary = engine.perf_monitor.summary

        result = {
            "total_seconds": round(total_seconds, 2),
            "ticks": ticks,
            "ms_per_tick": round(ms_per_tick, 2),
            "peak_memory_mb": round(peak_memory_mb, 2),
            "perf_summary": perf_summary,
        }

        print(f"  [OK] {name}: {ticks} ticks in {total_seconds:.2f}s "
              f"({ms_per_tick:.2f}ms/tick, {peak_memory_mb:.2f}MB peak)")

        return result

    def run_all(self, only: str | None = None) -> dict:
        """Run all benchmarks (or single if --only specified).

        Args:
            only: Optional benchmark name to run in isolation

        Returns:
            Dict of benchmark_name -> result_dict
        """
        results = {}

        if only:
            if only not in self.configs:
                raise ValueError(f"Unknown benchmark: {only}")
            results[only] = self.run_single(only, self.configs[only])
        else:
            for name, config in self.configs.items():
                results[name] = self.run_single(name, config)

        return results

    def print_summary(self, results: dict, warnings: list[str] | None = None) -> None:
        """Print formatted summary of results.

        Args:
            results: Dict of benchmark results
            warnings: Optional list of regression warnings
        """
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)

        for name, result in results.items():
            status = "[OK]" if not warnings else "[WARN]"
            print(f"{status} {name}")
            print(f"   {result['ticks']} ticks in {result['total_seconds']}s")
            print(f"   {result['ms_per_tick']:.2f}ms/tick")
            print(f"   {result['peak_memory_mb']:.2f}MB peak memory")

        if warnings:
            print("\n" + "=" * 70)
            print("REGRESSIONS DETECTED")
            print("=" * 70)
            for warning in warnings:
                print(warning)


def main():
    """CLI entry point for benchmark suite."""
    parser = argparse.ArgumentParser(description="Run AUTOCOG benchmarks")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare against baseline",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Save results as new baseline",
    )
    parser.add_argument(
        "--only",
        type=str,
        help="Run only the specified benchmark",
    )
    parser.add_argument(
        "--baseline-path",
        type=str,
        default="benchmarks/baseline.json",
        help="Path to baseline file",
    )

    args = parser.parse_args()

    runner = BenchmarkRunner()
    results = runner.run_all(only=args.only)

    warnings = []
    if args.compare:
        detector = RegressionDetector(baseline_path=args.baseline_path)
        warnings = detector.check(results)

    runner.print_summary(results, warnings)

    if args.update_baseline:
        detector = RegressionDetector(baseline_path=args.baseline_path)
        detector.update_baseline(results)
        print(f"\n[OK] Baseline updated at {args.baseline_path}")


if __name__ == "__main__":
    main()
