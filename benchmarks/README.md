# AUTOCOG Benchmarking Harness

Performance benchmarking and regression detection for AUTOCOG simulations.

## Quick Start

Run all benchmarks:
```bash
python -m benchmarks.benchmark_suite
```

Run a single benchmark:
```bash
python -m benchmarks.benchmark_suite --only baseline_5_agents_500_ticks
```

Compare against baseline and detect regressions:
```bash
python -m benchmarks.benchmark_suite --compare
```

Update baseline with current results:
```bash
python -m benchmarks.benchmark_suite --update-baseline
```

## Benchmark Configurations

- `baseline_5_agents_500_ticks` - 5 reactive agents, 500 tick max
- `scaling_10_agents` - 10 reactive agents, 500 tick max
- `scaling_25_agents` - 25 reactive agents, 200 tick max
- `architecture_cautious` - 5 cautious architecture agents
- `architecture_dual_process` - 5 dual-process architecture agents
- `planning_enabled` - 5 planning architecture agents
- `theory_of_mind_enabled` - 5 agents with Theory of Mind

## Metrics Tracked

For each benchmark:
- `total_seconds` - Wall-clock time
- `ticks` - Number of simulation ticks completed
- `ms_per_tick` - Average milliseconds per tick
- `peak_memory_mb` - Peak memory usage in megabytes
- `perf_summary` - Detailed performance breakdown from PerformanceMonitor

## Regression Detection

The regression detector flags any benchmark where `ms_per_tick` has increased by more than 20% compared to the baseline.

Example warning:
```
[WARN] baseline_5_agents_500_ticks: +30.0% slower (baseline: 2.50ms, current: 3.25ms)
```

## Adding New Benchmarks

Edit `benchmarks/benchmark_suite.py` and add to `BENCHMARK_CONFIGS`:

```python
BENCHMARK_CONFIGS = {
    "my_new_benchmark": {
        "num_agents": 10,
        "max_ticks": 1000,
        "default_architecture": "reactive",
        # Any other SimulationConfig fields...
    },
}
```

All benchmarks use `seed=42` for determinism.
