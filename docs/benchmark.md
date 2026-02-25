# AUTOCOG Benchmark CLI

Command-line interface for running benchmark suites and publishing results to HuggingFace.

## Quick Start

### Run Full Benchmark Suite

```bash
python -m src.experiments benchmark --suite experiments/benchmarks/
```

This will:
1. Execute all 10 benchmark scenarios
2. Generate per-scenario reports
3. Create a cross-scenario comparison report
4. Save all results to `data/experiments/benchmarks/`

### Run Single Scenario

```bash
python -m src.experiments benchmark --scenario experiments/benchmarks/01_survival_baseline.yaml
```

### Override Architectures

Test specific architectures across all scenarios:

```bash
python -m src.experiments benchmark --suite experiments/benchmarks/ --architectures reactive dual_process social
```

### Generate Report from Existing Results

If you already have results and just want to regenerate reports:

```bash
python -m src.experiments benchmark --report data/experiments/benchmarks/
```

### Push to HuggingFace

Publish benchmark results as a HuggingFace dataset:

```bash
# Run benchmark and push results
python -m src.experiments benchmark --suite experiments/benchmarks/ --push-hf cogni/autocog-benchmarks

# Or push existing results
python -m src.experiments benchmark --report data/experiments/benchmarks/ --push-hf cogni/autocog-benchmarks
```

**Requirements for HF push:**
```bash
pip install huggingface_hub
huggingface-cli login
```

## Command Reference

### `benchmark` Subcommand

```
python -m src.experiments benchmark [OPTIONS]
```

**Options:**

- `--suite PATH`
  - Path to benchmark suite directory (e.g., `experiments/benchmarks/`)
  - Runs all YAML files in the directory

- `--scenario PATH`
  - Path to single scenario YAML file
  - Alternative to `--suite` for running one scenario

- `--architectures ARCH [ARCH ...]`
  - Override architectures to test
  - Example: `--architectures reactive dual_process social`
  - Replaces the `conditions` in each YAML file

- `--report PATH`
  - Generate report from existing results directory
  - Does not execute simulations

- `--push-hf REPO_ID`
  - Push results to HuggingFace dataset
  - Example: `--push-hf cogni/autocog-benchmarks`
  - Requires `huggingface_hub` and authentication

## Output Structure

```
data/experiments/benchmarks/
├── 01_survival_baseline_report.md       # Per-scenario analysis
├── 01_survival_baseline_results.csv     # Raw data
├── 02_cooperation_report.md
├── 02_cooperation_results.csv
├── ...
├── benchmark_comparison.md              # Cross-scenario comparison
├── benchmark_summary.json               # Machine-readable summary
├── README.md                            # HF dataset card (if --push-hf)
└── trajectories/                        # Full simulation trajectories (if enabled)
    ├── reactive_0.jsonl
    ├── reactive_1.jsonl
    └── ...
```

## Report Contents

### Per-Scenario Reports (`*_report.md`)

- Summary statistics per condition (N, mean, std, min, max, 95% CI)
- One table per metric
- Organized by scenario

### Cross-Scenario Comparison (`benchmark_comparison.md`)

- Architecture rankings (wins by scenario)
- Scenario-by-scenario winners
- Detailed results tables for all scenarios
- Summary findings

### Summary JSON (`benchmark_summary.json`)

Machine-readable format with:
- Full statistics per scenario and condition
- All metrics and confidence intervals
- Easy to load for custom analysis

## HuggingFace Dataset Format

When using `--push-hf`, creates a dataset with:

### Files
- `trajectories/*.jsonl` - Full simulation trajectories (if recorded)
- `*_results.csv` - Raw experimental data
- `*_report.md` - Analysis reports
- `benchmark_comparison.md` - Cross-scenario comparison
- `benchmark_summary.json` - Aggregated results
- `README.md` - Auto-generated dataset card

### Dataset Card

Includes:
- Dataset description
- Architecture comparison summary
- Data format documentation
- Usage examples
- Citation information

### Loading from HuggingFace

```python
from datasets import load_dataset
import pandas as pd

# Load dataset
dataset = load_dataset("cogni/autocog-benchmarks")

# Load CSV results
df = pd.read_csv("01_survival_baseline_results.csv")

# Load trajectories
import json
with open("trajectories/reactive_0.jsonl") as f:
    trajectory = [json.loads(line) for line in f]
```

## Architecture Override Use Case

The `--architectures` flag is useful for:

1. **Comparing new architectures**: Test a new architecture against baselines across all scenarios
   ```bash
   python -m src.experiments benchmark --suite experiments/benchmarks/ --architectures baseline new_arch
   ```

2. **Focused comparison**: Only test architectures of interest
   ```bash
   python -m src.experiments benchmark --suite experiments/benchmarks/ --architectures reactive dual_process
   ```

3. **Quick validation**: Test a single architecture across scenarios
   ```bash
   python -m src.experiments benchmark --suite experiments/benchmarks/ --architectures reactive
   ```

This overrides the `conditions` specified in each YAML file, allowing you to run the same scenarios with different architectural configurations.

## Error Handling

The benchmark runner:
- Continues if a scenario fails (doesn't abort entire suite)
- Validates suite directory exists and contains YAML files
- Handles missing HuggingFace credentials gracefully
- Provides clear error messages for authentication issues

## Performance Notes

- Full benchmark suite (10 scenarios × 3-5 architectures × 5 replicates) takes approximately **30-60 minutes** depending on hardware
- Single scenario typically completes in **2-5 minutes**
- Trajectory recording increases runtime by ~20-30% but provides full observability
- Reports are generated quickly (< 5 seconds) from existing results

## Example Workflow

```bash
# 1. Run benchmarks with trajectory recording
python -m src.experiments benchmark --suite experiments/benchmarks/

# 2. Review comparison report
cat data/experiments/benchmarks/benchmark_comparison.md

# 3. If satisfied, push to HuggingFace
python -m src.experiments benchmark \
  --report data/experiments/benchmarks/ \
  --push-hf cogni/autocog-benchmarks

# 4. Share dataset URL with collaborators
# https://huggingface.co/datasets/cogni/autocog-benchmarks
```

## Integration with Experiment Registry

All benchmark runs are automatically registered in the experiment registry:
- Full provenance tracking (git commit, dependencies, environment)
- Config hash for reproducibility
- Duration and performance metrics

Query benchmark experiments:

```bash
# List all experiments
python -m src.experiments list

# Find experiments by config
python -m src.experiments find-config <hash>

# Verify reproducibility
python -m src.experiments verify <experiment_id>
```

## Troubleshooting

### HuggingFace Authentication

If push fails with authentication error:

```bash
# Get token from https://huggingface.co/settings/tokens
huggingface-cli login

# Or set environment variable
export HF_TOKEN=your_token_here
```

### Missing Dependencies

```bash
# For YAML support
pip install pyyaml

# For HuggingFace push
pip install huggingface_hub
```

### Empty Results

If benchmark completes but generates no results:
1. Check that YAML files are valid
2. Verify base simulation config is correct
3. Check console output for error messages
4. Review individual scenario logs

## See Also

- [Benchmark Scenarios](experiments/benchmarks/README.md) - Description of all 10 scenarios
- [Experiment Framework](src/experiments/README.md) - Core experiment system
- [Metrics Reference](docs/metrics.md) - All available metrics
