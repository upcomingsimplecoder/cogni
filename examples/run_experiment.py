"""Tutorial: Running Batch Experiments Programmatically

This demonstrates two approaches for running experiments with the AUTOCOG framework:

1. APPROACH A: Load from existing YAML config
   - Useful when you have pre-defined experiment configurations
   - Supports full reproducibility and version control
   - Clean separation between config and execution code

2. APPROACH B: Create config programmatically
   - Useful for dynamic experiment generation
   - Easier to iterate and prototype
   - Good for parameter sweeps and automated testing

Both approaches use:
- ExperimentConfig: Defines base settings, conditions, and metrics
- ExperimentRunner: Executes simulations and collects results
- RunResult: Contains metrics from individual simulation runs

Quick Concepts:
- Conditions: Different parameter settings to compare (e.g., 3 vs 5 vs 8 agents)
- Replicates: Number of runs per condition with different seeds
- Metrics: Measurements collected at simulation end (survival_rate, cooperation_ratio, etc.)
"""

from pathlib import Path
from src.experiments.config import ExperimentConfig, ExperimentCondition
from src.experiments.runner import ExperimentRunner


def progress_callback(condition_name, replicate, total_runs):
    """Simple progress indicator for experiment execution."""
    print(f"  Running: {condition_name} [replicate {replicate + 1}]")


def print_summary_table(results):
    """Print a clean summary table of experiment results."""
    from collections import defaultdict

    # Group results by condition
    by_condition = defaultdict(list)
    for result in results:
        by_condition[result.condition_name].append(result)

    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)

    # Get all metric names from first result
    if results:
        metric_names = list(results[0].metrics.keys())

        # Print header
        print(f"\n{'Condition':<20} {'Metric':<30} {'Mean':<12} {'Min':<12} {'Max':<12}")
        print("-" * 86)

        # Print each condition's metrics
        for condition_name in sorted(by_condition.keys()):
            condition_results = by_condition[condition_name]

            for i, metric_name in enumerate(metric_names):
                values = [r.metrics[metric_name] for r in condition_results]
                mean_val = sum(values) / len(values)
                min_val = min(values)
                max_val = max(values)

                # Only print condition name on first metric row
                cond_display = condition_name if i == 0 else ""
                print(f"{cond_display:<20} {metric_name:<30} {mean_val:<12.3f} {min_val:<12.3f} {max_val:<12.3f}")

            print()  # Blank line between conditions

    print("=" * 80)


def approach_a_load_from_yaml():
    """APPROACH A: Load experiment config from existing YAML file.

    This demonstrates loading the architecture comparison experiment
    and running it programmatically. The YAML file defines all settings.
    """
    print("\n" + "=" * 80)
    print("APPROACH A: Load from YAML")
    print("=" * 80)

    # Path to existing YAML config
    yaml_path = "examples/architecture_comparison.yaml"

    print(f"\nLoading experiment config from: {yaml_path}")

    # Load config from YAML
    config = ExperimentConfig.from_yaml(yaml_path)

    print(f"Experiment: {config.name}")
    print(f"Conditions: {[c.name for c in config.conditions]}")
    print(f"Replicates per condition: {config.replicates}")
    print(f"Total runs: {len(config.conditions) * config.replicates}")
    print(f"Metrics tracked: {len(config.metrics)}")

    # Create runner and execute
    print("\nStarting experiment runs...")
    runner = ExperimentRunner(config, config_yaml_path=yaml_path)
    results = runner.run_all(progress_callback=progress_callback)

    print(f"\nCompleted {len(results)} simulation runs")
    print(f"Results saved to: {config.output_dir}")

    # Print summary
    print_summary_table(results)

    return results


def approach_b_programmatic_config():
    """APPROACH B: Create experiment config programmatically.

    This demonstrates creating a population density experiment from scratch
    in Python code, comparing how group size affects survival and cooperation.
    """
    print("\n" + "=" * 80)
    print("APPROACH B: Programmatic Config")
    print("=" * 80)

    print("\nCreating population density experiment...")

    # Define base configuration (shared across all conditions)
    base_config = {
        "world_width": 28,
        "world_height": 28,
        "max_ticks": 200,  # Shorter for faster execution
        "coalitions_enabled": True,
        "cultural_transmission_enabled": False,
        "language_enabled": True,
        "metacognition_enabled": False,
        "default_architecture": "social",  # Use social architecture
    }

    # Define conditions: different population sizes
    conditions = [
        ExperimentCondition(
            name="population_3",
            overrides={"num_agents": 3},
        ),
        ExperimentCondition(
            name="population_5",
            overrides={"num_agents": 5},
        ),
        ExperimentCondition(
            name="population_8",
            overrides={"num_agents": 8},
        ),
    ]

    # Define metrics to track
    metrics = [
        "survival_rate",           # Fraction of agents alive at end
        "avg_survival_ticks",      # Mean ticks survived
        "cooperation_ratio",       # cooperation / (cooperation + aggression)
        "avg_trust_network_density",  # Density of trust relationships
        "coalition_count",         # Number of coalitions formed
        "total_cooperation_events",
        "total_aggression_events",
    ]

    # Create experiment config
    config = ExperimentConfig(
        name="Population Density Study",
        description="Tests how population size affects survival and cooperation dynamics",
        base=base_config,
        conditions=conditions,
        replicates=3,  # 3 replicates per condition = 9 total runs
        seed_start=42,
        metrics=metrics,
        output_dir="data/experiments/population_density_study",
        formats=["csv", "markdown", "json"],
        record_trajectories=False,  # Skip trajectory recording for speed
    )

    print(f"Experiment: {config.name}")
    print(f"Conditions: {[c.name for c in config.conditions]}")
    print(f"Replicates per condition: {config.replicates}")
    print(f"Total runs: {len(config.conditions) * config.replicates}")
    print(f"Metrics tracked: {len(config.metrics)}")

    # Create runner and execute
    print("\nStarting experiment runs...")
    runner = ExperimentRunner(config, config_yaml_path="generated_from_code.yaml")
    results = runner.run_all(progress_callback=progress_callback)

    print(f"\nCompleted {len(results)} simulation runs")
    print(f"Results saved to: {config.output_dir}")

    # Print summary
    print_summary_table(results)

    # Example: Access individual results programmatically
    print("\n" + "=" * 80)
    print("PROGRAMMATIC ANALYSIS EXAMPLE")
    print("=" * 80)

    # Group by condition and calculate statistics
    from collections import defaultdict
    by_condition = defaultdict(list)
    for result in results:
        by_condition[result.condition_name].append(result)

    # Find condition with highest mean cooperation ratio
    print("\nFinding condition with highest cooperation ratio:")
    for condition_name, condition_results in by_condition.items():
        coop_ratios = [r.metrics["cooperation_ratio"] for r in condition_results]
        mean_coop = sum(coop_ratios) / len(coop_ratios)
        print(f"  {condition_name}: {mean_coop:.3f}")

    best_condition = max(
        by_condition.items(),
        key=lambda x: sum(r.metrics["cooperation_ratio"] for r in x[1]) / len(x[1])
    )
    print(f"\nHighest cooperation: {best_condition[0]}")

    # Check survival rates
    print("\nSurvival rate analysis:")
    for condition_name, condition_results in by_condition.items():
        survival_rates = [r.metrics["survival_rate"] for r in condition_results]
        mean_survival = sum(survival_rates) / len(survival_rates)
        print(f"  {condition_name}: {mean_survival:.1%} survival")

    return results


def main():
    """Run both demonstration approaches."""
    print("\n" + "=" * 80)
    print("BATCH EXPERIMENT TUTORIAL")
    print("=" * 80)
    print("\nThis tutorial demonstrates two approaches for running experiments:")
    print("  A) Load from YAML config file")
    print("  B) Create config programmatically in Python")
    print()

    # Choose which approach to run
    import sys

    if len(sys.argv) > 1:
        choice = sys.argv[1].lower()
    else:
        print("Select approach:")
        print("  1) Load from YAML (architecture comparison)")
        print("  2) Programmatic config (population density)")
        print("  3) Both")
        choice = input("\nEnter choice (1/2/3): ").strip()

    if choice in ["1", "a", "yaml"]:
        approach_a_load_from_yaml()
    elif choice in ["2", "b", "programmatic", "code"]:
        approach_b_programmatic_config()
    elif choice in ["3", "both"]:
        approach_a_load_from_yaml()
        approach_b_programmatic_config()
    else:
        print(f"Invalid choice: {choice}")
        print("Usage: python examples/run_experiment.py [1|2|3]")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("TUTORIAL COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  - Check output directories for CSV/markdown/JSON results")
    print("  - Modify conditions to test different parameters")
    print("  - Add custom metrics to config.metrics list")
    print("  - Use record_trajectories=True for detailed analysis")
    print("  - Load results from CSV for plotting and further analysis")


if __name__ == "__main__":
    main()
