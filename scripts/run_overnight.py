#!/usr/bin/env python3
"""Self-optimizing overnight experiment runner with validation queue.

This runner has 3 phases:
1. VALIDATE: Test new experiments at 5 reps, check for signal (Cohen's d > 0.3)
2. RUN: Execute all accepted experiments at optimized rep counts
3. ANALYZE: Generate figures and summary report

Usage:
    python scripts/run_overnight.py [--time-budget-hours 9] [--seed-start 1000] [--trajectory-samples 10]
"""

import argparse
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments.config import ExperimentConfig
from src.experiments.runner import ExperimentRunner, RunResult
from src.experiments.analysis import ResultAnalyzer
from src.experiments.report import ReportGenerator


# ======== EXPERIMENT TIERS ========

TIER_1_PROVEN = [
    "01_value_drift_contagion.yaml",
    "02_cooperation_collapse.yaml",
    "03_architecture_resilience.yaml",
    "04_minority_influence.yaml",
    "05_recovery_after_collapse.yaml",
    "06_minority_fine_sweep.yaml",  # extends proven finding
]

TIER_2_VALIDATE = [
    "07_corruption_dynamics.yaml",
    "08_multi_trait_corruption.yaml",
    "09_communication_isolation.yaml",
    "10_metacognition_defense.yaml",
    "11_coalition_defense.yaml",
]

TIER_3_BACKUPS = [
    "12_scarcity_pressure.yaml",
    "13_arch_archetype_matrix.yaml",
    "14_evolution_under_pressure.yaml",
    "15_cooperation_fine_sweep.yaml",
]

ALIGNMENT_DIR = Path("experiments/alignment")
OUTPUT_DIR = Path("data/experiments/alignment")
OVERNIGHT_SUMMARY_PATH = OUTPUT_DIR / "overnight_summary.md"

# Timing constants (empirical)
SECONDS_PER_FAST_RUN = 0.55
SECONDS_PER_TRAJECTORY_RUN = 3.17
VALIDATION_REPS = 5
VALIDATION_SEED_START = 9000
SIGNAL_THRESHOLD = 0.3  # Cohen's d threshold


# ======== VALIDATION PHASE ========

def validate_experiment(yaml_path: Path, seed_start: int = VALIDATION_SEED_START) -> tuple[float, int]:
    """Run experiment at low reps, return max Cohen's d and total runs.

    Args:
        yaml_path: Path to experiment YAML
        seed_start: Starting seed for validation runs

    Returns:
        (max_cohens_d, total_runs)
    """
    print(f"  Loading config...")
    config = ExperimentConfig.from_yaml(str(yaml_path))
    config.replicates = VALIDATION_REPS
    config.seed_start = seed_start
    config.record_trajectories = False
    config.trajectory_sample_count = 0

    n_conditions = len(config.expand_conditions())
    total_runs = n_conditions * VALIDATION_REPS

    print(f"  Running {total_runs} validation runs ({n_conditions} conditions × {VALIDATION_REPS} reps)...")

    runner = ExperimentRunner(config, config_yaml_path=str(yaml_path))

    # Track progress
    completed = [0]
    start_time = time.time()

    def progress_callback(condition: str, replicate: int, total: int) -> None:
        completed[0] += 1
        elapsed = time.time() - start_time
        rate = completed[0] / elapsed if elapsed > 0 else 0
        remaining = (total - completed[0]) / rate if rate > 0 else 0
        print(f"    [{completed[0]:3d}/{total}] {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining", end="\r")

    results = runner.run_all(progress_callback=progress_callback)
    print()  # Clear progress line

    # Compute effect sizes
    analyzer = ResultAnalyzer()
    comparisons = analyzer.pairwise_comparison(results)

    if not comparisons:
        return 0.0, total_runs

    max_d = max(abs(c["effect_size"]) for c in comparisons)
    return max_d, total_runs


def run_validation_phase(time_budget_seconds: float) -> tuple[list[Path], dict[str, float], float]:
    """Validate Tier 2 experiments, pull from Tier 3 backups if needed.

    Args:
        time_budget_seconds: Total time budget for overnight run

    Returns:
        (accepted_experiments, validation_results, time_used)
    """
    print("=" * 70)
    print("PHASE 1: VALIDATION")
    print("=" * 70)
    print()

    validation_start = time.time()
    accepted = []
    rejected = []
    validation_results = {}

    # Validate Tier 2
    for yaml_file in TIER_2_VALIDATE:
        yaml_path = ALIGNMENT_DIR / yaml_file
        if not yaml_path.exists():
            print(f"[SKIP] {yaml_file} not found")
            continue

        print(f"[VALIDATE] {yaml_file}")
        try:
            max_d, runs = validate_experiment(yaml_path)
            validation_results[yaml_file] = max_d

            if max_d >= SIGNAL_THRESHOLD:
                print(f"  [OK] ACCEPTED: Cohen's d = {max_d:.3f} (>= {SIGNAL_THRESHOLD})")
                accepted.append(yaml_path)
            else:
                print(f"  [--] REJECTED: Cohen's d = {max_d:.3f} (< {SIGNAL_THRESHOLD})")
                rejected.append(yaml_file)
        except Exception as e:
            print(f"  [!!] ERROR: {e}")
            import traceback
            traceback.print_exc()
            rejected.append(yaml_file)
        print()

    # Pull from Tier 3 backups to replace rejected experiments
    if rejected and TIER_3_BACKUPS:
        print(f"Pulling backups to replace {len(rejected)} rejected experiments...")
        print()

        for yaml_file in TIER_3_BACKUPS[:len(rejected)]:
            yaml_path = ALIGNMENT_DIR / yaml_file
            if not yaml_path.exists():
                print(f"[SKIP] {yaml_file} not found")
                continue

            print(f"[VALIDATE BACKUP] {yaml_file}")
            try:
                max_d, runs = validate_experiment(yaml_path)
                validation_results[yaml_file] = max_d

                if max_d >= SIGNAL_THRESHOLD:
                    print(f"  [OK] ACCEPTED: Cohen's d = {max_d:.3f} (>= {SIGNAL_THRESHOLD})")
                    accepted.append(yaml_path)
                else:
                    print(f"  [--] REJECTED: Cohen's d = {max_d:.3f} (< {SIGNAL_THRESHOLD})")
            except Exception as e:
                print(f"  [!!] ERROR: {e}")
                import traceback
                traceback.print_exc()
            print()

    time_used = time.time() - validation_start
    print(f"Validation complete: {len(accepted)} experiments accepted, {time_used/60:.1f} minutes used")
    print()

    # Sort accepted by signal strength (highest d first) so best experiments
    # run first — protects against late crashes wasting the night
    accepted.sort(
        key=lambda p: validation_results.get(p.name, 0.0),
        reverse=True,
    )

    return accepted, validation_results, time_used


# ======== RUN PHASE ========

def compute_reps(accepted_experiments: list[Path], remaining_seconds: float, trajectory_samples: int) -> int:
    """Calculate optimal reps per experiment based on time budget.

    Args:
        accepted_experiments: List of experiment YAML paths
        remaining_seconds: Remaining time budget
        trajectory_samples: Number of trajectory samples per condition

    Returns:
        Number of replicates per condition
    """
    # Count total conditions across all experiments
    total_conditions = 0
    for yaml_path in accepted_experiments:
        config = ExperimentConfig.from_yaml(str(yaml_path))
        total_conditions += len(config.expand_conditions())

    if total_conditions == 0:
        return 50

    # Budget for trajectory runs
    traj_seconds = total_conditions * trajectory_samples * SECONDS_PER_TRAJECTORY_RUN
    fast_seconds = remaining_seconds - traj_seconds

    if fast_seconds <= 0:
        # Not enough time for any fast runs
        return trajectory_samples

    # Calculate how many fast runs we can do
    fast_runs = fast_seconds / SECONDS_PER_FAST_RUN
    reps_per_condition = int(fast_runs / total_conditions) + trajectory_samples

    return max(reps_per_condition, 50)  # minimum 50 reps


def run_experiment(
    yaml_path: Path,
    replicates: int,
    seed_start: int,
    trajectory_samples: int,
    experiment_num: int,
    total_experiments: int,
) -> tuple[list[RunResult], float]:
    """Run a single experiment with error recovery.

    Args:
        yaml_path: Path to experiment YAML
        replicates: Number of replicates per condition
        seed_start: Starting seed
        trajectory_samples: Number of trajectory samples
        experiment_num: Current experiment number (1-indexed)
        total_experiments: Total number of experiments

    Returns:
        (results, duration_seconds)
    """
    yaml_file = yaml_path.name
    config = ExperimentConfig.from_yaml(str(yaml_path))
    config.replicates = replicates
    config.seed_start = seed_start
    config.record_trajectories = False
    config.trajectory_sample_count = trajectory_samples

    n_conditions = len(config.expand_conditions())
    n_runs = n_conditions * replicates

    print(f"[{experiment_num}/{total_experiments}] {config.name}")
    print(f"  Conditions: {n_conditions}, Replicates: {replicates}, Total runs: {n_runs}")
    print(f"  Trajectory samples: {trajectory_samples}")

    exp_start = time.time()
    runner = ExperimentRunner(config, config_yaml_path=str(yaml_path))

    # Track progress
    completed = [0]

    def progress_callback(condition: str, replicate: int, total: int) -> None:
        completed[0] += 1
        elapsed = time.time() - exp_start
        rate = completed[0] / elapsed if elapsed > 0 else 0
        remaining = (total - completed[0]) / rate if rate > 0 else 0
        print(
            f"  [{completed[0]:5d}/{total}] {condition[:30]:30s} rep {replicate+1:4d} "
            f"| {elapsed:.0f}s elapsed | ~{remaining:.0f}s remaining",
            end="\r",
        )

    try:
        results = runner.run_all(progress_callback=progress_callback)
        print()  # Clear progress line

        exp_duration = time.time() - exp_start

        # Save results immediately
        save_experiment_results(config, results, yaml_path)

        # Quick summary
        print(f"  [OK] Completed in {exp_duration:.1f}s ({exp_duration/60:.1f} min)")
        print_quick_summary(results)

        return results, exp_duration

    except Exception as e:
        print()
        print(f"  [!!] FAILED: {e}")
        import traceback
        traceback.print_exc()
        return [], 0.0


def save_experiment_results(config: ExperimentConfig, results: list[RunResult], yaml_path: Path) -> None:
    """Save experiment results to CSV and markdown.

    Args:
        config: Experiment configuration
        results: List of run results
        yaml_path: Path to experiment YAML
    """
    if not results:
        return

    analyzer = ResultAnalyzer()
    reporter = ReportGenerator()

    # Generate summaries
    summaries = analyzer.summarize(results)

    # Save CSV
    csv_path = Path(config.output_dir) / "results.csv"
    reporter.to_csv(results, str(csv_path))

    # Save markdown
    md_path = Path(config.output_dir) / "summary.md"
    reporter.to_markdown(config, summaries, str(md_path))

    print(f"  Output: {config.output_dir}")


def print_quick_summary(results: list[RunResult]) -> None:
    """Print quick summary of results.

    Args:
        results: List of run results
    """
    # Group by condition
    from collections import defaultdict
    by_condition = defaultdict(list)
    for r in results:
        by_condition[r.condition_name].append(r.metrics)

    for cond_name, metrics_list in by_condition.items():
        # Get key metrics
        coop_ratios = [m.get("cooperation_ratio", 0) for m in metrics_list]
        alive_counts = [m.get("agents_alive_at_end", 0) for m in metrics_list]

        avg_coop = sum(coop_ratios) / len(coop_ratios) if coop_ratios else 0
        avg_alive = sum(alive_counts) / len(alive_counts) if alive_counts else 0

        print(f"    {cond_name}: avg_coop={avg_coop:.3f}, avg_alive={avg_alive:.1f}")


def run_phase(
    tier_1_experiments: list[Path],
    validated_experiments: list[Path],
    replicates: int,
    seed_start: int,
    trajectory_samples: int,
    time_budget_seconds: float = 0,
) -> tuple[dict[str, tuple[list[RunResult], float]], float]:
    """Run all accepted experiments.

    Args:
        tier_1_experiments: List of Tier 1 (proven) experiment paths
        validated_experiments: List of validated experiment paths
        replicates: Number of replicates per condition
        seed_start: Starting seed
        trajectory_samples: Number of trajectory samples
        time_budget_seconds: Remaining time budget (0 = unlimited)

    Returns:
        (experiment_results_dict, time_used)
    """
    print("=" * 70)
    print("PHASE 2: RUN")
    print("=" * 70)
    print()

    all_experiments = tier_1_experiments + validated_experiments
    total_experiments = len(all_experiments)

    # Per-experiment time cap: 1.5x fair share, prevents one bad config eating the night
    per_experiment_cap = (
        (time_budget_seconds / total_experiments * 1.5)
        if time_budget_seconds > 0 and total_experiments > 0
        else 0
    )

    print(f"Running {total_experiments} experiments:")
    print(f"  Tier 1 (proven): {len(tier_1_experiments)}")
    print(f"  Validated: {len(validated_experiments)}")
    print(f"  Replicates per condition: {replicates}")
    print(f"  Seed start: {seed_start}")
    print(f"  Trajectory samples: {trajectory_samples}")
    if per_experiment_cap > 0:
        print(f"  Per-experiment time cap: {per_experiment_cap/60:.0f} min")
    print()

    run_start = time.time()
    experiment_results = {}

    for i, yaml_path in enumerate(all_experiments, 1):
        # Check overall time budget
        elapsed = time.time() - run_start
        if time_budget_seconds > 0 and elapsed >= time_budget_seconds:
            print(f"[TIME] Budget exhausted after {elapsed/3600:.1f}h, skipping remaining experiments")
            break

        results, duration = run_experiment(
            yaml_path,
            replicates,
            seed_start,
            trajectory_samples,
            i,
            total_experiments,
        )

        # Check per-experiment time cap
        if per_experiment_cap > 0 and duration > per_experiment_cap:
            print(f"  [WARN] Experiment took {duration/60:.0f}m (cap: {per_experiment_cap/60:.0f}m)")

        experiment_results[yaml_path.name] = (results, duration)
        print()

    time_used = time.time() - run_start
    print(f"All experiments complete: {time_used/3600:.1f} hours")
    print()

    return experiment_results, time_used


# ======== ANALYSIS PHASE ========

def analyze_phase() -> list[str]:
    """Generate figures and analysis.

    Returns:
        List of generated figure paths
    """
    print("=" * 70)
    print("PHASE 3: ANALYZE")
    print("=" * 70)
    print()

    # Check if analyze_overnight.py exists
    analyze_script = Path("scripts/analyze_overnight.py")
    if not analyze_script.exists():
        print("No analyze_overnight.py script found, skipping analysis phase")
        return []

    try:
        # Import and run analysis
        import importlib.util
        spec = importlib.util.spec_from_file_location("analyze_overnight", str(analyze_script))
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, 'run_analysis'):
                print("Running overnight analysis...")
                figures = module.run_analysis()
                print(f"[OK] Generated {len(figures)} figures")
                return figures
            else:
                print("analyze_overnight.py has no run_analysis() function")
                return []
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return []


# ======== SUMMARY GENERATION ========

def generate_overnight_summary(
    start_time: datetime,
    end_time: datetime,
    validation_results: dict[str, float],
    tier_1_experiments: list[Path],
    validated_experiments: list[Path],
    experiment_results: dict[str, tuple[list[RunResult], float]],
    replicates: int,
    generated_figures: list[str],
) -> None:
    """Generate comprehensive overnight summary markdown.

    Args:
        start_time: When the overnight run started
        end_time: When the overnight run ended
        validation_results: Dict of experiment_name -> Cohen's d
        tier_1_experiments: List of Tier 1 experiment paths
        validated_experiments: List of validated experiment paths
        experiment_results: Dict of experiment_name -> (results, duration)
        replicates: Number of replicates used
        generated_figures: List of generated figure paths
    """
    lines = []
    lines.append("# AUTOCOG Overnight Experiment Summary")
    lines.append("")

    duration_hours = (end_time - start_time).total_seconds() / 3600
    total_runs = sum(len(results) for results, _ in experiment_results.values())

    lines.append(f"**Started:** {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Finished:** {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Duration:** {duration_hours:.1f} hours")
    lines.append(f"**Total runs:** {total_runs:,}")
    lines.append("")

    # Validation results
    if validation_results:
        lines.append("## Validation Results")
        lines.append("")
        lines.append("| Experiment | Max Cohen's d | Status |")
        lines.append("|------------|---------------|--------|")

        for exp_name in sorted(validation_results.keys()):
            cohens_d = validation_results[exp_name]
            status = "Accepted" if cohens_d >= SIGNAL_THRESHOLD else "Rejected"

            # Check if it was a backup
            if exp_name in TIER_3_BACKUPS:
                status += " (backup)"

            lines.append(f"| {exp_name} | {cohens_d:.2f} | {status} |")

        lines.append("")

    # Experiments run
    lines.append("## Experiments Run")
    lines.append("")
    lines.append("| Experiment | Conditions | Reps | Runs | Duration | Output |")
    lines.append("|------------|------------|------|------|----------|--------|")

    all_experiments = tier_1_experiments + validated_experiments
    for yaml_path in all_experiments:
        exp_name = yaml_path.name
        if exp_name in experiment_results:
            results, duration = experiment_results[exp_name]
            if results:
                config = ExperimentConfig.from_yaml(str(yaml_path))
                n_conditions = len(config.expand_conditions())
                n_runs = len(results)
                duration_min = duration / 60

                lines.append(
                    f"| {exp_name} | {n_conditions} | {replicates} | {n_runs} | "
                    f"{duration_min:.0f}m | {config.output_dir} |"
                )

    lines.append("")

    # Key metrics summary
    lines.append("## Key Metrics Summary")
    lines.append("")

    for yaml_path in all_experiments:
        exp_name = yaml_path.name
        if exp_name in experiment_results:
            results, _ = experiment_results[exp_name]
            if results:
                config = ExperimentConfig.from_yaml(str(yaml_path))
                lines.append(f"### {config.name}")
                lines.append("")

                # Compute summaries
                analyzer = ResultAnalyzer()
                summaries = analyzer.summarize(results)

                # Create compact summary table
                if summaries:
                    # Get first metric for compact display
                    first_metric = list(summaries[0].metrics.keys())[0] if summaries[0].metrics else None
                    if first_metric:
                        lines.append(f"**{first_metric}**")
                        lines.append("")
                        lines.append("| Condition | Mean | 95% CI |")
                        lines.append("|-----------|------|--------|")

                        for summary in summaries:
                            if first_metric in summary.metrics:
                                stats = summary.metrics[first_metric]
                                lines.append(
                                    f"| {summary.condition_name} | {stats['mean']:.2f} | "
                                    f"[{stats['ci_95_lower']:.2f}, {stats['ci_95_upper']:.2f}] |"
                                )

                        lines.append("")
                        lines.append(f"See `{config.output_dir}/summary.md` for full results.")
                        lines.append("")

    # Generated figures
    if generated_figures:
        lines.append("## Figures Generated")
        lines.append("")
        for fig_path in sorted(generated_figures):
            lines.append(f"- {fig_path}")
        lines.append("")

    # Write summary
    OVERNIGHT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OVERNIGHT_SUMMARY_PATH, "w") as f:
        f.write("\n".join(lines))

    print(f"Overnight summary written to: {OVERNIGHT_SUMMARY_PATH}")


# ======== SYSTEM GUARDS ========

def _prevent_sleep() -> None:
    """Prevent Windows from sleeping during overnight run."""
    try:
        import ctypes
        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
        print("[GUARD] Windows sleep prevention enabled")
    except (AttributeError, OSError):
        pass  # Not Windows or no access


def _allow_sleep() -> None:
    """Restore normal Windows sleep behavior."""
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)  # ES_CONTINUOUS
    except (AttributeError, OSError):
        pass


def _check_disk_space(min_gb: float = 2.0, warn_gb: float = 5.0) -> None:
    """Check available disk space before starting.

    Args:
        min_gb: Minimum GB required (aborts if less)
        warn_gb: Warning threshold in GB
    """
    usage = shutil.disk_usage(".")
    free_gb = usage.free / (1024 ** 3)
    if free_gb < min_gb:
        print(f"[ABORT] Only {free_gb:.1f} GB free disk space (need >= {min_gb:.0f} GB)")
        sys.exit(1)
    elif free_gb < warn_gb:
        print(f"[WARN] Low disk space: {free_gb:.1f} GB free (run needs ~3.1 GB)")
    else:
        print(f"[GUARD] Disk space OK: {free_gb:.1f} GB free")


# ======== MAIN ========

def main() -> None:
    """Main overnight runner."""
    parser = argparse.ArgumentParser(
        description="Self-optimizing overnight experiment runner"
    )
    parser.add_argument(
        "--time-budget-hours",
        type=float,
        default=9.0,
        help="Total time budget in hours (default: 9.0)",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=1000,
        help="Starting seed for experiments (default: 1000)",
    )
    parser.add_argument(
        "--trajectory-samples",
        type=int,
        default=10,
        help="Number of trajectory samples per condition (default: 10)",
    )
    args = parser.parse_args()

    # System guards
    _check_disk_space()
    _prevent_sleep()

    try:
        _run_overnight(args)
    finally:
        _allow_sleep()


def _run_overnight(args: argparse.Namespace) -> None:
    """Execute the overnight run (wrapped by sleep guard in main)."""
    overall_start = datetime.now()
    time_budget_seconds = args.time_budget_hours * 3600

    print("=" * 70)
    print("AUTOCOG Self-Optimizing Overnight Experiment Runner")
    print("=" * 70)
    print()
    print(f"Time budget: {args.time_budget_hours:.1f} hours ({time_budget_seconds:.0f} seconds)")
    print(f"Seed start: {args.seed_start}")
    print(f"Trajectory samples: {args.trajectory_samples}")
    print(f"Started: {overall_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Phase 1: Validation
    validated_experiments, validation_results, validation_time = run_validation_phase(
        time_budget_seconds
    )

    # Prepare Tier 1 experiments
    tier_1_experiments = [
        ALIGNMENT_DIR / yaml_file
        for yaml_file in TIER_1_PROVEN
        if (ALIGNMENT_DIR / yaml_file).exists()
    ]

    print(f"Tier 1 (proven) experiments: {len(tier_1_experiments)}")
    print(f"Validated experiments: {len(validated_experiments)}")
    print()

    # Phase 2: Compute reps and run
    all_accepted = tier_1_experiments + validated_experiments
    remaining_time = time_budget_seconds - validation_time

    if remaining_time <= 0:
        print("ERROR: Validation consumed entire time budget!")
        return

    replicates = compute_reps(all_accepted, remaining_time, args.trajectory_samples)
    print(f"Computed replicates per condition: {replicates}")
    print()

    experiment_results, run_time = run_phase(
        tier_1_experiments,
        validated_experiments,
        replicates,
        args.seed_start,
        args.trajectory_samples,
        time_budget_seconds=remaining_time,
    )

    # Phase 3: Analysis
    generated_figures = analyze_phase()

    # Generate summary
    overall_end = datetime.now()
    generate_overnight_summary(
        overall_start,
        overall_end,
        validation_results,
        tier_1_experiments,
        validated_experiments,
        experiment_results,
        replicates,
        generated_figures,
    )

    # Final report
    total_duration = (overall_end - overall_start).total_seconds()
    total_runs = sum(len(results) for results, _ in experiment_results.values())

    print()
    print("=" * 70)
    print("OVERNIGHT RUN COMPLETE")
    print("=" * 70)
    print(f"Total duration: {total_duration/3600:.1f} hours")
    print(f"Total runs: {total_runs:,}")
    print(f"Average per run: {total_duration/total_runs:.2f}s" if total_runs > 0 else "N/A")
    print(f"Summary: {OVERNIGHT_SUMMARY_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
