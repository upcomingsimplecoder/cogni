"""CLI for experiment management and provenance queries.

Usage:
    python -m src.experiments --list
    python -m src.experiments --compare exp_abc123 exp_def456
    python -m src.experiments --verify exp_abc123
    python -m src.experiments --find-config <yaml_hash>
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

from src.experiments.analysis import ResultAnalyzer
from src.experiments.benchmark import BenchmarkRunner
from src.experiments.config import ExperimentConfig
from src.experiments.registry import ExperimentRegistry
from src.experiments.report import ReportGenerator
from src.experiments.runner import ExperimentRunner


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AUTOCOG Experiment Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # List experiments
    list_parser = subparsers.add_parser("list", help="List all experiments")
    list_parser.add_argument(
        "--filter",
        type=str,
        help="Filter by field (e.g., 'provenance.git_commit=abc123')",
    )

    # Compare experiments
    compare_parser = subparsers.add_parser("compare", help="Compare two experiments")
    compare_parser.add_argument("exp_id_a", help="First experiment ID")
    compare_parser.add_argument("exp_id_b", help="Second experiment ID")

    # Verify experiment
    verify_parser = subparsers.add_parser("verify", help="Verify experiment reproducibility")
    verify_parser.add_argument("exp_id", help="Experiment ID to verify")

    # Find by config
    find_parser = subparsers.add_parser("find-config", help="Find experiments with same config")
    find_parser.add_argument("yaml_hash", help="SHA256 hash of config YAML")

    # Run experiment
    run_parser = subparsers.add_parser("run", help="Run experiment from YAML")
    run_parser.add_argument("yaml_path", help="Path to experiment YAML config")

    # Benchmark suite
    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmark suite")
    benchmark_parser.add_argument(
        "--suite",
        type=str,
        help="Path to benchmark suite directory (e.g., experiments/benchmarks/)",
    )
    benchmark_parser.add_argument(
        "--scenario",
        type=str,
        help="Path to single scenario YAML (alternative to --suite)",
    )
    benchmark_parser.add_argument(
        "--architectures",
        nargs="+",
        help="Override architectures to test (e.g., reactive dual_process social)",
    )
    benchmark_parser.add_argument(
        "--report",
        type=str,
        help="Generate report from existing results directory (no execution)",
    )
    benchmark_parser.add_argument(
        "--push-hf",
        type=str,
        metavar="REPO_ID",
        help="Push results to HuggingFace dataset (e.g., upcomingsimplecoder/cogniarch-benchmarks)",
    )

    # HuggingFace push (standalone)
    hf_push_parser = subparsers.add_parser("hf-push", help="Push dataset to HuggingFace")
    hf_push_parser.add_argument(
        "directory",
        help="Directory to upload (e.g., data/hf_dataset)",
    )
    hf_push_parser.add_argument(
        "--repo-id",
        default="upcomingsimplecoder/cogniarch-benchmarks",
        help="HuggingFace repository ID (default: upcomingsimplecoder/cogniarch-benchmarks)",
    )
    hf_push_parser.add_argument(
        "--assemble",
        action="store_true",
        help="Assemble dataset from catalog before pushing",
    )

    args = parser.parse_args()

    if args.command == "list":
        list_experiments(args)
    elif args.command == "compare":
        compare_experiments(args)
    elif args.command == "verify":
        verify_experiment(args)
    elif args.command == "find-config":
        find_by_config(args)
    elif args.command == "run":
        run_experiment(args)
    elif args.command == "benchmark":
        run_benchmark(args)
    elif args.command == "hf-push":
        hf_push(args)
    else:
        parser.print_help()
        sys.exit(1)


def list_experiments(args: argparse.Namespace) -> None:
    """List all experiments in registry."""
    registry = ExperimentRegistry()

    # Parse filter if provided
    filter_by = None
    if args.filter:
        try:
            key, value = args.filter.split("=", 1)
            filter_by = {key: value}
        except ValueError:
            print(f"Invalid filter format: {args.filter}")
            print("Expected format: field=value (e.g., 'provenance.git_commit=abc123')")
            sys.exit(1)

    experiments = registry.list_experiments(filter_by=filter_by)

    if not experiments:
        print("No experiments found.")
        return

    print(f"Found {len(experiments)} experiment(s):\n")
    for exp in experiments:
        prov = exp["provenance"]
        print(f"ID: {prov['experiment_id']}")
        print(f"  Timestamp: {prov['timestamp']}")
        print(f"  Name: {prov['config_resolved']['name']}")
        print(f"  Git: {prov['git_commit'] or 'N/A'} {'(dirty)' if prov['git_dirty'] else ''}")
        print(f"  Config Hash: {prov['config_yaml_hash'][:12]}...")
        print(f"  Duration: {prov['duration_seconds']:.2f}s")
        print(f"  Results: {exp['results_dir']}")
        print()


def compare_experiments(args: argparse.Namespace) -> None:
    """Compare two experiments."""
    registry = ExperimentRegistry()

    try:
        comparison = registry.compare(args.exp_id_a, args.exp_id_b)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    exp_a = comparison["exp_a"]["provenance"]
    exp_b = comparison["exp_b"]["provenance"]

    print("Comparing experiments:\n")
    print(f"A: {exp_a['experiment_id']}")
    print(f"   {exp_a['timestamp']}")
    print(f"   {exp_a['config_resolved']['name']}")
    print()
    print(f"B: {exp_b['experiment_id']}")
    print(f"   {exp_b['timestamp']}")
    print(f"   {exp_b['config_resolved']['name']}")
    print()

    # Show config differences
    config_diff = comparison["config_diff"]
    if any(config_diff.values()):
        print("Configuration Differences:")
        if config_diff["only_in_a"]:
            print("  Only in A:")
            for key, value in config_diff["only_in_a"].items():
                print(f"    {key}: {value}")
        if config_diff["only_in_b"]:
            print("  Only in B:")
            for key, value in config_diff["only_in_b"].items():
                print(f"    {key}: {value}")
        if config_diff["different_values"]:
            print("  Different values:")
            for key, values in config_diff["different_values"].items():
                print(f"    {key}:")
                print(f"      A: {values['a']}")
                print(f"      B: {values['b']}")
    else:
        print("Configuration: Identical")

    print()

    # Show metadata differences
    metadata_diff = comparison["metadata_diff"]
    if metadata_diff["different_values"]:
        print("Metadata Differences:")
        for key, values in metadata_diff["different_values"].items():
            print(f"  {key}:")
            print(f"    A: {values['a']}")
            print(f"    B: {values['b']}")


def verify_experiment(args: argparse.Namespace) -> None:
    """Verify experiment reproducibility."""
    registry = ExperimentRegistry()
    experiments = registry.list_experiments(filter_by={"provenance.experiment_id": args.exp_id})

    if not experiments:
        print(f"Experiment {args.exp_id} not found in registry.")
        sys.exit(1)

    exp = experiments[0]
    prov = exp["provenance"]

    print(f"Experiment: {prov['experiment_id']}")
    print(f"Timestamp: {prov['timestamp']}")
    print()
    print("Reproducibility Status:")
    print(f"  Git Commit: {prov['git_commit'] or 'N/A'}")

    if prov["git_dirty"]:
        print("  [WARN] Uncommitted changes present at runtime")
    else:
        print("  [OK] Clean working tree")

    print(f"  Config Hash: {prov['config_yaml_hash']}")

    if prov["config_yaml_hash"] == "file_not_found":
        print("  [WARN] Config file not found")
    else:
        print("  [OK] Config file tracked")

    print()
    print("Environment:")
    print(f"  Python: {prov['python_version']}")
    print(f"  Platform: {prov['platform_info']}")
    print(f"  AUTOCOG: {prov['autocog_version']}")
    print()
    print("Dependencies:")
    for pkg, version in prov["dependencies"].items():
        print(f"  {pkg}: {version}")


def find_by_config(args: argparse.Namespace) -> None:
    """Find experiments with same config."""
    registry = ExperimentRegistry()
    experiments = registry.find_by_config(args.yaml_hash)

    if not experiments:
        print(f"No experiments found with config hash {args.yaml_hash}")
        return

    print(f"Found {len(experiments)} experiment(s) with config hash {args.yaml_hash}:\n")
    for exp in experiments:
        prov = exp["provenance"]
        print(f"ID: {prov['experiment_id']}")
        print(f"  Timestamp: {prov['timestamp']}")
        print(f"  Duration: {prov['duration_seconds']:.2f}s")
        print(f"  Results: {exp['results_dir']}")
        print()


def run_experiment(args: argparse.Namespace) -> None:
    """Run experiment from YAML config."""
    yaml_path = Path(args.yaml_path)
    if not yaml_path.exists():
        print(f"Config file not found: {yaml_path}")
        sys.exit(1)

    print(f"Loading experiment from {yaml_path}")
    config = ExperimentConfig.from_yaml(str(yaml_path))

    print(f"Experiment: {config.name}")
    print(f"Conditions: {len(config.conditions)}")
    print(f"Replicates: {config.replicates}")
    print(f"Total runs: {len(config.conditions) * config.replicates}")
    print()

    runner = ExperimentRunner(config, config_yaml_path=str(yaml_path))

    def progress(condition: str, replicate: int, total: int) -> None:
        print(f"Running {condition} replicate {replicate + 1}/{config.replicates}", end="\r")

    runner.run_all(progress_callback=progress)
    print()

    provenance = runner.get_provenance()
    if provenance:
        print(f"Experiment complete: {provenance.experiment_id}")
        print(f"Duration: {provenance.duration_seconds:.2f}s")
        print(f"Results: {config.output_dir}")


def run_benchmark(args: argparse.Namespace) -> None:
    """Run benchmark suite or single scenario."""
    # Validate arguments
    if not args.suite and not args.scenario and not args.report:
        print("Error: Must specify --suite, --scenario, or --report")
        print()
        print("Usage:")
        print("  python -m src.experiments benchmark --suite experiments/benchmarks/")
        print(
            "  python -m src.experiments benchmark --scenario "
            "experiments/benchmarks/01_survival_baseline.yaml"
        )
        print("  python -m src.experiments benchmark --report data/experiments/benchmarks/")
        sys.exit(1)

    # Report generation mode (no execution)
    if args.report:
        report_dir = Path(args.report)
        if not report_dir.exists():
            print(f"Error: Report directory not found: {report_dir}")
            sys.exit(1)

        print(f"Generating report from: {report_dir}")

        # Load existing results from CSV files
        results = {}
        for csv_file in report_dir.glob("*_results.csv"):
            scenario_name = csv_file.stem.replace("_results", "")
            # Load CSV and convert to RunResult objects
            import csv

            from src.experiments.runner import RunResult

            scenario_results = []
            with open(csv_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Extract metrics (all columns except condition, replicate, seed, duration)
                    metrics = {}
                    for key, value in row.items():
                        if key not in ["condition", "replicate", "seed", "duration_seconds"]:
                            with contextlib.suppress(ValueError, TypeError):
                                metrics[key] = float(value)

                    scenario_results.append(
                        RunResult(
                            condition_name=row["condition"],
                            replicate=int(row["replicate"]),
                            seed=int(row["seed"]),
                            metrics=metrics,
                            duration_seconds=float(row["duration_seconds"]),
                        )
                    )

            results[scenario_name] = scenario_results

        if not results:
            print(f"Error: No results CSV files found in {report_dir}")
            sys.exit(1)

        # Generate report
        runner = BenchmarkRunner(suite_dir=".")  # Dummy, won't be used
        runner.generate_report(results, str(report_dir))

        # Handle HF push if requested
        if args.push_hf:
            push_to_huggingface(str(report_dir), args.push_hf)

        return

    # Execution mode
    if args.suite:
        # Run full benchmark suite
        runner = BenchmarkRunner(
            suite_dir=args.suite,
            architectures=args.architectures,
        )

        try:
            results = runner.run_suite()
        except Exception as e:
            print(f"Error running benchmark suite: {e}")
            sys.exit(1)

        # Determine output directory
        if results:
            # Use the output_dir from first scenario result's config
            first_scenario_results = next(iter(results.values()))
            if first_scenario_results:
                # Extract parent directory (the benchmark suite output dir)
                output_dir = (
                    Path(first_scenario_results[0].trajectory_path).parent.parent
                    if first_scenario_results[0].trajectory_path
                    else Path("data/experiments/benchmarks")
                )
            else:
                output_dir = Path("data/experiments/benchmarks")
        else:
            output_dir = Path("data/experiments/benchmarks")

        # Generate report
        runner.generate_report(results, str(output_dir))

        # Handle HF push if requested
        if args.push_hf:
            push_to_huggingface(str(output_dir), args.push_hf)

    elif args.scenario:
        # Run single scenario
        runner = BenchmarkRunner(
            suite_dir=str(Path(args.scenario).parent),
            architectures=args.architectures,
        )

        try:
            single_scenario_results: list[RunResult] = runner.run_scenario(args.scenario)
        except Exception as e:
            print(f"Error running scenario: {e}")
            sys.exit(1)

        # Analyze and report
        analyzer = ResultAnalyzer()
        generator = ReportGenerator()

        summaries = analyzer.summarize(single_scenario_results)

        scenario_name_path = Path(args.scenario)
        scenario_name = scenario_name_path.stem
        output_dir = Path("data/experiments/benchmarks") / scenario_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate reports
        report_path_str = str(output_dir / "report.md")

        # Create minimal config
        replicates_count = (
            len(
                [
                    r
                    for r in single_scenario_results
                    if r.condition_name == single_scenario_results[0].condition_name
                ]
            )
            if single_scenario_results
            else 0
        )
        dummy_config = ExperimentConfig(
            name=scenario_name.replace("_", " ").title(),
            description=f"Benchmark scenario: {scenario_name}",
            base={},
            conditions=[],
            replicates=replicates_count,
        )

        generator.to_markdown(dummy_config, summaries, report_path_str)
        generator.to_csv(single_scenario_results, str(output_dir / "results.csv"))

        print(f"\n[OK] Report generated: {report_path_str}")
        print(f"  Results CSV: {output_dir / 'results.csv'}")

        # Handle HF push if requested
        if args.push_hf:
            push_to_huggingface(str(output_dir), args.push_hf)


def push_to_huggingface(results_dir: str, repo_id: str) -> None:
    """Push benchmark results to HuggingFace.

    Args:
        results_dir: Directory containing benchmark results
        repo_id: HuggingFace repository ID
    """
    try:
        from src.experiments.hf_push import HuggingFacePublisher

        publisher = HuggingFacePublisher(repo_id=repo_id)
        dataset_url = publisher.push_benchmark(results_dir)

        print(f"\n{'=' * 70}")
        print("[OK] Published to HuggingFace!")
        print(f"  URL: {dataset_url}")
        print(f"{'=' * 70}\n")

    except ImportError as e:
        print(f"\n[FAIL] Failed to push to HuggingFace: {e}")
        print("To enable HF push, install: pip install huggingface_hub")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] Failed to push to HuggingFace: {e}")
        sys.exit(1)


def hf_push(args: argparse.Namespace) -> None:
    """Standalone HuggingFace push command.

    Args:
        args: Command-line arguments
    """
    directory = Path(args.directory)

    # Handle --assemble flag
    if args.assemble:
        print("Assembling dataset from catalog...")
        import subprocess

        try:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/assemble_dataset.py",
                    "--output-dir",
                    str(directory),
                ],
                check=True,
                capture_output=False,
            )
            print("\nAssembly complete. Proceeding to upload...\n")
        except subprocess.CalledProcessError as e:
            print(f"\n[FAIL] Assembly failed with exit code {e.returncode}")
            sys.exit(1)
        except FileNotFoundError:
            print("\n[FAIL] Assembly script not found at scripts/assemble_dataset.py")
            sys.exit(1)

    # Validate directory exists
    if not directory.exists():
        print(f"ERROR: Directory not found: {directory}")
        sys.exit(1)

    # Check for expected files
    expected_files = ["metadata.json", "catalog.parquet"]
    missing_files = [f for f in expected_files if not (directory / f).exists()]

    if missing_files:
        print(f"WARNING: Expected files missing from {directory}:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nContinuing anyway...")

    # Push to HuggingFace
    push_to_huggingface(str(directory), args.repo_id)


if __name__ == "__main__":
    main()
