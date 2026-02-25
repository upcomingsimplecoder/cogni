"""CLI for trajectory analysis.

Usage:
    python -m src.analysis report data/trajectories/abc123/
    python -m src.analysis report --compare data/trajectories/run1/ data/trajectories/run2/
    python -m src.analysis report --aggregate data/trajectories/
    python -m src.analysis catalog [--data-dir data]
    python -m src.analysis list [--architecture reactive] [--min-ticks 100]
    python -m src.analysis show <run_id>
    python -m src.analysis phenomena <run_dir> [--output phenomena.json]
    python -m src.analysis query "SELECT archetype, COUNT(*) FROM snapshots GROUP BY archetype"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.trajectory.loader import TrajectoryLoader


def cmd_report(args: argparse.Namespace) -> None:
    """Generate analysis reports (original CLI behavior)."""
    from src.analysis.reports import AnalysisReportGenerator

    generator = AnalysisReportGenerator()

    if args.aggregate:
        base_path = Path(args.paths[0])
        trajectory_files = list(base_path.rglob("trajectory.jsonl"))

        if not trajectory_files:
            print(f"No trajectory files found in {base_path}")
            sys.exit(1)

        print(f"Found {len(trajectory_files)} trajectory files")

        datasets = []
        for traj_file in trajectory_files:
            try:
                dataset = TrajectoryLoader.from_jsonl(str(traj_file))
                datasets.append(dataset)
                print(f"  Loaded: {traj_file.parent.name}")
            except Exception as e:
                print(f"  Failed to load {traj_file}: {e}")

        if not datasets:
            print("No valid trajectories loaded")
            sys.exit(1)

        output_path = args.output or "analysis_aggregate_report.md"
        generator.generate_comparison_report(datasets, output_path)
        print(f"\nAggregate report written to: {output_path}")

    elif args.compare:
        datasets = []
        for path in args.paths:
            try:
                dataset = TrajectoryLoader.from_run_dir(path)
                datasets.append(dataset)
                print(f"Loaded: {path}")
            except Exception as e:
                print(f"Failed to load {path}: {e}")
                sys.exit(1)

        output_path = args.output or "analysis_comparison_report.md"
        generator.generate_comparison_report(datasets, output_path)
        print(f"\nComparison report written to: {output_path}")

    else:
        run_path = args.paths[0]
        try:
            dataset = TrajectoryLoader.from_run_dir(run_path)
            print(f"Loaded: {run_path}")
        except Exception as e:
            print(f"Failed to load {run_path}: {e}")
            sys.exit(1)

        output_path = args.output or f"analysis_report_{dataset.metadata.run_id[:8]}.md"
        generator.generate_run_report(dataset, output_path)
        print(f"\nReport written to: {output_path}")


def cmd_catalog(args: argparse.Namespace) -> None:
    """Rebuild the run catalog from disk."""
    from src.trajectory.catalog import RunCatalog

    catalog = RunCatalog(data_dir=args.data_dir)
    count = catalog.rebuild()
    print(f"Indexed {count} runs in catalog")

    stats = catalog.stats()
    if stats["total_runs"] > 0:
        print(f"  Total ticks: {stats['total_ticks']}")
        archs = stats.get("architectures", [])
        if archs:
            print(f"  Architectures: {', '.join(archs)}")
    catalog.close()


def cmd_list(args: argparse.Namespace) -> None:
    """List runs in the catalog with optional filters."""
    from src.trajectory.catalog import RunCatalog

    catalog = RunCatalog(data_dir=args.data_dir)
    runs = catalog.list_runs(
        architecture=args.architecture,
        min_ticks=args.min_ticks,
        min_agents=args.min_agents,
        seed=args.seed,
        limit=args.limit,
    )

    if not runs:
        print("No runs found. Try: python -m src.analysis catalog")
        catalog.close()
        return

    print(f"{'Run ID':<20} {'Arch':<15} {'Agents':<8} {'Ticks':<8} {'Alive':<7} {'Parquet'}")
    print("-" * 75)
    for run in runs:
        print(
            f"{run['run_id']:<20} "
            f"{run.get('architecture', '-'):<15} "
            f"{run.get('num_agents', '-'):<8} "
            f"{run.get('actual_ticks', '-'):<8} "
            f"{run.get('agents_alive', '-'):<7} "
            f"{'✓' if run.get('has_parquet') else '-'}"
        )
    catalog.close()


def cmd_show(args: argparse.Namespace) -> None:
    """Show details for a specific run."""
    from src.trajectory.catalog import RunCatalog

    catalog = RunCatalog(data_dir=args.data_dir)
    run = catalog.get_run(args.run_id)

    if not run:
        print(f"Run not found: {args.run_id}")
        catalog.close()
        sys.exit(1)

    for key, value in run.items():
        if key == "config":
            continue  # Skip verbose config
        print(f"  {key}: {value}")
    catalog.close()


def cmd_phenomena(args: argparse.Namespace) -> None:
    """Compute and display phenomena for a run."""
    from src.analysis.phenomena import PhenomenaComputer

    try:
        dataset = TrajectoryLoader.from_run_dir(args.run_dir)
    except Exception as e:
        print(f"Failed to load {args.run_dir}: {e}")
        sys.exit(1)

    computer = PhenomenaComputer()
    report = computer.compute_all(dataset)

    if args.output:
        report.save(args.run_dir)
        print(f"Phenomena saved to: {Path(args.run_dir) / 'phenomena.json'}")

    # Print summary
    print(f"\n=== Phenomena Report: {report.run_id} ===\n")

    print(f"Value Drift Curves: {len(report.value_drift_curves)}")
    for curve in report.value_drift_curves[:10]:
        print(
            f"  {curve.agent_name}.{curve.trait_name}: {curve.direction} "
            f"(drift={curve.total_drift:.4f}, rate={curve.drift_rate:.6f})"
        )

    print(f"\nNorm Convergence: {len(report.norm_convergence)}")
    for norm in report.norm_convergence:
        status = "CONVERGED" if norm.converged else "diverging"
        print(f"  {norm.trait_name}: {status} (ratio={norm.convergence_ratio:.3f})")

    print(f"\nFailure Signatures: {len(report.failure_signatures)}")
    for sig in report.failure_signatures:
        factors = ", ".join(sig.contributing_factors) if sig.contributing_factors else "unknown"
        print(
            f"  tick {sig.trigger_tick}: {len(sig.agents_dying)} deaths "
            f"(rate={sig.death_rate:.2f}, factors: {factors})"
        )


def cmd_query(args: argparse.Namespace) -> None:
    """Execute SQL query against Parquet trajectory data."""
    from src.trajectory.query import TrajectoryQuery

    try:
        query = TrajectoryQuery(data_dir=args.data_dir)
    except Exception as e:
        print(f"Failed to initialize query engine: {e}")
        sys.exit(1)

    try:
        results = query.sql(args.sql)
    except Exception as e:
        print(f"Query error: {e}")
        query.close()
        sys.exit(1)

    if not results:
        print("No results")
    else:
        # Print as formatted table
        headers = list(results[0].keys())
        print("\t".join(headers))
        for row in results:
            print("\t".join(str(row.get(h, "")) for h in headers))

    query.close()


def main() -> None:
    """Main entry point for analysis CLI."""
    parser = argparse.ArgumentParser(
        description="AUTOCOG trajectory analysis tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # report — original CLI behavior
    report_parser = subparsers.add_parser("report", help="Generate analysis reports")
    report_parser.add_argument("paths", nargs="+", help="Path(s) to trajectory directories")
    report_parser.add_argument("--compare", action="store_true", help="Compare multiple runs")
    report_parser.add_argument("--aggregate", action="store_true", help="Aggregate runs in tree")
    report_parser.add_argument("--output", "-o", help="Output report path")

    # catalog — rebuild index
    catalog_parser = subparsers.add_parser("catalog", help="Rebuild the run catalog")
    catalog_parser.add_argument("--data-dir", default="data", help="Data directory (default: data)")

    # list — query catalog
    list_parser = subparsers.add_parser("list", help="List runs in catalog")
    list_parser.add_argument("--data-dir", default="data", help="Data directory")
    list_parser.add_argument("--architecture", help="Filter by architecture")
    list_parser.add_argument("--min-ticks", type=int, help="Minimum ticks completed")
    list_parser.add_argument("--min-agents", type=int, help="Minimum agents")
    list_parser.add_argument("--seed", type=int, help="Filter by seed")
    list_parser.add_argument("--limit", type=int, default=50, help="Max results")

    # show — run details
    show_parser = subparsers.add_parser("show", help="Show run details")
    show_parser.add_argument("run_id", help="Run identifier")
    show_parser.add_argument("--data-dir", default="data", help="Data directory")

    # phenomena — compute derived metrics
    phenomena_parser = subparsers.add_parser("phenomena", help="Compute phenomena from run")
    phenomena_parser.add_argument("run_dir", help="Path to run directory")
    phenomena_parser.add_argument("--output", action="store_true", help="Save to phenomena.json")

    # query — SQL on Parquet
    query_parser = subparsers.add_parser("query", help="SQL query on Parquet data")
    query_parser.add_argument("sql", help="SQL query string")
    query_parser.add_argument(
        "--data-dir", default="data/trajectories", help="Trajectories directory"
    )

    args = parser.parse_args()

    if args.command is None:
        # Backward compat: if no subcommand, try legacy behavior
        # Re-parse as report command
        parser.print_help()
        sys.exit(1)

    commands = {
        "report": cmd_report,
        "catalog": cmd_catalog,
        "list": cmd_list,
        "show": cmd_show,
        "phenomena": cmd_phenomena,
        "query": cmd_query,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
