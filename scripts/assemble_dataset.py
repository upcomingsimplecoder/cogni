#!/usr/bin/env python3
"""Assemble HuggingFace dataset from DuckDB catalog and trajectory files.

Collects all trajectory data into a unified directory structure for upload:
- data/hf_dataset/
  ├── README.md
  ├── metadata.json
  ├── catalog.parquet
  └── runs/{run_id}/
      ├── agent_snapshots.parquet
      ├── emergence_events.parquet
      └── metadata.json

Usage:
    python scripts/assemble_dataset.py
    python scripts/assemble_dataset.py --output-dir custom/path
    python scripts/assemble_dataset.py --catalog path/to/catalog.duckdb
"""

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def export_catalog_to_parquet(conn, output_path: Path) -> bool:
    """Export DuckDB catalog to Parquet using pandas.

    Args:
        conn: DuckDB connection
        output_path: Output parquet file path

    Returns:
        True if successful, False otherwise
    """
    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas is required for Parquet export")
        print("Install with: pip install pandas pyarrow")
        return False

    try:
        # Query all runs and convert to DataFrame
        df = conn.execute("SELECT * FROM runs").fetchdf()
        df.to_parquet(output_path, index=False)
        return True
    except Exception as e:
        print(f"ERROR: Failed to export catalog to Parquet: {e}")
        return False


def copy_run_files(run_path: Path, output_run_dir: Path) -> dict[str, bool]:
    """Copy run files (Parquet or JSONL) to output directory.

    Args:
        run_path: Source run directory
        output_run_dir: Destination run directory

    Returns:
        Dict with status: {"parquet": bool, "metadata": bool, "jsonl_converted": bool}
    """
    status = {"parquet": False, "metadata": False, "jsonl_converted": False}

    output_run_dir.mkdir(parents=True, exist_ok=True)

    # Copy Parquet files if they exist
    parquet_files = [
        "agent_snapshots.parquet",
        "emergence_events.parquet",
    ]

    parquet_exists = all((run_path / f).exists() for f in parquet_files)

    if parquet_exists:
        for filename in parquet_files:
            src = run_path / filename
            dst = output_run_dir / filename
            shutil.copy2(src, dst)
        status["parquet"] = True
    else:
        # Try to convert JSONL to Parquet
        jsonl_file = run_path / "trajectory.jsonl"
        if jsonl_file.exists():
            try:
                from src.trajectory.parquet import ParquetExporter

                # Load JSONL and convert
                from src.trajectory.schema import (
                    AgentSnapshot,
                    EmergenceSnapshot,
                    RunMetadata,
                    TrajectoryDataset,
                )

                metadata = None
                agent_snapshots = []
                emergence_events = []

                with open(jsonl_file) as f:
                    for line in f:
                        data = json.loads(line.strip())
                        data_type = data.pop("type", "unknown")

                        if data_type == "metadata":
                            metadata = RunMetadata(**data)
                        elif data_type == "agent_snapshot":
                            agent_snapshots.append(AgentSnapshot(**data))
                        elif data_type == "emergence":
                            emergence_events.append(EmergenceSnapshot(**data))

                if metadata:
                    dataset = TrajectoryDataset(
                        metadata=metadata,
                        agent_snapshots=agent_snapshots,
                        emergence_events=emergence_events,
                    )
                    ParquetExporter.export(dataset, str(output_run_dir))
                    status["jsonl_converted"] = True
                else:
                    print(f"  WARNING: No metadata found in {jsonl_file}")
            except ImportError:
                print(f"  WARNING: pyarrow not installed, skipping JSONL conversion for {run_path.name}")
            except Exception as e:
                print(f"  WARNING: Failed to convert JSONL for {run_path.name}: {e}")

    # Copy metadata.json if it exists
    metadata_file = run_path / "metadata.json"
    if metadata_file.exists():
        shutil.copy2(metadata_file, output_run_dir / "metadata.json")
        status["metadata"] = True

    return status


def generate_readme(output_dir: Path) -> None:
    """Generate placeholder README.md for the dataset.

    Args:
        output_dir: Dataset output directory
    """
    readme_path = output_dir / "README.md"

    content = """# AUTOCOG Trajectory Dataset

This dataset contains simulation trajectories from the AUTOCOG multi-agent cognitive architecture benchmark.

## Dataset Structure

- `catalog.parquet`: Index of all simulation runs with metadata
- `metadata.json`: Global dataset metadata
- `runs/{run_id}/`: Individual run data
  - `agent_snapshots.parquet`: Per-tick agent state
  - `emergence_events.parquet`: Emergent behavior events
  - `metadata.json`: Run metadata

## Usage

```python
import pandas as pd

# Load catalog
catalog = pd.read_parquet("catalog.parquet")

# Load specific run
run_id = catalog.iloc[0]["run_id"]
snapshots = pd.read_parquet(f"runs/{run_id}/agent_snapshots.parquet")
```

## Schema Version

See `metadata.json` for schema version information.

## Citation

If you use this dataset, please cite:

```
@dataset{autocog_trajectories,
  title={AUTOCOG Multi-Agent Trajectory Dataset},
  year={2026},
  publisher={HuggingFace},
}
```
"""

    with open(readme_path, "w") as f:
        f.write(content)


def main() -> None:
    """Main assembly logic."""
    parser = argparse.ArgumentParser(
        description="Assemble HuggingFace dataset from DuckDB catalog"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/hf_dataset",
        help="Output directory for assembled dataset",
    )
    parser.add_argument(
        "--catalog",
        type=str,
        default="data/catalog.duckdb",
        help="Path to DuckDB catalog file",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    catalog_path = Path(args.catalog)

    # Check catalog exists
    if not catalog_path.exists():
        print(f"ERROR: Catalog not found at {catalog_path}")
        print("Run simulations with trajectory recording enabled to create the catalog.")
        sys.exit(1)

    # Check dependencies
    try:
        import duckdb
    except ImportError:
        print("ERROR: duckdb is required")
        print("Install with: pip install duckdb")
        sys.exit(1)

    try:
        import pandas  # noqa: F401
        import pyarrow  # noqa: F401
    except ImportError:
        print("ERROR: pandas and pyarrow are required")
        print("Install with: pip install pandas pyarrow")
        sys.exit(1)

    print(f"Assembling dataset to: {output_dir}")
    print(f"Source catalog: {catalog_path}")
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Connect to catalog
    conn = duckdb.connect(str(catalog_path))

    # Export catalog to Parquet
    print("Exporting catalog to Parquet...")
    catalog_parquet = output_dir / "catalog.parquet"
    if not export_catalog_to_parquet(conn, catalog_parquet):
        conn.close()
        sys.exit(1)
    print(f"  ✓ {catalog_parquet}")

    # Query all runs
    runs = conn.execute("""
        SELECT run_id, path, has_parquet
        FROM runs
        ORDER BY timestamp DESC
    """).fetchall()

    conn.close()

    print(f"\nProcessing {len(runs)} runs...")
    print()

    runs_dir = output_dir / "runs"
    runs_dir.mkdir(exist_ok=True)

    included_count = 0
    skipped_count = 0
    architectures = set()

    for run_id, path, has_parquet in runs:
        run_path = Path(path)

        if not run_path.exists():
            print(f"  ✗ {run_id}: Path not found ({path})")
            skipped_count += 1
            continue

        print(f"  Processing {run_id}...")

        try:
            output_run_dir = runs_dir / run_id
            status = copy_run_files(run_path, output_run_dir)

            if status["parquet"] or status["jsonl_converted"]:
                included_count += 1

                # Extract architecture from metadata
                metadata_file = output_run_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                        arch = metadata.get("architecture")
                        if arch and arch != "unknown":
                            architectures.add(arch)

                print(f"    ✓ Included (parquet: {status['parquet']}, converted: {status['jsonl_converted']})")
            else:
                print(f"    ✗ Skipped (no valid data)")
                skipped_count += 1
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            skipped_count += 1

    # Generate global metadata
    print("\nGenerating global metadata...")
    global_metadata = {
        "assembled_at": datetime.now(timezone.utc).isoformat(),
        "total_runs": len(runs),
        "total_runs_included": included_count,
        "runs_skipped": skipped_count,
        "schema_version": "1.0.0",
        "architectures": sorted(architectures),
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(global_metadata, f, indent=2)
    print(f"  ✓ {metadata_path}")

    # Generate README
    print("\nGenerating README...")
    generate_readme(output_dir)
    print(f"  ✓ {output_dir / 'README.md'}")

    # Summary
    print()
    print("=" * 70)
    print("Assembly complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  Total runs: {len(runs)}")
    print(f"  Included: {included_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Architectures: {', '.join(sorted(architectures)) or 'None'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
