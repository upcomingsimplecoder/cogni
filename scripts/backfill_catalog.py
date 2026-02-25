#!/usr/bin/env python3
"""Backfill NULL architecture values in the DuckDB run catalog.

Infers architecture from:
1. config.default_architecture field in the config JSON
2. Directory name (fallback for older runs)

Usage:
    python scripts/backfill_catalog.py
"""

import json
import sys
from pathlib import Path


def infer_architecture_from_config(config_json: str) -> str | None:
    """Extract default_architecture from config JSON string.

    Args:
        config_json: JSON string of run config

    Returns:
        Architecture name or None if not found
    """
    try:
        config = json.loads(config_json)
        return config.get("default_architecture")
    except (json.JSONDecodeError, KeyError):
        return None


def infer_architecture_from_path(path: str) -> str | None:
    """Infer architecture from directory path.

    Searches for known architecture names in the path components.

    Args:
        path: File system path to run directory

    Returns:
        Architecture name or None if not found
    """
    known_architectures = [
        "reactive",
        "cautious",
        "optimistic",
        "social",
        "dual_process",
        "planning",
    ]

    path_lower = path.lower()
    for arch in known_architectures:
        if arch in path_lower:
            return arch

    return None


def main() -> None:
    """Backfill NULL architecture values in the catalog."""
    catalog_path = Path("data/catalog.duckdb")

    if not catalog_path.exists():
        print("No catalog found at data/catalog.duckdb")
        print("Run a simulation with trajectory recording to create the catalog.")
        sys.exit(1)

    try:
        import duckdb
    except ImportError:
        print("ERROR: duckdb is required for this script")
        print("Install with: pip install duckdb")
        sys.exit(1)

    print(f"Connecting to catalog: {catalog_path}")
    conn = duckdb.connect(str(catalog_path))

    # Query runs with NULL or 'unknown' architecture
    result = conn.execute("""
        SELECT run_id, path, config, architecture
        FROM runs
        WHERE architecture IS NULL OR architecture = 'unknown'
    """).fetchall()

    if not result:
        print("No runs with NULL or 'unknown' architecture found.")
        conn.close()
        return

    print(f"Found {len(result)} runs to backfill")
    print()

    updated_count = 0
    failed_count = 0

    for run_id, path, config_json, old_arch in result:
        # Try to infer from config first
        inferred_arch = infer_architecture_from_config(config_json)

        # Fallback to path-based inference
        if not inferred_arch:
            inferred_arch = infer_architecture_from_path(path)

        if inferred_arch:
            # Update the run
            conn.execute("""
                UPDATE runs
                SET architecture = ?
                WHERE run_id = ?
            """, [inferred_arch, run_id])

            print(f"[OK] {run_id}: {old_arch or 'NULL'} -> {inferred_arch}")
            updated_count += 1
        else:
            print(f"[SKIP] {run_id}: Could not infer architecture (path: {path})")
            failed_count += 1

    conn.close()

    print()
    print("=" * 60)
    print(f"Backfill complete:")
    print(f"  Updated: {updated_count}")
    print(f"  Failed:  {failed_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
