"""Run catalog: DuckDB-backed index of all simulation runs.

Indexes run metadata for fast search/filter/compare across hundreds of runs.
Requires: duckdb (optional dependency, install with pip install cogniarch[data])
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path


class RunCatalog:
    """DuckDB-backed catalog of simulation runs.

    Scans data directories for trajectory runs and maintains a queryable index.
    Works with both JSONL and Parquet run formats.
    """

    def __init__(self, data_dir: str = "data", db_path: str | None = None):
        """Initialize catalog.

        Args:
            data_dir: Base data directory to scan for runs
            db_path: Path to DuckDB file (default: {data_dir}/catalog.duckdb)

        Raises:
            ImportError: If duckdb is not installed
        """
        # Import duckdb here (not at module level) so module can be imported without it
        try:
            import duckdb
        except ImportError:
            raise ImportError(
                "duckdb is required for the run catalog.\nInstall with: pip install cogniarch[data]"
            ) from None
        self.data_dir = Path(data_dir)
        self.db_path = db_path or str(self.data_dir / "catalog.duckdb")
        self._conn = duckdb.connect(self.db_path)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create runs table if not exists."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id VARCHAR PRIMARY KEY,
                timestamp VARCHAR,
                seed INTEGER,
                architecture VARCHAR,
                num_agents INTEGER,
                max_ticks INTEGER,
                actual_ticks INTEGER,
                agents_alive INTEGER,
                agents_dead INTEGER,
                has_parquet BOOLEAN,
                path VARCHAR,
                config VARCHAR,
                indexed_at VARCHAR
            )
        """)

    def rebuild(self) -> int:
        """Scan data_dir for all runs, rebuild catalog from scratch.

        Looks for directories containing trajectory.jsonl or metadata.json.

        Returns:
            Number of runs indexed
        """
        # Clear existing data
        self._conn.execute("DELETE FROM runs")

        count = 0
        trajectories_dir = self.data_dir / "trajectories"
        if trajectories_dir.exists():
            for run_dir in trajectories_dir.iterdir():
                if run_dir.is_dir():
                    try:
                        self.register_run(str(run_dir))
                        count += 1
                    except Exception:
                        pass  # Skip malformed runs
        return count

    def register_run(self, run_dir: str) -> None:
        """Register a single run into the catalog.

        Extracts metadata from the run directory (prefers metadata.json sidecar
        from Parquet export, falls back to JSONL first line).

        Args:
            run_dir: Path to run directory

        Raises:
            ValueError: If no valid metadata found in directory
        """
        run_path = Path(run_dir)
        metadata = None

        # Prefer metadata.json (from Parquet export)
        metadata_file = run_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
                # Remove schema_version if present
                metadata.pop("schema_version", None)
        else:
            # Fall back to first line of trajectory.jsonl
            jsonl_file = run_path / "trajectory.jsonl"
            if jsonl_file.exists():
                with open(jsonl_file) as f:
                    first_line = f.readline().strip()
                    if first_line:
                        data = json.loads(first_line)
                        if data.get("type") == "metadata":
                            metadata = data

        if not metadata:
            raise ValueError(f"No valid metadata found in {run_dir}")

        # Extract fields
        run_id = metadata["run_id"]
        timestamp = metadata["timestamp"]
        seed = metadata["seed"]
        architecture = metadata.get("architecture", "unknown")
        num_agents = metadata["num_agents"]
        max_ticks = metadata["max_ticks"]
        actual_ticks = metadata.get("actual_ticks", 0)

        # Extract final_state
        final_state = metadata.get("final_state", {})
        agents_alive = final_state.get("agents_alive", 0)
        agents_dead = final_state.get("agents_dead", 0)

        # Check for Parquet format
        has_parquet = (run_path / "agent_snapshots.parquet").exists()

        # Serialize config as JSON string
        config = metadata.get("config", {})
        config_json = json.dumps(config)

        # Current timestamp
        indexed_at = datetime.now(UTC).isoformat()

        # UPSERT (INSERT OR REPLACE)
        self._conn.execute(
            """
            INSERT OR REPLACE INTO runs (
                run_id, timestamp, seed, architecture, num_agents,
                max_ticks, actual_ticks, agents_alive, agents_dead,
                has_parquet, path, config, indexed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
                run_id,
                timestamp,
                seed,
                architecture,
                num_agents,
                max_ticks,
                actual_ticks,
                agents_alive,
                agents_dead,
                has_parquet,
                str(run_path),
                config_json,
                indexed_at,
            ],
        )

    def list_runs(
        self,
        *,
        architecture: str | None = None,
        min_ticks: int | None = None,
        min_agents: int | None = None,
        seed: int | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Query runs with optional filters.

        Args:
            architecture: Filter by cognitive architecture name
            min_ticks: Minimum actual ticks completed
            min_agents: Minimum number of agents
            seed: Filter by random seed
            limit: Maximum results to return

        Returns:
            List of dicts with run metadata
        """
        # Build WHERE clauses
        where_clauses = []
        params: list[str | int] = []

        if architecture is not None:
            where_clauses.append("architecture = ?")
            params.append(architecture)

        if min_ticks is not None:
            where_clauses.append("actual_ticks >= ?")
            params.append(min_ticks)

        if min_agents is not None:
            where_clauses.append("num_agents >= ?")
            params.append(min_agents)

        if seed is not None:
            where_clauses.append("seed = ?")
            params.append(seed)

        # Build query
        query = "SELECT * FROM runs"
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        result = self._conn.execute(query, params).fetchall()

        # Convert to list of dicts
        columns = [desc[0] for desc in self._conn.description]
        return [dict(zip(columns, row, strict=False)) for row in result]

    def get_run(self, run_id: str) -> dict | None:
        """Get metadata for a specific run.

        Args:
            run_id: Run identifier

        Returns:
            Dict with run metadata, or None if not found
        """
        result = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", [run_id]).fetchone()

        if result is None:
            return None

        columns = [desc[0] for desc in self._conn.description]
        return dict(zip(columns, result, strict=False))

    def compare_runs(self, run_ids: list[str]) -> dict:
        """Side-by-side comparison of run metadata.

        Args:
            run_ids: List of run IDs to compare

        Returns:
            Dict with 'runs' key containing list of run dicts
        """
        if not run_ids:
            return {"runs": []}

        # Build placeholders for parameterized query
        placeholders = ", ".join(["?"] * len(run_ids))
        query = f"SELECT * FROM runs WHERE run_id IN ({placeholders})"

        result = self._conn.execute(query, run_ids).fetchall()

        columns = [desc[0] for desc in self._conn.description]
        runs = [dict(zip(columns, row, strict=False)) for row in result]

        return {"runs": runs}

    def stats(self) -> dict:
        """Catalog summary statistics.

        Returns:
            Dict with: total_runs, total_ticks, architectures,
            earliest_run, latest_run
        """
        # Get aggregate stats
        result = self._conn.execute("""
            SELECT
                COUNT(*) as total_runs,
                COALESCE(SUM(actual_ticks), 0) as total_ticks,
                MIN(timestamp) as earliest_run,
                MAX(timestamp) as latest_run
            FROM runs
        """).fetchone()

        if result is None:
            return {
                "total_runs": 0,
                "total_ticks": 0,
                "architectures": [],
                "earliest_run": None,
                "latest_run": None,
            }

        total_runs, total_ticks, earliest_run, latest_run = result

        # Get distinct architectures
        arch_result = self._conn.execute(
            "SELECT DISTINCT architecture FROM runs ORDER BY architecture"
        ).fetchall()
        architectures = [row[0] for row in arch_result]

        return {
            "total_runs": total_runs,
            "total_ticks": total_ticks,
            "architectures": architectures,
            "earliest_run": earliest_run,
            "latest_run": latest_run,
        }

    def close(self) -> None:
        """Close the DuckDB connection."""
        if self._conn:
            self._conn.close()
