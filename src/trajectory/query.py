"""Cross-run query interface using DuckDB on Parquet files.

Enables analytical queries across multiple runs without loading
all data into memory. Queries Parquet files directly via DuckDB.
Requires: duckdb (optional dependency, install with pip install cogniarch[data])
"""

from __future__ import annotations

from pathlib import Path


class TrajectoryQuery:
    """Query interface for Parquet trajectory data.

    Creates DuckDB views over all Parquet files found in the data directory,
    enabling SQL queries across multiple simulation runs.
    """

    def __init__(self, data_dir: str = "data/trajectories"):
        """Initialize with data directory containing run subdirectories.

        Scans for all agent_snapshots.parquet and emergence_events.parquet files
        and creates DuckDB views over them.

        Args:
            data_dir: Directory containing run subdirectories with Parquet files

        Raises:
            ImportError: If duckdb is not installed
        """
        try:
            import duckdb
        except ImportError:
            raise ImportError(
                "duckdb is required for trajectory queries.\n"
                "Install with: pip install cogniarch[data]"
            ) from None
        self.data_dir = Path(data_dir)
        self._conn = duckdb.connect()  # in-memory
        self._setup_views()

    def _setup_views(self) -> None:
        """Create DuckDB views over all Parquet files found.

        Creates:
        - 'snapshots' view over all agent_snapshots.parquet files
        - 'events' view over all emergence_events.parquet files

        Uses glob patterns to find files. If no files found, creates empty views.
        """
        # Find all parquet files
        snapshot_files = list(self.data_dir.glob("*/agent_snapshots.parquet"))
        event_files = list(self.data_dir.glob("*/emergence_events.parquet"))

        if snapshot_files:
            # Use forward slashes for DuckDB compatibility on Windows
            # Enable filename column to identify which run each row came from
            snapshot_pattern = str(self.data_dir / "*/agent_snapshots.parquet")
            snapshot_pattern = snapshot_pattern.replace("\\", "/")
            self._conn.execute(
                f"CREATE VIEW snapshots AS SELECT * FROM "
                f"read_parquet('{snapshot_pattern}', "
                f"union_by_name=true, filename=true)"
            )

        if event_files:
            event_pattern = str(self.data_dir / "*/emergence_events.parquet")
            event_pattern = event_pattern.replace("\\", "/")
            self._conn.execute(
                f"CREATE VIEW events AS SELECT * FROM "
                f"read_parquet('{event_pattern}', "
                f"union_by_name=true, filename=true)"
            )

    def sql(self, query: str) -> list[dict]:
        """Execute raw SQL against trajectory data.

        Available tables/views:
        - 'snapshots': all agent_snapshots across runs
        - 'events': all emergence_events across runs

        Args:
            query: SQL query string

        Returns:
            List of result dicts (one dict per row)

        Raises:
            duckdb.Error: If query is invalid or tables don't exist
        """
        try:
            result = self._conn.execute(query)
            columns = [desc[0] for desc in result.description]
            return [dict(zip(columns, row, strict=False)) for row in result.fetchall()]
        except Exception as e:
            # Check if error is due to missing views (no data)
            if "snapshots" in str(e) or "events" in str(e):
                raise ValueError(
                    f"No Parquet data found in {self.data_dir}. "
                    "Ensure runs have been exported to Parquet format."
                ) from e
            raise

    def trait_evolution(self, agent_id: str, run_id: str | None = None) -> list[dict]:
        """Get trait values over time for an agent.

        Args:
            agent_id: Agent identifier
            run_id: Optional run identifier to filter by (matches against path)

        Returns:
            List of dicts with tick + all 6 trait values, ordered by tick
        """
        where = f"WHERE agent_id = '{agent_id}'"
        if run_id:
            where += f" AND filename LIKE '%{run_id}%'"

        return self.sql(f"""
            SELECT tick, cooperation_tendency, curiosity, risk_tolerance,
                   resource_sharing, aggression, sociability
            FROM snapshots
            {where}
            ORDER BY tick
        """)

    def action_distribution(
        self,
        *,
        archetype: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, int]:
        """Count actions by type, optionally filtered.

        Args:
            archetype: Filter by agent archetype
            run_id: Filter by run ID (matches against path)

        Returns:
            Dict mapping action_type to count
        """
        conditions = []
        if archetype:
            conditions.append(f"archetype = '{archetype}'")
        if run_id:
            conditions.append(f"filename LIKE '%{run_id}%'")

        where = "WHERE " + " AND ".join(conditions) if conditions else ""

        rows = self.sql(f"""
            SELECT action_type, COUNT(*) as count
            FROM snapshots
            {where}
            GROUP BY action_type
            ORDER BY count DESC
        """)
        return {row["action_type"]: row["count"] for row in rows}

    def survival_curve(self, run_id: str) -> list[dict]:
        """Agents alive per tick for a run.

        Args:
            run_id: Run identifier (matches against path)

        Returns:
            List of dicts with tick and agents_alive count
        """
        return self.sql(f"""
            SELECT tick, COUNT(*) as agents_alive
            FROM snapshots
            WHERE alive = true AND filename LIKE '%{run_id}%'
            GROUP BY tick
            ORDER BY tick
        """)

    def cross_run_comparison(self, metric: str, group_by: str = "archetype") -> list[dict]:
        """Aggregate a metric across runs grouped by a column.

        Args:
            metric: Column name to aggregate (e.g., 'health', 'hunger',
                    'cooperation_tendency')
            group_by: Column to group by (e.g., 'archetype', 'action_type')

        Returns:
            List of dicts with group, avg, min, max values
        """
        return self.sql(f"""
            SELECT {group_by},
                   AVG({metric}) as avg_{metric},
                   MIN({metric}) as min_{metric},
                   MAX({metric}) as max_{metric},
                   COUNT(*) as sample_count
            FROM snapshots
            GROUP BY {group_by}
            ORDER BY avg_{metric} DESC
        """)

    def close(self) -> None:
        """Close the DuckDB connection."""
        if self._conn:
            self._conn.close()
