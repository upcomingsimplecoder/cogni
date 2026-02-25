"""Trajectory recording and dataset management.

Captures full agent trajectories (personality → decisions → outcomes → trait changes)
for building a proprietary dataset of personality-behavior mappings.
"""

from src.trajectory.exporter import TrajectoryExporter
from src.trajectory.loader import TrajectoryLoader
from src.trajectory.recorder import TrajectoryRecorder
from src.trajectory.schema import (
    AgentSnapshot,
    EmergenceSnapshot,
    RunMetadata,
    TrajectoryDataset,
)

__all__ = [
    "AgentSnapshot",
    "EmergenceSnapshot",
    "RunMetadata",
    "TrajectoryDataset",
    "TrajectoryRecorder",
    "TrajectoryExporter",
    "TrajectoryLoader",
]

# Optional exports — available only when pyarrow/duckdb are installed
try:
    from src.trajectory.parquet import ParquetExporter  # noqa: F401

    __all__.append("ParquetExporter")
except ImportError:
    pass

try:
    from src.trajectory.catalog import RunCatalog  # noqa: F401

    __all__.append("RunCatalog")
except ImportError:
    pass

try:
    from src.trajectory.query import TrajectoryQuery  # noqa: F401

    __all__.append("TrajectoryQuery")
except ImportError:
    pass
