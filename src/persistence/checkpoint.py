"""Checkpoint management: save, load, auto-checkpoint, and pruning.

Provides atomic writes and automatic checkpoint lifecycle management.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from src.persistence.serializer import StateSerializer

if TYPE_CHECKING:
    from src.simulation.engine import SimulationEngine


class CheckpointManager:
    """Manages checkpoint lifecycle: save, load, auto-checkpoint, prune."""

    def __init__(
        self,
        checkpoint_dir: str = "data/checkpoints",
        auto_interval: int = 100,
        max_checkpoints: int = 10,
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            auto_interval: Ticks between auto-checkpoints (0 = disabled)
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self._dir = Path(checkpoint_dir)
        self._interval = auto_interval
        self._max = max_checkpoints
        self._serializer = StateSerializer()
        self._last_checkpoint_tick = -1

        # Create checkpoint directory if it doesn't exist
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, engine: SimulationEngine, label: str = "") -> str:
        """Save checkpoint with atomic write.

        Args:
            engine: Simulation engine to save
            label: Optional label for the checkpoint

        Returns:
            Path to saved checkpoint file
        """
        # Serialize state
        data = self._serializer.serialize(engine)

        # Build filename: {tick:06d}_{label}_{timestamp}.json
        tick = engine.state.tick
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        label_part = f"{label}_" if label else ""
        filename = f"{tick:06d}_{label_part}{timestamp}.json"
        filepath = self._dir / filename

        # Atomic write: write to temp file, then rename
        temp_path = self._dir / f".{filename}.tmp"
        try:
            with open(temp_path, "w") as f:
                json.dump(data, f, indent=2)

            # Atomic rename
            os.replace(temp_path, filepath)

            return str(filepath)

        finally:
            # Clean up temp file if it still exists
            if temp_path.exists():
                temp_path.unlink()

    def load(self, path: str, config_override: dict | None = None) -> SimulationEngine:
        """Load checkpoint from file.

        Args:
            path: Path to checkpoint file
            config_override: Optional config overrides for branching

        Returns:
            Reconstructed SimulationEngine
        """
        with open(path) as f:
            data = json.load(f)

        return self._serializer.deserialize(data, config_override)

    def auto_checkpoint(self, engine: SimulationEngine) -> str | None:
        """Auto-checkpoint if interval reached.

        Should be called every tick. Handles pruning automatically.

        Args:
            engine: Simulation engine to checkpoint

        Returns:
            Path to saved checkpoint, or None if not saved
        """
        if self._interval <= 0:
            return None

        tick = engine.state.tick
        if tick - self._last_checkpoint_tick >= self._interval:
            filepath = self.save(engine, label="auto")
            self._last_checkpoint_tick = tick
            self._prune_old()
            return filepath

        return None

    def list_checkpoints(self) -> list[dict]:
        """List available checkpoints with metadata.

        Returns:
            List of checkpoint metadata dicts with keys:
            - path: str
            - tick: int
            - timestamp: str
            - label: str
        """
        checkpoints = []

        for filepath in sorted(self._dir.glob("*.json")):
            # Skip temp files
            if filepath.name.startswith("."):
                continue

            # Parse filename: {tick:06d}_{label}_{timestamp}.json
            parts = filepath.stem.split("_", 2)
            try:
                tick = int(parts[0])
                # Check if there's a label (middle part before timestamp)
                if len(parts) == 3:
                    # Has label
                    label = parts[1]
                    timestamp = parts[2]
                else:
                    # No label
                    label = ""
                    timestamp = parts[1] if len(parts) > 1 else ""

                checkpoints.append(
                    {
                        "path": str(filepath),
                        "tick": tick,
                        "timestamp": timestamp,
                        "label": label,
                    }
                )
            except (ValueError, IndexError):
                # Skip malformed filenames
                continue

        return checkpoints

    def latest_checkpoint(self) -> str | None:
        """Get path to most recent checkpoint.

        Returns:
            Path to latest checkpoint, or None if no checkpoints exist
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None

        # Sort by tick descending
        checkpoints.sort(key=lambda c: c["tick"], reverse=True)
        checkpoint_path: str | None = checkpoints[0]["path"]
        return checkpoint_path

    def _prune_old(self) -> None:
        """Remove oldest checkpoints beyond max_checkpoints."""
        checkpoints = self.list_checkpoints()

        if len(checkpoints) <= self._max:
            return

        # Sort by tick ascending (oldest first)
        checkpoints.sort(key=lambda c: c["tick"])

        # Remove oldest checkpoints
        to_remove = checkpoints[: len(checkpoints) - self._max]
        for checkpoint in to_remove:
            Path(checkpoint["path"]).unlink()
