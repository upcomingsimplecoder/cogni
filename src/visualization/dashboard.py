"""Dashboard generator for trajectory visualization.

Loads trajectory data, compresses it, and embeds into a self-contained HTML file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.trajectory.loader import TrajectoryLoader
from src.trajectory.schema import AgentSnapshot, TrajectoryDataset

# Archetype color mappings (matching src.agents.archetypes)
ARCHETYPE_COLORS = {
    "gatherer": "#22c55e",  # green
    "explorer": "#06b6d4",  # cyan
    "diplomat": "#eab308",  # yellow
    "aggressor": "#ef4444",  # red
    "survivalist": "#f8fafc",  # white
}


class DashboardGenerator:
    """Generate interactive dashboard HTML from trajectory data."""

    def __init__(self, dataset: TrajectoryDataset):
        """Initialize with a loaded trajectory dataset.

        Args:
            dataset: TrajectoryDataset loaded via TrajectoryLoader
        """
        self.dataset = dataset

    @classmethod
    def from_file(cls, trajectory_path: str) -> DashboardGenerator:
        """Load trajectory and create generator.

        Args:
            trajectory_path: Path to trajectory.jsonl file

        Returns:
            DashboardGenerator instance
        """
        dataset = TrajectoryLoader.from_jsonl(trajectory_path)
        return cls(dataset)

    def generate(self, output_path: str) -> None:
        """Generate dashboard HTML file.

        Args:
            output_path: Where to write the output HTML file
        """
        data = self._prepare_data()
        template = self._load_template()

        # Embed data into template
        json_data = json.dumps(data, separators=(",", ":"))
        html = template.replace("/*__DATA__*/", json_data)

        # Write to disk
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html, encoding="utf-8")

    def _prepare_data(self) -> dict[str, Any]:
        """Prepare compressed data for embedding in HTML.

        Returns:
            Dict with metadata, ticks, agents, and events
        """
        return {
            "metadata": {
                "run_id": self.dataset.metadata.run_id,
                "timestamp": self.dataset.metadata.timestamp,
                "seed": self.dataset.metadata.seed,
                "num_agents": self.dataset.metadata.num_agents,
                "max_ticks": self.dataset.metadata.max_ticks,
                "actual_ticks": self.dataset.metadata.actual_ticks,
                "world_width": self.dataset.metadata.config.get("world_width", 64),
                "world_height": self.dataset.metadata.config.get("world_height", 64),
            },
            "agents": self._agent_metadata(),
            "ticks": self._compact_tick_data(),
            "emergence_events": [
                {
                    "tick": e.tick,
                    "type": e.pattern_type,
                    "agents": e.agents_involved,
                    "description": e.description,
                }
                for e in self.dataset.emergence_events
            ],
            "cultural_summary": self._cultural_summary(),
        }

    def _agent_metadata(self) -> list[dict[str, Any]]:
        """Extract agent metadata from dataset.

        Returns:
            List of dicts with id, name, archetype, color
        """
        agents_seen: dict[str, dict[str, Any]] = {}

        for snap in self.dataset.agent_snapshots:
            if snap.agent_id not in agents_seen:
                agents_seen[snap.agent_id] = {
                    "id": snap.agent_id,
                    "name": snap.agent_name,
                    "archetype": snap.archetype,
                    "color": ARCHETYPE_COLORS.get(snap.archetype, "#ffffff"),
                }

        return list(agents_seen.values())

    def _compact_tick_data(self) -> dict[int, list[dict[str, Any]]]:
        """Group snapshots by tick and compress fields.

        Returns:
            Dict mapping tick -> list of compact agent states
        """
        ticks: dict[int, list[dict[str, Any]]] = {}

        for snap in self.dataset.agent_snapshots:
            if snap.tick not in ticks:
                ticks[snap.tick] = []

            compact: dict[str, Any] = {
                "id": snap.agent_id,
                "pos": snap.position,
                "alive": snap.alive,
                "hunger": round(snap.hunger, 1),
                "thirst": round(snap.thirst, 1),
                "energy": round(snap.energy, 1),
                "health": round(snap.health, 1),
                "action": snap.action_type,
                "success": snap.action_succeeded,
                "inventory": snap.inventory,
                "monologue": snap.internal_monologue,
                "intention": snap.intention.get("primary_goal", "unknown"),
            }

            # Cultural data (optional — only when cultural transmission is active)
            if snap.cultural_learning_style:
                compact["cultural"] = {
                    "group": snap.cultural_group_id,
                    "style": snap.cultural_learning_style,
                    "repertoire_size": len(snap.cultural_repertoire),
                    "adopted_count": sum(
                        1
                        for v in snap.cultural_repertoire.values()
                        if isinstance(v, dict) and v.get("adopted", False)
                    ),
                    "events": snap.transmission_events_this_tick,
                }

            ticks[snap.tick].append(compact)

        return ticks

    def _cultural_summary(self) -> dict[int, dict[str, Any]]:
        """Aggregate per-tick cultural stats from snapshots.

        Returns:
            Dict mapping tick -> cultural summary stats.
            Empty dict if no cultural data exists.
        """
        from collections import defaultdict

        tick_data: dict[int, list[AgentSnapshot]] = defaultdict(list)
        has_cultural = False

        for snap in self.dataset.agent_snapshots:
            if snap.cultural_learning_style:
                has_cultural = True
            tick_data[snap.tick].append(snap)

        if not has_cultural:
            return {}

        summary: dict[int, dict[str, Any]] = {}

        for tick, snapshots in tick_data.items():
            groups: set[int] = set()
            total_variants = 0
            total_adopted = 0
            style_distribution: dict[str, int] = {}

            for snap in snapshots:
                if not snap.cultural_learning_style:
                    continue

                if snap.cultural_group_id >= 0:
                    groups.add(snap.cultural_group_id)

                total_variants += len(snap.cultural_repertoire)
                total_adopted += sum(
                    1
                    for v in snap.cultural_repertoire.values()
                    if isinstance(v, dict) and v.get("adopted", False)
                )

                style = snap.cultural_learning_style
                style_distribution[style] = style_distribution.get(style, 0) + 1

            if style_distribution:  # at least one agent had cultural data
                summary[tick] = {
                    "group_count": len(groups),
                    "total_variants": total_variants,
                    "total_adopted": total_adopted,
                    "style_distribution": style_distribution,
                }

        return summary

    def _load_template(self) -> str:
        """Load HTML template from disk.

        Returns:
            HTML template string with /*__DATA__*/ placeholder
        """
        # Try new template first, fallback to old
        template_path = Path(__file__).parent / "templates" / "dashboard_new.html"
        if not template_path.exists():
            template_path = Path(__file__).parent / "templates" / "dashboard.html"
        if not template_path.exists():
            raise FileNotFoundError(f"Dashboard template not found: {template_path}")
        return template_path.read_text(encoding="utf-8")
