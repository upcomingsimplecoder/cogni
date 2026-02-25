"""Trajectory data loader and query utilities.

Load trajectory files from disk and provide filtering/query API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from src.trajectory.schema import (
    AgentSnapshot,
    EmergenceSnapshot,
    RunMetadata,
    TrajectoryDataset,
)

if TYPE_CHECKING:
    pass


class TrajectoryLoader:
    """Load and query trajectory files."""

    @staticmethod
    def from_jsonl(path: str) -> TrajectoryDataset:
        """Load from JSONL file."""
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Trajectory file not found: {path}")

        metadata: RunMetadata | None = None
        snapshots: list[AgentSnapshot] = []
        events: list[EmergenceSnapshot] = []

        with open(path_obj) as f:
            for line in f:
                if not line.strip():
                    continue

                data = json.loads(line)
                record_type = data.pop("type", None)

                if record_type == "metadata":
                    metadata = TrajectoryLoader._dict_to_metadata(data)
                elif record_type == "agent_snapshot":
                    snapshots.append(TrajectoryLoader._dict_to_snapshot(data))
                elif record_type == "emergence":
                    events.append(TrajectoryLoader._dict_to_emergence(data))
                elif record_type == "run_complete" and metadata and "actual_ticks" in data:
                    # Update metadata with final state if present
                    metadata.actual_ticks = data["actual_ticks"]
                    metadata.final_state = data.get("final_state", {})

        if metadata is None:
            raise ValueError("No metadata found in trajectory file")

        return TrajectoryDataset(
            metadata=metadata,
            agent_snapshots=snapshots,
            emergence_events=events,
        )

    @staticmethod
    def from_run_dir(run_dir: str) -> TrajectoryDataset:
        """Load from run directory (looks for trajectory.jsonl)."""
        run_path = Path(run_dir)
        jsonl_path = run_path / "trajectory.jsonl"
        return TrajectoryLoader.from_jsonl(str(jsonl_path))

    @staticmethod
    def filter_agent(dataset: TrajectoryDataset, agent_id: str) -> list[AgentSnapshot]:
        """Get trajectory for single agent."""
        return [s for s in dataset.agent_snapshots if s.agent_id == agent_id]

    @staticmethod
    def filter_ticks(dataset: TrajectoryDataset, start: int, end: int) -> TrajectoryDataset:
        """Slice by tick range (inclusive)."""
        filtered_snapshots = [s for s in dataset.agent_snapshots if start <= s.tick <= end]
        filtered_events = [e for e in dataset.emergence_events if start <= e.tick <= end]

        return TrajectoryDataset(
            metadata=dataset.metadata,
            agent_snapshots=filtered_snapshots,
            emergence_events=filtered_events,
        )

    @staticmethod
    def action_distribution(
        dataset: TrajectoryDataset, agent_id: str | None = None
    ) -> dict[str, int]:
        """Count actions by type.

        If agent_id is None, counts across all agents.
        Otherwise, counts only for the specified agent.
        """
        snapshots = dataset.agent_snapshots
        if agent_id:
            snapshots = [s for s in snapshots if s.agent_id == agent_id]

        distribution: dict[str, int] = {}
        for snap in snapshots:
            action = snap.action_type
            distribution[action] = distribution.get(action, 0) + 1

        return distribution

    @staticmethod
    def trait_evolution_trace(dataset: TrajectoryDataset, agent_id: str) -> list[dict]:
        """Personality trait values over time.

        Returns list of dicts: [{tick, cooperation_tendency, curiosity, ...}, ...]
        """
        snapshots = TrajectoryLoader.filter_agent(dataset, agent_id)
        return [
            {
                "tick": s.tick,
                **s.traits,
            }
            for s in snapshots
        ]

    @staticmethod
    def _dict_to_metadata(data: dict) -> RunMetadata:
        """Convert dictionary to RunMetadata."""
        return RunMetadata(
            run_id=data["run_id"],
            timestamp=data["timestamp"],
            seed=data["seed"],
            config=data["config"],
            num_agents=data["num_agents"],
            max_ticks=data["max_ticks"],
            actual_ticks=data.get("actual_ticks", 0),
            agents=data.get("agents", []),
            architecture=data.get("architecture"),
            final_state=data.get("final_state", {}),
        )

    @staticmethod
    def _dict_to_snapshot(data: dict) -> AgentSnapshot:
        """Convert dictionary to AgentSnapshot."""
        return AgentSnapshot(
            tick=data["tick"],
            agent_id=data["agent_id"],
            agent_name=data["agent_name"],
            archetype=data["archetype"],
            position=tuple(data["position"]),
            alive=data["alive"],
            hunger=data["hunger"],
            thirst=data["thirst"],
            energy=data["energy"],
            health=data["health"],
            traits=data["traits"],
            sensation_summary=data["sensation_summary"],
            reflection=data["reflection"],
            intention=data["intention"],
            action_type=data["action_type"],
            action_target=data["action_target"],
            action_target_agent=data["action_target_agent"],
            action_succeeded=data["action_succeeded"],
            needs_delta=data["needs_delta"],
            inventory=data["inventory"],
            messages_sent=data.get("messages_sent", []),
            messages_received=data.get("messages_received", []),
            internal_monologue=data.get("internal_monologue", ""),
            trait_changes=data.get("trait_changes", []),
        )

    @staticmethod
    def _dict_to_emergence(data: dict) -> EmergenceSnapshot:
        """Convert dictionary to EmergenceSnapshot."""
        return EmergenceSnapshot(
            tick=data["tick"],
            pattern_type=data["pattern_type"],
            agents_involved=data["agents_involved"],
            description=data["description"],
            data=data.get("data", {}),
        )
