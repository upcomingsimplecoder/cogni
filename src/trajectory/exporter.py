"""Trajectory data exporters.

Export trajectory data to analysis-friendly formats:
- JSONL (primary format, streaming)
- CSV (flat format for quick analysis)
- Agent summaries (aggregated per-agent stats)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.trajectory.schema import AgentSnapshot, TrajectoryDataset


class TrajectoryExporter:
    """Export trajectory data to analysis-friendly formats."""

    @staticmethod
    def to_jsonl(dataset: TrajectoryDataset, path: str) -> None:
        """Export to JSONL format (one JSON object per line).

        This is the primary format written incrementally by TrajectoryRecorder.
        This method re-exports a loaded dataset.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            # Write metadata
            f.write(json.dumps({"type": "metadata", **dataset.metadata.to_dict()}) + "\n")

            # Write agent snapshots
            for snapshot in dataset.agent_snapshots:
                f.write(json.dumps({"type": "agent_snapshot", **snapshot.to_dict()}) + "\n")

            # Write emergence events
            for event in dataset.emergence_events:
                f.write(json.dumps({"type": "emergence", **event.to_dict()}) + "\n")

            # Write completion marker
            f.write(
                json.dumps(
                    {
                        "type": "run_complete",
                        "actual_ticks": dataset.metadata.actual_ticks,
                        "final_state": dataset.metadata.final_state,
                    }
                )
                + "\n"
            )

    @staticmethod
    def to_csv(dataset: TrajectoryDataset, path: str) -> None:
        """Flat CSV with one row per agent per tick.

        Columns: tick, agent_id, agent_name, archetype, x, y, alive,
                 hunger, thirst, energy, health,
                 action_type, action_succeeded,
                 primary_goal, threat_level, opportunity_score,
                 cooperation_tendency, curiosity, risk_tolerance,
                 resource_sharing, aggression, sociability
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not dataset.agent_snapshots:
            # Write empty CSV with headers
            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(TrajectoryExporter._get_csv_headers())
            return

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(TrajectoryExporter._get_csv_headers())

            for snapshot in dataset.agent_snapshots:
                writer.writerow(TrajectoryExporter._snapshot_to_csv_row(snapshot))

    @staticmethod
    def to_agent_summary(dataset: TrajectoryDataset, path: str) -> None:
        """One row per agent: initial_traits, final_traits, trait_deltas,
        action_distribution, survival_ticks, cooperation_rate, etc.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Group snapshots by agent
        agent_trajectories: dict[str, list[AgentSnapshot]] = {}
        for snapshot in dataset.agent_snapshots:
            if snapshot.agent_id not in agent_trajectories:
                agent_trajectories[snapshot.agent_id] = []
            agent_trajectories[snapshot.agent_id].append(snapshot)

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "agent_id",
                    "agent_name",
                    "archetype",
                    "survival_ticks",
                    "died",
                    # Initial traits
                    "initial_cooperation",
                    "initial_curiosity",
                    "initial_risk_tolerance",
                    "initial_resource_sharing",
                    "initial_aggression",
                    "initial_sociability",
                    # Final traits
                    "final_cooperation",
                    "final_curiosity",
                    "final_risk_tolerance",
                    "final_resource_sharing",
                    "final_aggression",
                    "final_sociability",
                    # Trait deltas
                    "delta_cooperation",
                    "delta_curiosity",
                    "delta_risk_tolerance",
                    "delta_resource_sharing",
                    "delta_aggression",
                    "delta_sociability",
                    # Action distribution
                    "gather_count",
                    "move_count",
                    "give_count",
                    "attack_count",
                    "wait_count",
                    "other_count",
                    # Success rates
                    "gather_success_rate",
                    "give_success_rate",
                    "attack_success_rate",
                    # Social stats
                    "messages_sent_count",
                    "messages_received_count",
                    # Needs stats
                    "avg_hunger",
                    "avg_thirst",
                    "avg_energy",
                    "avg_health",
                    "final_hunger",
                    "final_thirst",
                    "final_energy",
                    "final_health",
                ]
            )

            for agent_id, snapshots in agent_trajectories.items():
                row = TrajectoryExporter._compute_agent_summary(agent_id, snapshots)
                writer.writerow(row)

    @staticmethod
    def _get_csv_headers() -> list[str]:
        """Get CSV column headers."""
        return [
            "tick",
            "agent_id",
            "agent_name",
            "archetype",
            "x",
            "y",
            "alive",
            "hunger",
            "thirst",
            "energy",
            "health",
            "action_type",
            "action_succeeded",
            "primary_goal",
            "threat_level",
            "opportunity_score",
            "cooperation_tendency",
            "curiosity",
            "risk_tolerance",
            "resource_sharing",
            "aggression",
            "sociability",
        ]

    @staticmethod
    def _snapshot_to_csv_row(snapshot: AgentSnapshot) -> list:
        """Convert agent snapshot to CSV row."""
        return [
            snapshot.tick,
            snapshot.agent_id,
            snapshot.agent_name,
            snapshot.archetype,
            snapshot.position[0],
            snapshot.position[1],
            snapshot.alive,
            snapshot.hunger,
            snapshot.thirst,
            snapshot.energy,
            snapshot.health,
            snapshot.action_type,
            snapshot.action_succeeded,
            snapshot.intention.get("primary_goal", ""),
            snapshot.reflection.get("threat_level", 0.0),
            snapshot.reflection.get("opportunity_score", 0.0),
            snapshot.traits.get("cooperation_tendency", 0.5),
            snapshot.traits.get("curiosity", 0.5),
            snapshot.traits.get("risk_tolerance", 0.5),
            snapshot.traits.get("resource_sharing", 0.5),
            snapshot.traits.get("aggression", 0.5),
            snapshot.traits.get("sociability", 0.5),
        ]

    @staticmethod
    def _compute_agent_summary(agent_id: str, snapshots: list[AgentSnapshot]) -> list:
        """Compute aggregated stats for one agent."""
        if not snapshots:
            return [agent_id] + [0] * 40

        first = snapshots[0]
        last = snapshots[-1]

        # Action distribution
        action_counts = {"gather": 0, "move": 0, "give": 0, "attack": 0, "wait": 0, "other": 0}
        action_successes: dict[str, list[bool]] = {"gather": [], "give": [], "attack": []}

        for snap in snapshots:
            action = snap.action_type.lower()
            if action in action_counts:
                action_counts[action] += 1
            else:
                action_counts["other"] += 1

            if action in action_successes:
                action_successes[action].append(snap.action_succeeded)

        # Success rates
        gather_success = (
            sum(action_successes["gather"]) / len(action_successes["gather"])
            if action_successes["gather"]
            else 0.0
        )
        give_success = (
            sum(action_successes["give"]) / len(action_successes["give"])
            if action_successes["give"]
            else 0.0
        )
        attack_success = (
            sum(action_successes["attack"]) / len(action_successes["attack"])
            if action_successes["attack"]
            else 0.0
        )

        # Message stats
        total_sent = sum(len(snap.messages_sent) for snap in snapshots)
        total_received = sum(len(snap.messages_received) for snap in snapshots)

        # Needs averages
        avg_hunger = sum(snap.hunger for snap in snapshots) / len(snapshots)
        avg_thirst = sum(snap.thirst for snap in snapshots) / len(snapshots)
        avg_energy = sum(snap.energy for snap in snapshots) / len(snapshots)
        avg_health = sum(snap.health for snap in snapshots) / len(snapshots)

        # Trait deltas
        initial_traits = first.traits
        final_traits = last.traits

        return [
            agent_id,
            first.agent_name,
            first.archetype,
            len(snapshots),
            not last.alive,
            # Initial traits
            initial_traits.get("cooperation_tendency", 0.5),
            initial_traits.get("curiosity", 0.5),
            initial_traits.get("risk_tolerance", 0.5),
            initial_traits.get("resource_sharing", 0.5),
            initial_traits.get("aggression", 0.5),
            initial_traits.get("sociability", 0.5),
            # Final traits
            final_traits.get("cooperation_tendency", 0.5),
            final_traits.get("curiosity", 0.5),
            final_traits.get("risk_tolerance", 0.5),
            final_traits.get("resource_sharing", 0.5),
            final_traits.get("aggression", 0.5),
            final_traits.get("sociability", 0.5),
            # Deltas
            final_traits.get("cooperation_tendency", 0.5)
            - initial_traits.get("cooperation_tendency", 0.5),
            final_traits.get("curiosity", 0.5) - initial_traits.get("curiosity", 0.5),
            final_traits.get("risk_tolerance", 0.5) - initial_traits.get("risk_tolerance", 0.5),
            final_traits.get("resource_sharing", 0.5) - initial_traits.get("resource_sharing", 0.5),
            final_traits.get("aggression", 0.5) - initial_traits.get("aggression", 0.5),
            final_traits.get("sociability", 0.5) - initial_traits.get("sociability", 0.5),
            # Action counts
            action_counts["gather"],
            action_counts["move"],
            action_counts["give"],
            action_counts["attack"],
            action_counts["wait"],
            action_counts["other"],
            # Success rates
            gather_success,
            give_success,
            attack_success,
            # Social
            total_sent,
            total_received,
            # Needs
            avg_hunger,
            avg_thirst,
            avg_energy,
            avg_health,
            last.hunger,
            last.thirst,
            last.energy,
            last.health,
        ]
