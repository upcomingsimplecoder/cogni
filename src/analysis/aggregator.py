"""Cross-run data aggregation.

Combine multiple trajectory runs into unified analysis corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.trajectory.schema import AgentSnapshot, EmergenceSnapshot, TrajectoryDataset


@dataclass
class AggregateDataset:
    """Combined dataset from multiple runs."""

    run_ids: list[str] = field(default_factory=list)
    agent_snapshots: list[tuple[str, AgentSnapshot]] = field(
        default_factory=list
    )  # (run_id, snapshot)
    emergence_events: list[tuple[str, EmergenceSnapshot]] = field(
        default_factory=list
    )  # (run_id, event)
    metadata_by_run: dict[str, dict] = field(default_factory=dict)


class DatasetAggregator:
    """Combine multiple trajectory runs into unified analysis corpus."""

    def combine(self, datasets: list[TrajectoryDataset]) -> AggregateDataset:
        """Merge multiple runs, tagging each snapshot with run_id."""
        agg = AggregateDataset()

        for dataset in datasets:
            run_id = dataset.metadata.run_id
            agg.run_ids.append(run_id)

            # Tag snapshots with run_id
            for snapshot in dataset.agent_snapshots:
                agg.agent_snapshots.append((run_id, snapshot))

            # Tag emergence events with run_id
            for event in dataset.emergence_events:
                agg.emergence_events.append((run_id, event))

            # Store metadata
            agg.metadata_by_run[run_id] = {
                "timestamp": dataset.metadata.timestamp,
                "seed": dataset.metadata.seed,
                "num_agents": dataset.metadata.num_agents,
                "actual_ticks": dataset.metadata.actual_ticks,
                "architecture": dataset.metadata.architecture,
                "final_state": dataset.metadata.final_state,
            }

        return agg

    def personality_behavior_matrix(self, agg: AggregateDataset) -> dict:
        """Pivot table: trait buckets (low/med/high) × action frequencies.

        Returns: {
            "cooperation_tendency": {
                "low": {"move": 0.15, "gather": 0.50, ...},
                "med": {"move": 0.12, "gather": 0.45, ...},
                "high": {"move": 0.10, "gather": 0.40, ...}
            },
            ...
        }
        """
        # Get all unique agents
        agent_data: dict[str, dict[str, list]] = {}  # agent_id -> {traits, actions}

        for run_id, snapshot in agg.agent_snapshots:
            agent_id = f"{run_id}_{snapshot.agent_id}"

            if agent_id not in agent_data:
                agent_data[agent_id] = {"trait_samples": [], "actions": []}

            agent_data[agent_id]["trait_samples"].append(snapshot.traits)
            agent_data[agent_id]["actions"].append(snapshot.action_type)

        # Compute average traits and action distribution for each agent
        agent_profiles = []
        for _agent_id, data in agent_data.items():
            # Average traits
            trait_names = list(data["trait_samples"][0].keys()) if data["trait_samples"] else []
            avg_traits = {}
            for trait in trait_names:
                values = [t.get(trait, 0.5) for t in data["trait_samples"]]
                avg_traits[trait] = sum(values) / len(values) if values else 0.5

            # Action distribution
            action_counts: dict[str, int] = {}
            total_actions = len(data["actions"])
            for action in data["actions"]:
                action_counts[action] = action_counts.get(action, 0) + 1

            action_dist = {a: c / total_actions for a, c in action_counts.items()}

            agent_profiles.append({"traits": avg_traits, "action_dist": action_dist})

        # Build pivot table for each trait
        result: dict[str, dict[str, dict[str, float]]] = {}
        trait_names = list(agent_profiles[0]["traits"].keys()) if agent_profiles else []

        for trait in trait_names:
            # Bucket agents by trait value (low/med/high)
            buckets: dict[str, list[dict]] = {"low": [], "med": [], "high": []}

            for profile in agent_profiles:
                trait_value = profile["traits"][trait]
                if trait_value < 0.33:
                    buckets["low"].append(profile)
                elif trait_value < 0.67:
                    buckets["med"].append(profile)
                else:
                    buckets["high"].append(profile)

            # Compute average action distribution for each bucket
            result[trait] = {}
            for bucket_name, profiles in buckets.items():
                if not profiles:
                    result[trait][bucket_name] = {}
                    continue

                # Aggregate action distributions
                action_sums: dict[str, float] = {}
                for profile in profiles:
                    for action, freq in profile["action_dist"].items():
                        action_sums[action] = action_sums.get(action, 0.0) + freq

                # Average
                action_avg = {a: s / len(profiles) for a, s in action_sums.items()}
                result[trait][bucket_name] = action_avg

        return result

    def architecture_comparison(self, agg: AggregateDataset) -> dict:
        """Compare metrics across cognitive architectures.

        Returns: {
            "SRIE": {
                "avg_survival_ticks": 145.2,
                "avg_agents_alive": 3.5,
                "total_emergence_events": 12
            },
            ...
        }
        """
        # Group runs by architecture
        by_architecture: dict[str, dict[str, list]] = {}

        for run_id in agg.run_ids:
            metadata = agg.metadata_by_run[run_id]
            arch = metadata.get("architecture", "unknown")

            if arch not in by_architecture:
                by_architecture[arch] = {
                    "run_ids": [],
                    "survival_ticks": [],
                    "agents_alive": [],
                    "emergence_events": [],
                }

            by_architecture[arch]["run_ids"].append(run_id)
            by_architecture[arch]["survival_ticks"].append(metadata["actual_ticks"])
            by_architecture[arch]["agents_alive"].append(
                metadata.get("final_state", {}).get("agents_alive", 0)
            )

            # Count emergence events for this run
            event_count = sum(1 for rid, _ in agg.emergence_events if rid == run_id)
            by_architecture[arch]["emergence_events"].append(event_count)

        # Compute statistics for each architecture
        result = {}
        for arch, data in by_architecture.items():
            result[arch] = {
                "num_runs": len(data["run_ids"]),
                "avg_survival_ticks": sum(data["survival_ticks"]) / len(data["survival_ticks"])
                if data["survival_ticks"]
                else 0.0,
                "avg_agents_alive": sum(data["agents_alive"]) / len(data["agents_alive"])
                if data["agents_alive"]
                else 0.0,
                "total_emergence_events": sum(data["emergence_events"]),
                "avg_emergence_events": sum(data["emergence_events"])
                / len(data["emergence_events"])
                if data["emergence_events"]
                else 0.0,
            }

        return result
