"""Survival analysis and predictors.

Analyze survival statistics and identify predictors of longevity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.trajectory.schema import TrajectoryDataset


class SurvivalAnalyzer:
    """Survival statistics and predictors."""

    def survival_curve(self, dataset: TrajectoryDataset) -> list[tuple[int, float]]:
        """Kaplan-Meier-style survival curve.

        Returns: [(tick, fraction_alive), ...]
        """
        # Get all unique agent IDs
        agent_ids = set(s.agent_id for s in dataset.agent_snapshots)
        total_agents = len(agent_ids)

        if total_agents == 0:
            return []

        # Find death tick for each agent (last tick they appear)
        death_ticks = {}
        for agent_id in agent_ids:
            agent_snapshots = [s for s in dataset.agent_snapshots if s.agent_id == agent_id]
            if agent_snapshots:
                # Sort by tick
                agent_snapshots.sort(key=lambda s: s.tick)
                death_ticks[agent_id] = agent_snapshots[-1].tick

        # Get all ticks
        all_ticks = sorted(set(s.tick for s in dataset.agent_snapshots))

        # Compute fraction alive at each tick
        curve = []
        for tick in all_ticks:
            alive_count = sum(
                1 for agent_id, death_tick in death_ticks.items() if death_tick >= tick
            )
            fraction_alive = alive_count / total_agents
            curve.append((tick, fraction_alive))

        return curve

    def death_cause_distribution(self, dataset: TrajectoryDataset) -> dict[str, int]:
        """Count deaths by cause.

        Infers cause from final state (health=0 vs other needs=0).
        """
        # Get all unique agent IDs
        agent_ids = set(s.agent_id for s in dataset.agent_snapshots)

        causes: dict[str, int] = {}

        for agent_id in agent_ids:
            agent_snapshots = [s for s in dataset.agent_snapshots if s.agent_id == agent_id]
            if not agent_snapshots:
                continue

            # Sort by tick
            agent_snapshots.sort(key=lambda s: s.tick)
            final_snapshot = agent_snapshots[-1]

            # If agent is still alive at end, skip
            if final_snapshot.alive:
                continue

            # Determine cause of death from final snapshot
            cause = self._infer_death_cause(final_snapshot)
            causes[cause] = causes.get(cause, 0) + 1

        return causes

    def survival_predictors(self, datasets: list[TrajectoryDataset]) -> dict:
        """Feature importance for survival prediction.

        Implement basic information gain (no sklearn required).

        Returns: {
            "cooperation_tendency": 0.42,
            "curiosity": 0.31,
            ...
        }
        """
        # Collect all agent data across datasets
        agent_data = []

        for dataset in datasets:
            agent_ids = set(s.agent_id for s in dataset.agent_snapshots)

            # Get trait names
            trait_names = []
            if dataset.agent_snapshots:
                trait_names = list(dataset.agent_snapshots[0].traits.keys())

            for agent_id in agent_ids:
                agent_snapshots = [s for s in dataset.agent_snapshots if s.agent_id == agent_id]
                if not agent_snapshots:
                    continue

                # Average trait values
                avg_traits = {}
                for trait in trait_names:
                    values = [s.traits.get(trait, 0.5) for s in agent_snapshots]
                    avg_traits[trait] = sum(values) / len(values)

                # Survival duration
                survival_ticks = len(agent_snapshots)

                # Classify as long survivor (top 50%) or short
                agent_data.append({"traits": avg_traits, "survival": survival_ticks})

        if not agent_data:
            return {}

        # Find median survival
        survival_values = [ad["survival"] for ad in agent_data]
        survival_values.sort()
        median_survival_obj = survival_values[len(survival_values) // 2]
        assert isinstance(median_survival_obj, (int, float)), "median must be numeric"
        median_survival = float(median_survival_obj)

        # Label agents as long (1) or short (0) survivors
        for ad in agent_data:
            survival = ad["survival"]
            assert isinstance(survival, (int, float)), "survival must be int or float"
            ad["survived_long"] = 1 if float(survival) >= median_survival else 0

        # Compute information gain for each trait
        if not agent_data:
            return {}
        traits_dict = agent_data[0]["traits"]
        assert isinstance(traits_dict, dict), "traits must be dict"
        trait_names = list(traits_dict.keys())
        result = {}

        for trait in trait_names:
            info_gain = self._compute_information_gain(agent_data, trait)
            result[trait] = info_gain

        return result

    def _infer_death_cause(self, snapshot) -> str:
        """Infer cause of death from final snapshot."""
        if snapshot.health <= 0:
            return "health"
        elif snapshot.hunger <= 0:
            return "hunger"
        elif snapshot.thirst <= 0:
            return "thirst"
        elif snapshot.energy <= 0:
            return "energy"
        else:
            return "unknown"

    def _compute_information_gain(self, agent_data: list[dict], trait: str) -> float:
        """Compute information gain for a trait predicting survival.

        Uses binary split at trait median.
        """
        # Get trait values and survival labels
        trait_values = [ad["traits"][trait] for ad in agent_data]
        labels = [ad["survived_long"] for ad in agent_data]

        # Find median trait value
        sorted_traits = sorted(trait_values)
        median_trait = sorted_traits[len(sorted_traits) // 2]

        # Split data by trait median
        low_labels = [labels[i] for i in range(len(labels)) if trait_values[i] < median_trait]
        high_labels = [labels[i] for i in range(len(labels)) if trait_values[i] >= median_trait]

        # Compute entropy before split
        entropy_before = self._entropy(labels)

        # Compute weighted average entropy after split
        total = len(labels)
        if total == 0:
            return 0.0

        low_weight = len(low_labels) / total
        high_weight = len(high_labels) / total

        entropy_after = low_weight * self._entropy(low_labels) + high_weight * self._entropy(
            high_labels
        )

        # Information gain
        return entropy_before - entropy_after

    def _entropy(self, labels: list[int]) -> float:
        """Compute binary entropy."""
        if not labels:
            return 0.0

        # Count positives and negatives
        positive = sum(labels)
        total = len(labels)
        negative = total - positive

        if positive == 0 or negative == 0:
            return 0.0

        p_pos = positive / total
        p_neg = negative / total

        # Entropy = -p*log2(p) - (1-p)*log2(1-p)
        import math

        return -(p_pos * math.log2(p_pos) + p_neg * math.log2(p_neg))
