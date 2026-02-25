"""Trait-behavior correlation analysis.

Compute statistical correlations between personality traits and behavioral outcomes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.trajectory.schema import TrajectoryDataset


class TraitBehaviorAnalyzer:
    """Correlate personality traits with behavioral outcomes."""

    def trait_action_correlation(self, dataset: TrajectoryDataset) -> dict:
        """Compute Pearson correlation between each trait and each action frequency.

        Returns: {"cooperation_tendency": {"give": 0.82, "attack": -0.71, ...}, ...}
        """
        # Get all unique agent IDs
        agent_ids = set(s.agent_id for s in dataset.agent_snapshots)

        # Get all trait names from first snapshot
        trait_names = []
        if dataset.agent_snapshots:
            trait_names = list(dataset.agent_snapshots[0].traits.keys())

        # Get all action types
        action_types = set(s.action_type for s in dataset.agent_snapshots)

        # For each agent, compute average trait values and action frequencies
        agent_data = {}
        for agent_id in agent_ids:
            agent_snapshots = [s for s in dataset.agent_snapshots if s.agent_id == agent_id]

            # Average trait values across the agent's lifetime
            avg_traits = {}
            for trait in trait_names:
                values = [s.traits.get(trait, 0.5) for s in agent_snapshots]
                avg_traits[trait] = sum(values) / len(values) if values else 0.5

            # Action frequencies (proportion of each action)
            action_counts = {}
            for action in action_types:
                count = sum(1 for s in agent_snapshots if s.action_type == action)
                action_counts[action] = count / len(agent_snapshots) if agent_snapshots else 0.0

            agent_data[agent_id] = {"traits": avg_traits, "actions": action_counts}

        # Compute Pearson correlation for each trait-action pair
        result: dict[str, dict[str, float]] = {}
        for trait in trait_names:
            result[trait] = {}
            for action in action_types:
                trait_values = [agent_data[aid]["traits"][trait] for aid in agent_ids]
                action_values = [agent_data[aid]["actions"][action] for aid in agent_ids]

                correlation = self._pearson_correlation(trait_values, action_values)
                result[trait][action] = correlation

        return result

    def trait_survival_correlation(self, dataset: TrajectoryDataset) -> dict:
        """Which traits predict longer survival?

        Returns: {"cooperation_tendency": 0.34, "risk_tolerance": -0.28, ...}
        """
        # Get all unique agent IDs
        agent_ids = set(s.agent_id for s in dataset.agent_snapshots)

        # Get trait names
        trait_names = []
        if dataset.agent_snapshots:
            trait_names = list(dataset.agent_snapshots[0].traits.keys())

        # For each agent, compute average traits and survival duration
        agent_data = []
        for agent_id in agent_ids:
            agent_snapshots = [s for s in dataset.agent_snapshots if s.agent_id == agent_id]
            if not agent_snapshots:
                continue

            # Average trait values
            avg_traits = {}
            for trait in trait_names:
                values = [s.traits.get(trait, 0.5) for s in agent_snapshots]
                avg_traits[trait] = sum(values) / len(values)

            # Survival duration (number of ticks alive)
            survival_ticks = len(agent_snapshots)

            agent_data.append({"traits": avg_traits, "survival": survival_ticks})

        # Compute correlation for each trait
        result = {}
        survival_values = [float(ad["survival"]) for ad in agent_data]  # type: ignore[arg-type]

        for trait in trait_names:
            trait_values_raw = [ad["traits"][trait] for ad in agent_data]  # type: ignore[index]
            trait_values = [float(v) for v in trait_values_raw]  # type: ignore[arg-type]
            correlation = self._pearson_correlation(trait_values, survival_values)
            result[trait] = correlation

        return result

    def optimal_trait_profile(self, datasets: list[TrajectoryDataset]) -> dict:
        """Across many runs, what trait profile maximizes survival?

        Returns: {"cooperation_tendency": 0.72, "curiosity": 0.35, ...}
        """
        # Aggregate all agent data across runs
        all_agent_data = []

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

                all_agent_data.append({"traits": avg_traits, "survival": survival_ticks})

        if not all_agent_data:
            return {}

        # Find the top 25% survivors
        all_agent_data.sort(key=lambda x: float(x["survival"]), reverse=True)  # type: ignore[arg-type]
        top_quartile_count = max(1, len(all_agent_data) // 4)
        top_survivors = all_agent_data[:top_quartile_count]

        # Average their trait profiles
        if not top_survivors:
            return {}
        traits_dict = top_survivors[0]["traits"]
        assert isinstance(traits_dict, dict), "traits must be dict"
        trait_names = list(traits_dict.keys())
        optimal_profile = {}

        for trait in trait_names:
            values_raw = [ad["traits"][trait] for ad in top_survivors]  # type: ignore[index]
            values = [float(v) for v in values_raw]  # type: ignore[arg-type]
            optimal_profile[trait] = sum(values) / len(values) if values else 0.5

        return optimal_profile

    def _pearson_correlation(self, x: list[float], y: list[float]) -> float:
        """Compute Pearson correlation coefficient.

        Formula: r = (n*Σxy - Σx*Σy) / sqrt((n*Σx² - (Σx)²) * (n*Σy² - (Σy)²))
        """
        if len(x) != len(y) or len(x) == 0:
            return 0.0

        n = len(x)

        # Compute sums
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y, strict=False))
        sum_x_sq = sum(xi * xi for xi in x)
        sum_y_sq = sum(yi * yi for yi in y)

        # Compute numerator and denominator
        numerator = n * sum_xy - sum_x * sum_y
        denominator_x = n * sum_x_sq - sum_x * sum_x
        denominator_y = n * sum_y_sq - sum_y * sum_y

        # Avoid division by zero
        if denominator_x <= 0 or denominator_y <= 0:
            return 0.0

        denominator = (denominator_x * denominator_y) ** 0.5

        if denominator == 0:
            return 0.0

        return float(numerator / denominator)
