"""Phenomena computation: derive behavioral patterns from trajectory data.

Tier 2 data: computed from Tier 1 trajectories, stored as first-class entities.
Phenomena are the research-valuable patterns extracted from raw simulation recordings.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.trajectory.schema import TrajectoryDataset


@dataclass
class ValueDriftCurve:
    """Trait evolution over time for one agent × one trait."""

    agent_id: str
    agent_name: str
    trait_name: str
    ticks: list[int]  # tick numbers where trait was sampled
    values: list[float]  # trait values at each tick
    total_drift: float  # abs(final - initial)
    drift_rate: float  # total_drift / num_ticks (0 if no ticks)
    direction: str  # "increasing" | "decreasing" | "stable"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class NormConvergenceMetric:
    """Measures whether agents' trait values converge (norm emergence)."""

    trait_name: str
    initial_variance: float  # variance of trait across agents in first quartile of ticks
    final_variance: float  # variance in last quartile of ticks
    convergence_ratio: float  # final_var / initial_var (< 1 = converging, > 1 = diverging)
    converged: bool  # convergence_ratio < 0.5

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class FailureSignature:
    """Pattern preceding mass agent death."""

    trigger_tick: int  # tick where death rate spiked
    agents_dying: list[str]  # agent_ids that died in the spike window
    preceding_conditions: dict  # avg needs, action distribution in lookback window
    death_rate: float  # deaths per tick in the failure window
    contributing_factors: list[str]  # ["starvation", "dehydration", "combat", etc.]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PhenomenaReport:
    """All derived phenomena from a single run."""

    run_id: str
    value_drift_curves: list[ValueDriftCurve]
    norm_convergence: list[NormConvergenceMetric]
    failure_signatures: list[FailureSignature]

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "run_id": self.run_id,
            "value_drift_curves": [vdc.to_dict() for vdc in self.value_drift_curves],
            "norm_convergence": [nc.to_dict() for nc in self.norm_convergence],
            "failure_signatures": [fs.to_dict() for fs in self.failure_signatures],
        }

    def save(self, run_dir: str) -> None:
        """Save to {run_dir}/phenomena.json."""
        import os

        output_path = os.path.join(run_dir, "phenomena.json")
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load(run_dir: str) -> PhenomenaReport:
        """Load from {run_dir}/phenomena.json."""
        import os

        input_path = os.path.join(run_dir, "phenomena.json")
        with open(input_path) as f:
            data = json.load(f)

        # Reconstruct dataclasses
        value_drift_curves = [ValueDriftCurve(**vdc) for vdc in data["value_drift_curves"]]
        norm_convergence = [NormConvergenceMetric(**nc) for nc in data["norm_convergence"]]
        failure_signatures = [FailureSignature(**fs) for fs in data["failure_signatures"]]

        return PhenomenaReport(
            run_id=data["run_id"],
            value_drift_curves=value_drift_curves,
            norm_convergence=norm_convergence,
            failure_signatures=failure_signatures,
        )


class PhenomenaComputer:
    """Compute derived phenomena from trajectory data."""

    TRAIT_NAMES = [
        "cooperation_tendency",
        "curiosity",
        "risk_tolerance",
        "resource_sharing",
        "aggression",
        "sociability",
    ]

    def compute_all(self, dataset: TrajectoryDataset) -> PhenomenaReport:
        """Run all phenomena computations.

        Args:
            dataset: Trajectory dataset from a single simulation run

        Returns:
            Complete phenomena report with all derived patterns
        """
        return PhenomenaReport(
            run_id=dataset.metadata.run_id,
            value_drift_curves=self.compute_value_drift(dataset),
            norm_convergence=self.compute_norm_convergence(dataset),
            failure_signatures=self.compute_failure_signatures(dataset),
        )

    def compute_value_drift(self, dataset: TrajectoryDataset) -> list[ValueDriftCurve]:
        """Compute trait evolution curves per agent × trait.

        For each living agent, for each of the 6 traits:
        1. Collect trait values over time (from snapshots)
        2. Compute total drift = abs(final - initial)
        3. Compute drift rate = total_drift / num_ticks
        4. Determine direction: if final > initial + 0.02 → "increasing",
           if final < initial - 0.02 → "decreasing", else "stable"

        Args:
            dataset: Trajectory dataset

        Returns:
            List of ValueDriftCurve objects, one per agent × trait combination
        """
        from src.trajectory.schema import AgentSnapshot

        # Group snapshots by agent_id
        agent_snapshots_map: dict[str, list[AgentSnapshot]] = {}
        for snapshot in dataset.agent_snapshots:
            if snapshot.agent_id not in agent_snapshots_map:
                agent_snapshots_map[snapshot.agent_id] = []
            agent_snapshots_map[snapshot.agent_id].append(snapshot)

        # Sort each agent's snapshots by tick
        for agent_id in agent_snapshots_map:
            agent_snapshots_map[agent_id].sort(key=lambda s: s.tick)

        curves = []

        # For each agent, for each trait
        for agent_id, snapshots in agent_snapshots_map.items():
            if not snapshots:
                continue

            agent_name = snapshots[0].agent_name

            for trait_name in self.TRAIT_NAMES:
                # Extract time series for this trait
                ticks = []
                values = []

                for snapshot in snapshots:
                    trait_value = snapshot.traits.get(trait_name, 0.5)
                    ticks.append(snapshot.tick)
                    values.append(trait_value)

                if not values:
                    continue

                # Compute drift metrics
                initial_value = values[0]
                final_value = values[-1]
                total_drift = abs(final_value - initial_value)
                drift_rate = total_drift / len(ticks) if len(ticks) > 0 else 0.0

                # Determine direction
                if final_value > initial_value + 0.02:
                    direction = "increasing"
                elif final_value < initial_value - 0.02:
                    direction = "decreasing"
                else:
                    direction = "stable"

                curves.append(
                    ValueDriftCurve(
                        agent_id=agent_id,
                        agent_name=agent_name,
                        trait_name=trait_name,
                        ticks=ticks,
                        values=values,
                        total_drift=total_drift,
                        drift_rate=drift_rate,
                        direction=direction,
                    )
                )

        return curves

    def compute_norm_convergence(self, dataset: TrajectoryDataset) -> list[NormConvergenceMetric]:
        """Measure whether agent traits converge (norm emergence).

        For each trait:
        1. Get all agent trait values in first 25% of ticks → compute variance
        2. Get all agent trait values in last 25% of ticks → compute variance
        3. convergence_ratio = final_variance / initial_variance
        4. converged = convergence_ratio < 0.5

        Handle edge case: if initial_variance is 0 (all agents identical),
        set convergence_ratio to 1.0 (no change possible).

        Args:
            dataset: Trajectory dataset

        Returns:
            List of NormConvergenceMetric objects, one per trait
        """
        if not dataset.agent_snapshots:
            return []

        # Get tick range
        all_ticks = [s.tick for s in dataset.agent_snapshots]
        min_tick = min(all_ticks)
        max_tick = max(all_ticks)
        tick_range = max_tick - min_tick

        # Define first and last quartile boundaries
        first_quartile_end = min_tick + tick_range * 0.25
        last_quartile_start = max_tick - tick_range * 0.25

        metrics = []

        for trait_name in self.TRAIT_NAMES:
            # Collect trait values in first quartile
            first_quartile_values = []
            for snapshot in dataset.agent_snapshots:
                if snapshot.tick <= first_quartile_end:
                    trait_value = snapshot.traits.get(trait_name, 0.5)
                    first_quartile_values.append(trait_value)

            # Collect trait values in last quartile
            last_quartile_values = []
            for snapshot in dataset.agent_snapshots:
                if snapshot.tick >= last_quartile_start:
                    trait_value = snapshot.traits.get(trait_name, 0.5)
                    last_quartile_values.append(trait_value)

            if not first_quartile_values or not last_quartile_values:
                continue

            # Compute variance for each quartile
            initial_variance = self._variance(first_quartile_values)
            final_variance = self._variance(last_quartile_values)

            # Handle edge case: if initial variance is 0, no convergence possible
            if initial_variance == 0.0:
                convergence_ratio = 1.0
            else:
                convergence_ratio = final_variance / initial_variance

            converged = convergence_ratio < 0.5

            metrics.append(
                NormConvergenceMetric(
                    trait_name=trait_name,
                    initial_variance=initial_variance,
                    final_variance=final_variance,
                    convergence_ratio=convergence_ratio,
                    converged=converged,
                )
            )

        return metrics

    def compute_failure_signatures(self, dataset: TrajectoryDataset) -> list[FailureSignature]:
        """Detect patterns preceding mass agent death.

        Algorithm:
        1. Build alive-count per tick from snapshots
        2. Compute death events: ticks where alive_count drops by >=2 agents within 5 ticks
        3. For each death spike:
           a. Look back 10 ticks before the spike
           b. Compute avg needs (hunger, thirst, energy, health) in lookback window
           c. Compute action distribution in lookback window
           d. Determine contributing factors:
              - "starvation" if avg hunger < 15
              - "dehydration" if avg thirst < 15
              - "exhaustion" if avg energy < 15
              - "combat" if attack actions > 20% in lookback
              - "general_decline" if avg health < 30
        4. Return FailureSignature for each detected spike

        If no mass death events, return empty list.

        Args:
            dataset: Trajectory dataset

        Returns:
            List of FailureSignature objects for detected mass death events
        """
        if not dataset.agent_snapshots:
            return []

        # Build alive count per tick
        tick_alive_count: dict[int, set[str]] = {}
        tick_alive_agents: dict[int, set[str]] = {}

        for snapshot in dataset.agent_snapshots:
            tick = snapshot.tick
            agent_id = snapshot.agent_id

            if tick not in tick_alive_count:
                tick_alive_count[tick] = set()
                tick_alive_agents[tick] = set()

            if snapshot.alive:
                tick_alive_count[tick].add(agent_id)
                tick_alive_agents[tick].add(agent_id)

        # Convert to sorted list of (tick, alive_count)
        sorted_ticks = sorted(tick_alive_count.keys())
        alive_counts = [(tick, len(tick_alive_count[tick])) for tick in sorted_ticks]

        # Detect death spikes: drops of >=2 agents within 5 ticks
        death_spikes = []

        for i in range(len(alive_counts)):
            current_tick, current_alive = alive_counts[i]

            # Look ahead 5 ticks
            for j in range(i + 1, min(i + 6, len(alive_counts))):
                future_tick, future_alive = alive_counts[j]

                # Check if alive count dropped by >=2
                if current_alive - future_alive >= 2:
                    # Find which agents died in this window
                    agents_at_current = tick_alive_agents.get(current_tick, set())
                    agents_at_future = tick_alive_agents.get(future_tick, set())
                    agents_dying = list(agents_at_current - agents_at_future)

                    death_rate = (current_alive - future_alive) / (future_tick - current_tick)

                    death_spikes.append(
                        {
                            "trigger_tick": future_tick,
                            "agents_dying": agents_dying,
                            "death_rate": death_rate,
                        }
                    )
                    break  # Only count first significant drop from this point

        # For each death spike, analyze preceding conditions
        signatures = []

        for spike in death_spikes:
            spike_trigger_tick: int = spike["trigger_tick"]  # type: ignore[assignment]
            spike_agents_dying: list[str] = spike["agents_dying"]  # type: ignore[assignment]
            spike_death_rate: float = spike["death_rate"]  # type: ignore[assignment]

            # Look back 10 ticks before the spike
            lookback_start = spike_trigger_tick - 10

            # Get all snapshots in lookback window
            lookback_snapshots = [
                s for s in dataset.agent_snapshots if lookback_start <= s.tick < spike_trigger_tick
            ]

            if not lookback_snapshots:
                continue

            # Compute average needs
            avg_hunger = sum(s.hunger for s in lookback_snapshots) / len(lookback_snapshots)
            avg_thirst = sum(s.thirst for s in lookback_snapshots) / len(lookback_snapshots)
            avg_energy = sum(s.energy for s in lookback_snapshots) / len(lookback_snapshots)
            avg_health = sum(s.health for s in lookback_snapshots) / len(lookback_snapshots)

            # Compute action distribution
            action_counts: dict[str, int] = {}
            for snapshot in lookback_snapshots:
                action = snapshot.action_type
                action_counts[action] = action_counts.get(action, 0) + 1

            total_actions = len(lookback_snapshots)
            action_distribution = {a: c / total_actions for a, c in action_counts.items()}

            # Determine contributing factors
            contributing_factors = []

            if avg_hunger < 15:
                contributing_factors.append("starvation")
            if avg_thirst < 15:
                contributing_factors.append("dehydration")
            if avg_energy < 15:
                contributing_factors.append("exhaustion")
            if avg_health < 30:
                contributing_factors.append("general_decline")

            # Check for combat (attack actions > 20%)
            attack_proportion = action_distribution.get("attack", 0.0)
            if attack_proportion > 0.2:
                contributing_factors.append("combat")

            # Build preceding conditions dict
            preceding_conditions = {
                "avg_hunger": avg_hunger,
                "avg_thirst": avg_thirst,
                "avg_energy": avg_energy,
                "avg_health": avg_health,
                "action_distribution": action_distribution,
            }

            signatures.append(
                FailureSignature(
                    trigger_tick=spike_trigger_tick,
                    agents_dying=spike_agents_dying,
                    preceding_conditions=preceding_conditions,
                    death_rate=spike_death_rate,
                    contributing_factors=contributing_factors,
                )
            )

        return signatures

    def _variance(self, values: list[float]) -> float:
        """Compute variance manually: sum((x - mean)^2) / n.

        Args:
            values: List of numeric values

        Returns:
            Variance of the values
        """
        if not values:
            return 0.0

        n = len(values)
        mean = sum(values) / n

        squared_diffs = [(x - mean) ** 2 for x in values]
        variance = sum(squared_diffs) / n

        return variance
