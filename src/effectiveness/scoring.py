"""Unified effectiveness scoring engine.

Tracks and scores:
1. Nudge effectiveness: did trait shifts improve agent fitness?
2. Router quality: which SRIE architectures produce good outcomes?
3. Classifier accuracy: how accurate are threat/opportunity predictions?

Runs periodic scoring on a configurable interval (cron-like).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.agents.evolution import TraitNudgeRecord


@dataclass
class NudgeEffectivenessScore:
    """Score for a trait nudge's impact on agent fitness."""

    nudge_tick: int
    agent_id: str
    event_type: str
    fitness_before: float
    fitness_after: float
    delta: float
    score: float  # normalized -1 to +1


@dataclass
class RouterQualityRecord:
    """Quality record for an SRIE pipeline decision."""

    tick: int
    agent_id: str
    architecture: str
    intention_goal: str
    action_succeeded: bool
    needs_delta_sum: float  # net change in needs (positive = good)


@dataclass
class ClassifierAccuracyRecord:
    """Record comparing predicted threat/opportunity to actual outcomes."""

    tick: int
    agent_id: str
    predicted_threat: float
    predicted_opportunity: float
    actual_damage_taken: float  # proxy for realized threat (health loss)
    actual_resources_gained: float  # proxy for realized opportunity
    threat_error: float
    opportunity_error: float


class EffectivenessEngine:
    """Periodic scoring engine for nudges, router quality, and classifier accuracy.

    Called every tick but only runs scoring on interval (fast-path no-op otherwise).
    """

    def __init__(
        self,
        scoring_interval: int = 50,
        lookback_window: int = 25,
    ):
        self._scoring_interval = scoring_interval
        self._lookback_window = lookback_window

        # Nudge effectiveness
        self._nudge_scores: list[NudgeEffectivenessScore] = []
        self._pending_nudges: dict[str, list] = {}  # agent_id -> unscored TraitNudgeRecords
        # agent_id -> [(tick, fitness)]
        self._fitness_snapshots: dict[str, list[tuple[int, float]]] = {}

        # Router quality
        self._router_records: deque[RouterQualityRecord] = deque(maxlen=1000)
        self._quality_scores: dict[str, float] = {}  # architecture -> rolling quality

        # Classifier accuracy
        self._classifier_records: deque[ClassifierAccuracyRecord] = deque(maxlen=1000)
        self._classifier_accuracy: dict[str, float] = {}  # agent_id -> rolling accuracy
        # agent_id -> (tick, predicted_threat, predicted_opportunity)
        self._pending_predictions: dict[str, tuple[int, float, float]] = {}
        self._pre_tick_health: dict[str, float] = {}  # agent_id -> health before tick
        self._pre_tick_inventory_value: dict[str, float] = {}  # agent_id -> inv value before tick

    def tick(self, engine: Any, current_tick: int) -> None:
        """Called every tick. Runs scoring crons on interval."""
        # Snapshot fitness for all living agents (needed for nudge scoring)
        self._snapshot_fitness(engine, current_tick)

        # Only run scoring on interval
        if current_tick > 0 and current_tick % self._scoring_interval == 0:
            self.score_nudges(engine, current_tick)
            self.score_router_quality(current_tick)
            self.score_classifier_accuracy(current_tick)

    # --- Nudge Effectiveness (T2B.1) ---

    def record_nudge(self, record: TraitNudgeRecord) -> None:
        """Queue a nudge for future effectiveness scoring."""
        agent_id = record.agent_id
        if agent_id not in self._pending_nudges:
            self._pending_nudges[agent_id] = []
        self._pending_nudges[agent_id].append(record)

    def _snapshot_fitness(self, engine: Any, tick: int) -> None:
        """Snapshot current fitness for all living agents."""
        if not hasattr(engine, "population_manager") or not engine.population_manager:
            # Use a simple fitness proxy: avg of needs
            for agent in engine.registry.living_agents():
                agent_id = str(agent.agent_id)
                fitness = (
                    agent.needs.hunger
                    + agent.needs.thirst
                    + agent.needs.energy
                    + agent.needs.health
                ) / 400.0  # normalized 0-1
                if agent_id not in self._fitness_snapshots:
                    self._fitness_snapshots[agent_id] = []
                snapshots = self._fitness_snapshots[agent_id]
                snapshots.append((tick, fitness))
                # Keep bounded
                if len(snapshots) > 200:
                    self._fitness_snapshots[agent_id] = snapshots[-100:]
        else:
            genetics = engine.population_manager.genetics
            for agent in engine.registry.living_agents():
                agent_id = str(agent.agent_id)
                fitness = genetics.fitness(agent)
                if agent_id not in self._fitness_snapshots:
                    self._fitness_snapshots[agent_id] = []
                snapshots = self._fitness_snapshots[agent_id]
                snapshots.append((tick, fitness))
                if len(snapshots) > 200:
                    self._fitness_snapshots[agent_id] = snapshots[-100:]

    def _get_fitness_at(self, agent_id: str, tick: int) -> float | None:
        """Get fitness snapshot closest to a given tick."""
        snapshots = self._fitness_snapshots.get(agent_id, [])
        if not snapshots:
            return None
        # Find closest tick
        best = min(snapshots, key=lambda s: abs(s[0] - tick))
        if abs(best[0] - tick) > self._lookback_window:
            return None  # Too far away
        return best[1]

    def score_nudges(self, engine: Any, current_tick: int) -> list[NudgeEffectivenessScore]:
        """Score pending nudges that are old enough to measure outcomes."""
        scores = []

        for agent_id, nudges in list(self._pending_nudges.items()):
            remaining = []
            for nudge in nudges:
                # Only score if enough time has passed
                if current_tick - nudge.tick < self._lookback_window:
                    remaining.append(nudge)
                    continue

                # Get fitness before and after
                fitness_before = self._get_fitness_at(agent_id, nudge.tick)
                fitness_after = self._get_fitness_at(agent_id, current_tick)

                if fitness_before is None or fitness_after is None:
                    remaining.append(nudge)
                    continue

                delta = fitness_after - fitness_before
                # Normalize to -1..+1 range
                score = max(-1.0, min(1.0, delta * 5.0))

                effectiveness = NudgeEffectivenessScore(
                    nudge_tick=nudge.tick,
                    agent_id=agent_id,
                    event_type=nudge.event_type,
                    fitness_before=fitness_before,
                    fitness_after=fitness_after,
                    delta=delta,
                    score=score,
                )
                scores.append(effectiveness)
                self._nudge_scores.append(effectiveness)

                # Mark nudge as viewed
                nudge.viewed_at = current_tick

            if remaining:
                self._pending_nudges[agent_id] = remaining
            else:
                self._pending_nudges.pop(agent_id, None)

        return scores

    def get_nudge_effectiveness(self, event_type: str | None = None) -> dict:
        """Get nudge effectiveness summary. Optionally filter by event_type."""
        relevant = self._nudge_scores
        if event_type:
            relevant = [s for s in relevant if s.event_type == event_type]

        if not relevant:
            return {"count": 0, "avg_score": 0.0, "avg_delta": 0.0}

        return {
            "count": len(relevant),
            "avg_score": sum(s.score for s in relevant) / len(relevant),
            "avg_delta": sum(s.delta for s in relevant) / len(relevant),
        }

    # --- Router Quality (T2B.2) ---

    def record_routing_outcome(
        self,
        tick: int,
        agent_id: str,
        architecture: str,
        intention_goal: str,
        action_succeeded: bool,
        needs_delta_sum: float,
    ) -> None:
        """Record outcome of an SRIE routing decision."""
        self._router_records.append(
            RouterQualityRecord(
                tick=tick,
                agent_id=agent_id,
                architecture=architecture,
                intention_goal=intention_goal,
                action_succeeded=action_succeeded,
                needs_delta_sum=needs_delta_sum,
            )
        )

    def score_router_quality(self, current_tick: int) -> dict[str, float]:
        """Compute rolling quality score per architecture."""
        # Filter to recent records
        cutoff = current_tick - self._scoring_interval
        recent = [r for r in self._router_records if r.tick >= cutoff]

        # Group by architecture
        by_arch: dict[str, list[RouterQualityRecord]] = {}
        for r in recent:
            if r.architecture not in by_arch:
                by_arch[r.architecture] = []
            by_arch[r.architecture].append(r)

        # Compute quality per architecture
        for arch, records in by_arch.items():
            success_rate = sum(1 for r in records if r.action_succeeded) / len(records)
            avg_needs_delta = sum(r.needs_delta_sum for r in records) / len(records)
            # Quality = weighted combination of success rate and needs improvement
            quality = 0.6 * success_rate + 0.4 * max(0.0, min(1.0, (avg_needs_delta + 10) / 20))
            self._quality_scores[arch] = quality

        return dict(self._quality_scores)

    def get_architecture_weights(self) -> dict[str, float]:
        """Get architecture weights for score-weighted routing."""
        if not self._quality_scores:
            return {}  # Empty = equal weights (cold start)

        # Normalize scores to weights
        total = sum(self._quality_scores.values())
        if total == 0:
            return {k: 1.0 for k in self._quality_scores}

        return {k: v / total for k, v in self._quality_scores.items()}

    # --- Classifier Accuracy (T2B.3 + T2B.4) ---

    def record_classification(
        self,
        tick: int,
        agent_id: str,
        predicted_threat: float,
        predicted_opportunity: float,
    ) -> None:
        """Record a classifier's prediction for later accuracy scoring."""
        self._pending_predictions[agent_id] = (tick, predicted_threat, predicted_opportunity)

    def record_actual_outcome(
        self,
        agent_id: str,
        health_before: float,
        health_after: float,
        inventory_value_before: float,
        inventory_value_after: float,
    ) -> None:
        """Record actual outcome to compare against predictions."""
        if agent_id not in self._pending_predictions:
            return

        tick, predicted_threat, predicted_opportunity = self._pending_predictions.pop(agent_id)

        # Compute actual threat/opportunity proxies
        actual_damage = max(0.0, health_before - health_after) / 100.0  # normalized 0-1
        actual_gain = max(0.0, inventory_value_after - inventory_value_before) / 30.0  # normalized
        actual_gain = min(1.0, actual_gain)

        # Compute errors
        threat_error = abs(predicted_threat - actual_damage)
        opportunity_error = abs(predicted_opportunity - actual_gain)

        record = ClassifierAccuracyRecord(
            tick=tick,
            agent_id=agent_id,
            predicted_threat=predicted_threat,
            predicted_opportunity=predicted_opportunity,
            actual_damage_taken=actual_damage,
            actual_resources_gained=actual_gain,
            threat_error=threat_error,
            opportunity_error=opportunity_error,
        )
        self._classifier_records.append(record)

    def score_classifier_accuracy(self, current_tick: int) -> dict[str, float]:
        """Compute rolling accuracy per agent."""
        cutoff = current_tick - self._scoring_interval
        recent = [r for r in self._classifier_records if r.tick >= cutoff]

        # Group by agent
        by_agent: dict[str, list[ClassifierAccuracyRecord]] = {}
        for r in recent:
            if r.agent_id not in by_agent:
                by_agent[r.agent_id] = []
            by_agent[r.agent_id].append(r)

        for agent_id, records in by_agent.items():
            avg_threat_error = sum(r.threat_error for r in records) / len(records)
            avg_opp_error = sum(r.opportunity_error for r in records) / len(records)
            # Accuracy = 1 - average error (0=worst, 1=perfect)
            accuracy = 1.0 - (avg_threat_error + avg_opp_error) / 2.0
            self._classifier_accuracy[agent_id] = max(0.0, accuracy)

        return dict(self._classifier_accuracy)

    def get_classifier_calibration(self, agent_id: str) -> float:
        """Get calibration score for a specific agent's classifier."""
        return self._classifier_accuracy.get(agent_id, 0.5)  # neutral default

    # --- Serialization ---

    def export_state(self) -> dict:
        """Export full engine state for checkpoint persistence."""
        return {
            "nudge_scores": [
                {
                    "nudge_tick": s.nudge_tick,
                    "agent_id": s.agent_id,
                    "event_type": s.event_type,
                    "fitness_before": s.fitness_before,
                    "fitness_after": s.fitness_after,
                    "delta": s.delta,
                    "score": s.score,
                }
                for s in self._nudge_scores
            ],
            "router_records": [
                {
                    "tick": r.tick,
                    "agent_id": r.agent_id,
                    "architecture": r.architecture,
                    "intention_goal": r.intention_goal,
                    "action_succeeded": r.action_succeeded,
                    "needs_delta_sum": r.needs_delta_sum,
                }
                for r in self._router_records
            ],
            "classifier_records": [
                {
                    "tick": r.tick,
                    "agent_id": r.agent_id,
                    "predicted_threat": r.predicted_threat,
                    "predicted_opportunity": r.predicted_opportunity,
                    "actual_damage_taken": r.actual_damage_taken,
                    "actual_resources_gained": r.actual_resources_gained,
                    "threat_error": r.threat_error,
                    "opportunity_error": r.opportunity_error,
                }
                for r in self._classifier_records
            ],
            "quality_scores": dict(self._quality_scores),
            "classifier_accuracy": dict(self._classifier_accuracy),
        }

    def import_state(self, data: dict) -> None:
        """Import state from checkpoint data."""
        # Nudge scores
        self._nudge_scores = [NudgeEffectivenessScore(**s) for s in data.get("nudge_scores", [])]

        # Router records
        self._router_records.clear()
        for r in data.get("router_records", []):
            self._router_records.append(RouterQualityRecord(**r))

        # Classifier records
        self._classifier_records.clear()
        for r in data.get("classifier_records", []):
            self._classifier_records.append(ClassifierAccuracyRecord(**r))

        # Quality scores
        self._quality_scores = dict(data.get("quality_scores", {}))

        # Classifier accuracy
        self._classifier_accuracy = dict(data.get("classifier_accuracy", {}))
