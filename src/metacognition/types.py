"""Core data types for the metacognition system.

Provides frozen records (per-tick snapshots), mutable per-agent state trackers,
and the AgentMetacogState bundle that centralizes all metacognitive state per agent.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

# ============================================================================
# Frozen Records (immutable per-tick snapshots)
# ============================================================================


@dataclass(frozen=True)
class PredictionRecord:
    """A single prediction the agent made and its outcome.

    Atomic unit of calibration data. Created once per tick, never mutated.
    """

    tick: int
    domain: str  # "action_success", "threat_assessment"
    predicted_confidence: float  # 0.0-1.0
    actual_outcome: str  # "succeed", "fail", "attacked", "safe"
    was_correct: bool
    context_tag: str  # from ContextTag
    strategy_name: str  # which strategy produced this


@dataclass(frozen=True)
class StrategyOutcomeRecord:
    """Outcome of one strategy usage in a specific context.

    Captures which strategy was used, in what context, and what happened.
    """

    tick: int
    strategy_name: str
    context_tag: str
    action_type: str  # ActionType.value
    succeeded: bool
    needs_delta_sum: float  # sum of needs changes (fitness proxy)
    confidence: float  # Intention.confidence when action was chosen
    deliberation_used: bool


@dataclass(frozen=True)
class MetacognitiveSnapshot:
    """Per-agent per-tick metacognitive state for trajectory/visualization.

    Serialized to trajectory JSONL and broadcast via WebSocket.
    """

    tick: int
    agent_id: str
    active_strategy: str
    calibration_score: float  # 0.0-1.0 (higher = better)
    self_awareness_score: float  # 0.0-1.0 composite
    confidence_bias: float  # negative = under, positive = overconfident
    strategy_switch_this_tick: bool
    previous_strategy: str  # empty if no switch
    switch_reason: str  # "underperforming", "context_change", ""
    deliberation_threshold: float
    self_model_summary: dict  # {"gathering": 0.8, ...}


# ============================================================================
# Mutable Per-Agent State
# ============================================================================


@dataclass
class StrategyPerformance:
    """Rolling performance tracker for one strategy in one context.

    Updated each tick when this (strategy, context) pair is used.
    Memory-bounded by deque maxlen.
    """

    strategy_name: str
    context_tag: str
    total_uses: int = 0
    total_successes: int = 0
    total_fitness_delta: float = 0.0
    recent_outcomes: deque = field(default_factory=lambda: deque(maxlen=50))

    def record(self, succeeded: bool, fitness_delta: float) -> None:
        """Record a new outcome for this strategy/context pair."""
        self.total_uses += 1
        if succeeded:
            self.total_successes += 1
        self.total_fitness_delta += fitness_delta
        self.recent_outcomes.append(succeeded)

    @property
    def success_rate(self) -> float:
        """Overall success rate. Returns 0.5 uninformative prior if no data."""
        if self.total_uses == 0:
            return 0.5
        return self.total_successes / self.total_uses

    @property
    def recent_success_rate(self) -> float:
        """Success rate over recent window only."""
        if not self.recent_outcomes:
            return 0.5
        return float(sum(self.recent_outcomes)) / len(self.recent_outcomes)

    @property
    def avg_fitness_delta(self) -> float:
        """Average fitness delta per use."""
        if self.total_uses == 0:
            return 0.0
        return self.total_fitness_delta / self.total_uses


class CalibrationTracker:
    """Tracks prediction accuracy for confidence calibration.

    Groups predictions into confidence bins (0.0-0.1, ..., 0.9-1.0)
    and compares predicted confidence to actual accuracy in each bin.

    Memory-bounded: stores rolling window of last MAX_PREDICTIONS records.
    """

    MAX_PREDICTIONS = 500
    NUM_BINS = 10
    MIN_BIN_SIZE = 5  # Skip bins with fewer samples in bias calculation

    def __init__(self):
        self._predictions: deque[PredictionRecord] = deque(maxlen=self.MAX_PREDICTIONS)

    def record(self, prediction: PredictionRecord) -> None:
        """Record a new prediction for calibration tracking."""
        self._predictions.append(prediction)

    @property
    def count(self) -> int:
        """Number of predictions currently tracked."""
        return len(self._predictions)

    @property
    def brier_score(self) -> float:
        """Brier score: mean squared error of probabilistic predictions.

        Lower = better calibration. Returns 0.25 (uninformative) if no data.
        """
        if not self._predictions:
            return 0.25
        total = sum(
            (p.predicted_confidence - (1.0 if p.was_correct else 0.0)) ** 2
            for p in self._predictions
        )
        return total / len(self._predictions)

    @property
    def calibration_score(self) -> float:
        """1.0 - brier_score. Higher = better. Range [0, 1]."""
        return max(0.0, 1.0 - self.brier_score)

    @property
    def expected_calibration_error(self) -> float:
        """ECE: weighted average of |bin_accuracy - bin_confidence| per bin.

        Standard calibration metric. Used by controller for decision-making.
        Returns 0.25 (uninformative) if insufficient data.
        """
        if len(self._predictions) < self.MIN_BIN_SIZE:
            return 0.25
        bins = self._build_bins()
        total_weight = 0
        weighted_error = 0.0
        for bin_idx, preds in bins.items():
            if not preds:
                continue
            bin_center = (bin_idx + 0.5) / self.NUM_BINS
            actual_accuracy = sum(1 for p in preds if p.was_correct) / len(preds)
            weighted_error += abs(bin_center - actual_accuracy) * len(preds)
            total_weight += len(preds)
        if total_weight == 0:
            return 0.25
        return weighted_error / total_weight

    @property
    def confidence_bias(self) -> float:
        """Average (confidence - accuracy) across bins with enough samples.

        Positive = overconfident, negative = underconfident.
        Returns 0.0 if insufficient data.
        """
        if len(self._predictions) < self.MIN_BIN_SIZE:
            return 0.0
        bins = self._build_bins()
        weighted_bias = 0.0
        total_weight = 0
        for bin_idx, preds in bins.items():
            if len(preds) < self.MIN_BIN_SIZE:
                continue
            bin_center = (bin_idx + 0.5) / self.NUM_BINS
            actual_accuracy = sum(1 for p in preds if p.was_correct) / len(preds)
            weighted_bias += (bin_center - actual_accuracy) * len(preds)
            total_weight += len(preds)
        if total_weight == 0:
            return 0.0
        return weighted_bias / total_weight

    def calibration_curve(self) -> list[tuple[float, float, int]]:
        """Return (bin_center, actual_accuracy, count) for each bin.

        Used for visualization (calibration plot in agent inspector).
        """
        bins = self._build_bins()
        result = []
        for bin_idx in range(self.NUM_BINS):
            preds = bins[bin_idx]
            bin_center = (bin_idx + 0.5) / self.NUM_BINS
            accuracy = sum(1 for p in preds if p.was_correct) / len(preds) if preds else 0.0
            result.append((bin_center, accuracy, len(preds)))
        return result

    def recent(self, n: int = 20) -> list[PredictionRecord]:
        """Return the last N predictions."""
        return list(self._predictions)[-n:]

    def _build_bins(self) -> dict[int, list[PredictionRecord]]:
        """Group predictions into confidence bins."""
        bins: dict[int, list[PredictionRecord]] = {i: [] for i in range(self.NUM_BINS)}
        for p in self._predictions:
            bin_idx = min(int(p.predicted_confidence * self.NUM_BINS), self.NUM_BINS - 1)
            bins[bin_idx].append(p)
        return bins


@dataclass
class CapabilityRating:
    """Agent's self-assessment in one domain."""

    domain: str
    rating: float = 0.5  # 0.0-1.0
    confidence: float = 0.0  # 0.0-1.0: how sure am I?
    sample_count: int = 0
    bias_estimate: float = 0.0  # positive = overestimate


@dataclass
class AgentMetacogState:
    """Bundle of all per-agent metacognitive state.

    Used as single dict value in MetacognitiveEngine and as input
    to MetacognitiveController.decide(). Avoids parallel dicts.
    """

    calibration: CalibrationTracker = field(default_factory=CalibrationTracker)
    strategy_performance: dict = field(default_factory=dict)  # (str,str) → StrategyPerformance
    self_model: object = None  # CognitiveSelfModel (forward ref to avoid circular import)
    deliberation_threshold: float = 0.7
    current_strategy_name: str = "personality"
    available_strategies: list = field(default_factory=lambda: ["personality", "planning"])
    switch_history: deque = field(default_factory=lambda: deque(maxlen=200))
    last_switch_tick: int = -999
    underperformance_streak: int = 0
