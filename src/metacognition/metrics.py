"""Population-level metacognition metrics and analysis.

MetacognitiveAnalyzer records per-tick aggregate statistics from the
MetacognitiveEngine for post-simulation analysis and visualization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.metacognition.engine import MetacognitiveEngine


@dataclass(frozen=True)
class MetacognitiveTickSnapshot:
    """Aggregate metacognitive state at one tick."""

    tick: int
    total_agents: int
    avg_calibration_score: float
    avg_self_awareness_score: float
    strategy_distribution: dict[str, int]
    total_switches_cumulative: int
    switches_this_tick: int
    avg_deliberation_threshold: float
    avg_confidence_bias: float


class MetacognitiveAnalyzer:
    """Records and queries population-level metacognitive data per tick.

    Called once per tick by SimulationEngine after MetacognitiveEngine.tick().
    Stores a rolling history of MetacognitiveTickSnapshot for analysis.
    """

    MAX_HISTORY = 5000

    def __init__(self):
        self._history: list[MetacognitiveTickSnapshot] = []
        self._previous_total_switches: int = 0

    def record_tick(
        self,
        metacognition_engine: MetacognitiveEngine,
        tick: int,
    ) -> MetacognitiveTickSnapshot:
        """Record aggregate metacognitive state for this tick.

        Args:
            metacognition_engine: The metacognitive engine instance
            tick: Current simulation tick

        Returns:
            The recorded MetacognitiveTickSnapshot
        """
        stats = metacognition_engine.get_metacognitive_stats()

        total_agents = stats["total_agents_tracked"]
        avg_calibration = stats["avg_calibration_score"]
        strategy_distribution = stats["strategy_distribution"]
        total_switches = stats["total_switches"]
        switches_this_tick = total_switches - self._previous_total_switches
        self._previous_total_switches = total_switches

        # Compute averages from individual agent states
        avg_self_awareness = 0.0
        avg_threshold = 0.0
        avg_bias = 0.0

        if total_agents > 0:
            total_awareness = 0.0
            total_threshold = 0.0
            total_bias = 0.0

            for agent_id in list(metacognition_engine._agents.keys()):
                state = metacognition_engine._agents[agent_id]
                total_awareness += metacognition_engine._compute_self_awareness_score(agent_id)
                total_threshold += state.deliberation_threshold
                total_bias += state.calibration.confidence_bias

            avg_self_awareness = total_awareness / total_agents
            avg_threshold = total_threshold / total_agents
            avg_bias = total_bias / total_agents

        snapshot = MetacognitiveTickSnapshot(
            tick=tick,
            total_agents=total_agents,
            avg_calibration_score=avg_calibration,
            avg_self_awareness_score=avg_self_awareness,
            strategy_distribution=dict(strategy_distribution),
            total_switches_cumulative=total_switches,
            switches_this_tick=switches_this_tick,
            avg_deliberation_threshold=avg_threshold,
            avg_confidence_bias=avg_bias,
        )

        self._history.append(snapshot)

        # Memory bound
        if len(self._history) > self.MAX_HISTORY:
            self._history = self._history[-self.MAX_HISTORY :]

        return snapshot

    @property
    def history(self) -> list[MetacognitiveTickSnapshot]:
        """All recorded tick snapshots."""
        return self._history

    def population_calibration(self, last_n: int = 50) -> list[tuple[int, float]]:
        """Return (tick, avg_calibration_score) for recent ticks.

        Args:
            last_n: Number of recent ticks to return

        Returns:
            List of (tick, avg_calibration_score) tuples
        """
        return [(s.tick, s.avg_calibration_score) for s in self._history[-last_n:]]

    def strategy_distribution_history(self, last_n: int = 50) -> list[tuple[int, dict[str, int]]]:
        """Return (tick, strategy_distribution) for recent ticks.

        Args:
            last_n: Number of recent ticks to return

        Returns:
            List of (tick, strategy_distribution) tuples
        """
        return [(s.tick, s.strategy_distribution) for s in self._history[-last_n:]]

    def total_switches(self) -> int:
        """Total strategy switches across the simulation."""
        if not self._history:
            return 0
        return self._history[-1].total_switches_cumulative

    def to_summary(self) -> dict[str, Any]:
        """Return summary statistics for the entire simulation run.

        Returns:
            Dict with aggregate statistics
        """
        if not self._history:
            return {
                "total_ticks_recorded": 0,
                "total_switches": 0,
                "final_avg_calibration": 0.0,
                "final_avg_self_awareness": 0.0,
                "final_strategy_distribution": {},
            }

        final = self._history[-1]
        return {
            "total_ticks_recorded": len(self._history),
            "total_switches": final.total_switches_cumulative,
            "final_avg_calibration": final.avg_calibration_score,
            "final_avg_self_awareness": final.avg_self_awareness_score,
            "final_strategy_distribution": final.strategy_distribution,
            "final_avg_deliberation_threshold": final.avg_deliberation_threshold,
            "final_avg_confidence_bias": final.avg_confidence_bias,
        }
