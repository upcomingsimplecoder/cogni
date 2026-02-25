"""Performance timing instrumentation for AUTOCOG simulations."""

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class TickTiming:
    """Timing breakdown for a single tick."""

    decision_ms: float = 0.0
    action_execution_ms: float = 0.0
    message_delivery_ms: float = 0.0
    need_decay_ms: float = 0.0
    memory_update_ms: float = 0.0
    emergence_detection_ms: float = 0.0
    total_ms: float = 0.0


class PerformanceMonitor:
    """Tracks per-phase timing and strategy-specific metrics."""

    def __init__(self):
        self._tick_timings: list[TickTiming] = []
        self._strategy_call_counts: dict[str, int] = defaultdict(int)
        self._strategy_total_ms: dict[str, float] = defaultdict(float)
        self._llm_call_count: int = 0
        self._llm_total_ms: float = 0.0
        self._llm_parse_failures: int = 0
        self._peak_memory_mb: float = 0.0

    def record_tick(self, timing: TickTiming) -> None:
        self._tick_timings.append(timing)

    def record_strategy_call(self, strategy_name: str, duration_ms: float) -> None:
        self._strategy_call_counts[strategy_name] += 1
        self._strategy_total_ms[strategy_name] += duration_ms

    @property
    def summary(self) -> dict:
        if not self._tick_timings:
            return {}
        n = len(self._tick_timings)
        return {
            "total_ticks": n,
            "avg_tick_ms": sum(t.total_ms for t in self._tick_timings) / n,
            "avg_decision_ms": sum(t.decision_ms for t in self._tick_timings) / n,
            "avg_action_ms": sum(t.action_execution_ms for t in self._tick_timings) / n,
            "slowest_tick_ms": max(t.total_ms for t in self._tick_timings),
            "strategy_breakdown": {
                name: {
                    "calls": self._strategy_call_counts[name],
                    "total_ms": round(self._strategy_total_ms[name], 2),
                    "avg_ms": round(
                        self._strategy_total_ms[name] / max(1, self._strategy_call_counts[name]), 2
                    ),
                }
                for name in self._strategy_call_counts
            },
            "llm_calls": self._llm_call_count,
            "llm_avg_ms": round(self._llm_total_ms / max(1, self._llm_call_count), 2),
            "llm_parse_failures": self._llm_parse_failures,
        }
