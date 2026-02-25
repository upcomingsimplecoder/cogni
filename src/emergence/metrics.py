"""Aggregate metrics collection for simulation analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TickMetrics:
    """Aggregate metrics for a single simulation tick."""

    tick: int
    agents_alive: int = 0
    agents_dead: int = 0
    total_messages_sent: int = 0
    cooperation_events: int = 0  # GIVE actions this tick
    aggression_events: int = 0  # ATTACK actions this tick
    trade_proposals: int = 0  # NEGOTIATE messages
    resource_sharing_rate: float = 0.0
    avg_health: float = 0.0
    avg_hunger: float = 0.0
    avg_thirst: float = 0.0
    avg_energy: float = 0.0
    cluster_count: int = 0
    territory_claims: int = 0


class MetricsCollector:
    """Collects per-tick metrics and maintains time series."""

    def __init__(self):
        self.history: list[TickMetrics] = []

    def collect(
        self,
        tick: int,
        living_agents: list[Any],
        dead_count: int,
        tick_actions: list[tuple[str, bool]] | None = None,
        messages_sent: int = 0,
        trade_proposals: int = 0,
        cluster_count: int = 0,
        territory_claims: int = 0,
    ) -> TickMetrics:
        """Build metrics for this tick.

        Args:
            tick: Current simulation tick
            living_agents: List of living Agent objects
            dead_count: Number of dead agents
            tick_actions: List of (action_type, success) from this tick
            messages_sent: Messages delivered this tick
            trade_proposals: NEGOTIATE messages this tick
            cluster_count: Detected clusters
            territory_claims: Detected territories
        """
        n = len(living_agents)

        # Compute averages
        avg_health = 0.0
        avg_hunger = 0.0
        avg_thirst = 0.0
        avg_energy = 0.0
        if n > 0:
            avg_health = sum(a.needs.health for a in living_agents) / n
            avg_hunger = sum(a.needs.hunger for a in living_agents) / n
            avg_thirst = sum(a.needs.thirst for a in living_agents) / n
            avg_energy = sum(a.needs.energy for a in living_agents) / n

        # Count cooperation and aggression
        cooperation = 0
        aggression = 0
        if tick_actions:
            for action_type, success in tick_actions:
                if action_type == "give" and success:
                    cooperation += 1
                elif action_type == "attack" and success:
                    aggression += 1

        sharing_rate = cooperation / n if n > 0 else 0.0

        metrics = TickMetrics(
            tick=tick,
            agents_alive=n,
            agents_dead=dead_count,
            total_messages_sent=messages_sent,
            cooperation_events=cooperation,
            aggression_events=aggression,
            trade_proposals=trade_proposals,
            resource_sharing_rate=sharing_rate,
            avg_health=avg_health,
            avg_hunger=avg_hunger,
            avg_thirst=avg_thirst,
            avg_energy=avg_energy,
            cluster_count=cluster_count,
            territory_claims=territory_claims,
        )
        self.history.append(metrics)
        return metrics

    def trend(self, metric_name: str, window: int = 50) -> str:
        """Get trend for a metric over last `window` ticks.

        Returns: "increasing", "stable", or "decreasing"
        """
        if len(self.history) < 2:
            return "stable"

        recent = self.history[-window:]
        if len(recent) < 2:
            return "stable"

        values = [getattr(m, metric_name, 0) for m in recent]
        first_half = values[: len(values) // 2]
        second_half = values[len(values) // 2 :]

        avg_first = sum(first_half) / len(first_half) if first_half else 0
        avg_second = sum(second_half) / len(second_half) if second_half else 0

        threshold = 0.05 * max(abs(avg_first), abs(avg_second), 1.0)

        if avg_second > avg_first + threshold:
            return "increasing"
        elif avg_second < avg_first - threshold:
            return "decreasing"
        return "stable"

    def latest(self) -> TickMetrics | None:
        """Most recent tick metrics."""
        return self.history[-1] if self.history else None

    @property
    def total_ticks(self) -> int:
        return len(self.history)
