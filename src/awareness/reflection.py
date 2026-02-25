"""Reflection module: evaluates recent experience for decision-making."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.awareness.types import InteractionOutcome, Reflection, Sensation

if TYPE_CHECKING:
    from src.simulation.entities import Agent


class ReflectionModule:
    """Evaluates recent experience for a specific agent.

    Uses tick history and episodic memory to assess:
    - Did recent actions work?
    - Are needs trending well or poorly?
    - Have recent interactions been positive or negative?
    - Current threat level and opportunity score
    """

    def evaluate(self, agent: Agent, engine: Any, sensation: Sensation) -> Reflection:
        """Build a Reflection from recent history and current perception."""
        last_succeeded = self._check_last_action(engine, agent)
        need_trends = self._compute_need_trends(engine, agent)
        interactions = self._get_recent_interactions(engine, agent)
        threat = self._assess_threat(sensation)
        opportunity = self._assess_opportunity(sensation)

        return Reflection(
            last_action_succeeded=last_succeeded,
            need_trends=need_trends,
            recent_interaction_outcomes=interactions,
            threat_level=threat,
            opportunity_score=opportunity,
        )

    def _check_last_action(self, engine: Any, agent: Agent) -> bool:
        """Check if the last action was successful."""
        if not engine.state.history:
            return True  # No history yet, assume ok

        last_record = engine.state.history[-1]

        # Multi-agent: look for this agent's record
        if hasattr(last_record, "agent_records"):
            for ar in last_record.agent_records:
                if ar.agent_id == agent.agent_id:
                    success: bool = ar.result.success if ar.result else True
                    return success
            return True

        # Single-agent backward compat
        if hasattr(last_record, "result") and last_record.result:
            success_compat: bool = last_record.result.success
            return success_compat
        return True

    def _compute_need_trends(self, engine: Any, agent: Agent) -> dict[str, str]:
        """Compute need trends from last 5 ticks of history.

        Returns dict mapping need name to "declining", "stable", or "improving".
        """
        trends: dict[str, str] = {}
        history = engine.state.history

        if len(history) < 2:
            return {
                "hunger": "stable",
                "thirst": "stable",
                "energy": "stable",
                "health": "stable",
            }

        # Get the last 5 records
        recent = history[-5:]

        for need in ("hunger", "thirst", "energy", "health"):
            # Extract values from history
            values = []
            for record in recent:
                if hasattr(record, "agent_records"):
                    # Multi-agent record
                    for ar in record.agent_records:
                        if ar.agent_id == agent.agent_id:
                            val = ar.needs_after.get(need, 50)
                            values.append(val)
                            break
                elif hasattr(record, "agent_needs_after"):
                    # Single-agent record
                    val = record.agent_needs_after.get(need, 50)
                    values.append(val)

            if len(values) < 2:
                trends[need] = "stable"
                continue

            # Compare first half to second half
            mid = len(values) // 2
            first_avg = sum(values[:mid]) / mid if mid > 0 else values[0]
            second_avg = (
                sum(values[mid:]) / (len(values) - mid) if len(values) > mid else values[-1]
            )

            diff = second_avg - first_avg
            if diff > 2.0:
                trends[need] = "improving"
            elif diff < -2.0:
                trends[need] = "declining"
            else:
                trends[need] = "stable"

        return trends

    def _get_recent_interactions(self, engine: Any, agent: Agent) -> list[InteractionOutcome]:
        """Pull recent interaction outcomes from episodic memory."""
        if hasattr(engine, "registry"):
            memory = engine.registry.get_memory(agent.agent_id)
            if memory:
                episodic, _ = memory
                raw = episodic.recent_interactions(n=5)
                return [
                    InteractionOutcome(
                        other_agent_id=i.other_agent_id,
                        tick=i.tick,
                        was_positive=i.was_positive,
                        interaction_type=i.interaction_type,
                    )
                    for i in raw
                ]
        return []

    def _assess_threat(self, sensation: Sensation) -> float:
        """0-1 threat level based on nearby agents + own vulnerability."""
        threat = 0.0

        # Low health = vulnerable
        health = sensation.own_needs.get("health", 100)
        health_factor = 1.0 - health / 100.0
        threat += health_factor * 0.3

        # Low hunger/thirst = weakened
        hunger = sensation.own_needs.get("hunger", 100)
        thirst = sensation.own_needs.get("thirst", 100)
        if hunger < 20 or thirst < 20:
            threat += 0.2

        # Nearby agents increase threat (proportional to count)
        agent_count = len(sensation.visible_agents)
        threat += min(agent_count * 0.1, 0.4)

        # Injured nearby agents are less threatening
        injured_count = sum(
            1 for a in sensation.visible_agents if a.apparent_health in ("injured", "critical")
        )
        threat -= injured_count * 0.05

        return max(0.0, min(1.0, threat))

    def _assess_opportunity(self, sensation: Sensation) -> float:
        """0-1 opportunity score based on nearby resources."""
        resource_count = sum(qty for tile in sensation.visible_tiles for _, qty in tile.resources)
        # Diminishing returns
        return min(resource_count / 30.0, 1.0)
