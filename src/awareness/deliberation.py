"""Deliberation strategies: System 1/System 2 dual-process thinking."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.awareness.types import Reflection, Sensation

if TYPE_CHECKING:
    from src.simulation.entities import Agent


class NullDeliberation:
    """No deliberation (System 1 only).

    Always returns the input reflection unchanged. Fast, reactive decisions.
    """

    def should_escalate(self, sensation: Sensation, reflection: Reflection) -> bool:
        """Never escalate — pure System 1."""
        return False

    def deliberate(self, agent: Agent, sensation: Sensation, reflection: Reflection) -> Reflection:
        """Return reflection as-is."""
        return reflection


class ThresholdDeliberation:
    """System 1/System 2 dual-process model.

    System 1: Fast, uses current reflection as-is
    System 2: Slow, re-evaluates with deeper analysis when:
      - Threat is high (> threat_threshold)
      - Multiple needs are critical
      - Novel situation (unfamiliar agents)
    """

    def __init__(
        self,
        threat_threshold: float = 0.7,
        critical_need_threshold: float = 20.0,
        registry: Any = None,
    ):
        self.threat_threshold = threat_threshold
        self.critical_need_threshold = critical_need_threshold
        self.registry = registry

    def should_escalate(self, sensation: Sensation, reflection: Reflection) -> bool:
        """Determine if System 2 should activate."""
        # High threat triggers deliberation
        if reflection.threat_level > self.threat_threshold:
            return True

        # Multiple critical needs trigger deliberation
        critical_needs = sum(
            1 for v in sensation.own_needs.values() if v < self.critical_need_threshold
        )
        return critical_needs >= 2

    def deliberate(self, agent: Agent, sensation: Sensation, reflection: Reflection) -> Reflection:
        """Deeper analysis: considers memory, social context, multi-step consequences."""
        # Re-evaluate with broader context
        refined = Reflection(
            last_action_succeeded=reflection.last_action_succeeded,
            need_trends=reflection.need_trends,
            recent_interaction_outcomes=reflection.recent_interaction_outcomes,
            threat_level=self._refined_threat(agent, sensation, reflection),
            opportunity_score=self._refined_opportunity(agent, sensation, reflection),
        )
        return refined

    def _refined_threat(self, agent: Agent, sensation: Sensation, reflection: Reflection) -> float:
        """Cross-reference social memory with visible agents.

        Reduces threat for known allies, increases for known enemies.
        """
        threat = reflection.threat_level

        # Cross-reference social memory with visible agents
        if self.registry is not None and agent.profile is not None:
            memory_tuple = self.registry.get_memory(agent.profile.agent_id)
            if memory_tuple is not None:
                _, social_memory = memory_tuple

                for visible_agent in sensation.visible_agents:
                    relationship = social_memory.get(visible_agent.agent_id)
                    if relationship is not None:
                        # Known allies reduce threat
                        if relationship.trust > 0.7:
                            threat -= 0.1
                        # Known enemies increase threat
                        elif relationship.trust < 0.3:
                            threat += 0.15
                            # Extra threat boost for attackers
                            if relationship.was_attacked_by:
                                threat += 0.1

        return max(0.0, min(1.0, threat))

    def _refined_opportunity(
        self,
        agent: Agent,
        sensation: Sensation,
        reflection: Reflection,
    ) -> float:
        """Factor in inventory, nearby allies, resource locations from memory.

        Boosts opportunity if:
        - Agent has inventory space
        - Visible allies nearby (potential for cooperation)
        - Resources on visible tiles
        """
        opportunity = reflection.opportunity_score

        # Boost if inventory has space
        inventory_size = sum(sensation.own_inventory.values())
        if inventory_size < 5:  # Assuming max inventory ~5-10
            opportunity = min(1.0, opportunity + 0.1)

        # Boost if visible agents with good apparent health (potential allies)
        healthy_agents = sum(1 for a in sensation.visible_agents if a.apparent_health == "healthy")
        if healthy_agents > 0:
            opportunity = min(1.0, opportunity + 0.1)

        return max(0.0, min(1.0, opportunity))


class ConsensusDeliberation:
    """Models other agents' likely reactions before committing.

    Precursor to Theory of Mind (Plan 04). Simplified version:
    - If approaching an aggressive agent, consider retreat
    - If sharing with an ally, boost confidence

    This is a placeholder for future Theory of Mind implementation.
    """

    def __init__(self, social_threshold: float = 0.5):
        self.social_threshold = social_threshold

    def should_escalate(self, sensation: Sensation, reflection: Reflection) -> bool:
        """Escalate if other agents are visible."""
        return len(sensation.visible_agents) > 0

    def deliberate(self, agent: Agent, sensation: Sensation, reflection: Reflection) -> Reflection:
        """Consider social implications before acting.

        Simplified version: adjust threat based on visible agents.
        """
        threat = reflection.threat_level
        opportunity = reflection.opportunity_score

        # If multiple agents visible, increase threat slightly (social complexity)
        if len(sensation.visible_agents) >= 2:
            threat = min(1.0, threat + 0.1)

        # If any agents are injured, reduce threat (less competition)
        injured_count = sum(
            1 for a in sensation.visible_agents if a.apparent_health in ("injured", "critical")
        )
        if injured_count > 0:
            threat = max(0.0, threat - 0.15)

        refined = Reflection(
            last_action_succeeded=reflection.last_action_succeeded,
            need_trends=reflection.need_trends,
            recent_interaction_outcomes=reflection.recent_interaction_outcomes,
            threat_level=threat,
            opportunity_score=opportunity,
        )
        return refined
