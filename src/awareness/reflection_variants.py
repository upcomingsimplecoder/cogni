"""Reflection strategy variants: different ways agents evaluate their situation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.awareness.reflection import ReflectionModule
from src.awareness.types import Reflection, Sensation

if TYPE_CHECKING:
    from src.simulation.entities import Agent


class OptimisticReflection:
    """Sees the glass half full. Underestimates threats, overestimates resources.

    Biases opportunity_score upward and threat_level downward.
    """

    def evaluate(self, agent: Agent, engine: Any, sensation: Sensation) -> Reflection:
        """Evaluate with optimistic bias."""
        base = ReflectionModule()
        reflection = base.evaluate(agent, engine, sensation)

        # Bias: dampen threat, amplify opportunity
        reflection.threat_level = max(0.0, reflection.threat_level * 0.6)
        reflection.opportunity_score = min(1.0, reflection.opportunity_score * 1.4)

        return reflection


class PessimisticReflection:
    """Sees danger everywhere. Overestimates threats, underestimates resources.

    Biases threat_level upward and opportunity_score downward.
    """

    def evaluate(self, agent: Agent, engine: Any, sensation: Sensation) -> Reflection:
        """Evaluate with pessimistic bias."""
        base = ReflectionModule()
        reflection = base.evaluate(agent, engine, sensation)

        # Bias: amplify threat, dampen opportunity
        reflection.threat_level = min(1.0, reflection.threat_level * 1.5)
        reflection.opportunity_score = max(0.0, reflection.opportunity_score * 0.6)

        return reflection


class SocialReflection:
    """Evaluates situations primarily through social lens.

    Weights interactions heavily in evaluation. Boosts threat if recent negative
    interactions, boosts opportunity if allies are visible.
    """

    def evaluate(self, agent: Agent, engine: Any, sensation: Sensation) -> Reflection:
        """Evaluate with social focus."""
        base = ReflectionModule()
        reflection = base.evaluate(agent, engine, sensation)

        # Boost threat if recent negative interactions
        negative_count = sum(
            1 for i in reflection.recent_interaction_outcomes if not i.was_positive
        )
        reflection.threat_level = min(1.0, reflection.threat_level + negative_count * 0.15)

        # Boost opportunity if allies visible
        social_mem = engine.registry.get_memory(agent.agent_id)
        if social_mem:
            _, social = social_mem
            for vis_agent in sensation.visible_agents:
                rel = social.get(vis_agent.agent_id)
                if rel and rel.trust > 0.7:
                    reflection.opportunity_score = min(1.0, reflection.opportunity_score + 0.2)
                    break  # Only boost once per ally

        return reflection
