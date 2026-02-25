"""Coalition dissolution detection.

Monitors coalition health and triggers dissolution when
cohesion drops too low or goals become unachievable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.memory.social import SocialMemory
    from src.social.coalition import Coalition


class DissolutionDetector:
    """Detects when coalitions should be dissolved.

    Monitors cohesion, trust, and effectiveness to determine
    when a coalition is no longer viable.
    """

    def __init__(
        self,
        min_cohesion: float = 0.3,
        min_effectiveness: float = 0.2,
        min_trust_threshold: float = 0.3,
        max_age_ticks: int = 500,
    ):
        """Initialize dissolution detector.

        Args:
            min_cohesion: Minimum cohesion before dissolution
            min_effectiveness: Minimum effectiveness before dissolution
            min_trust_threshold: Minimum average trust before dissolution
            max_age_ticks: Maximum coalition age before natural dissolution
        """
        self.min_cohesion = min_cohesion
        self.min_effectiveness = min_effectiveness
        self.min_trust_threshold = min_trust_threshold
        self.max_age_ticks = max_age_ticks

    def check(
        self,
        coalition: Coalition,
        tick: int,
        social_memories: dict[str, SocialMemory],
    ) -> str | None:
        """Check if coalition should be dissolved.

        Args:
            coalition: Coalition to evaluate
            tick: Current simulation tick
            social_memories: Map from agent_id to SocialMemory

        Returns:
            Dissolution reason if should dissolve, None otherwise
        """
        # Too small
        if coalition.size < 2:
            return "insufficient_members"

        # Too old
        age = tick - coalition.formation_tick
        if age > self.max_age_ticks:
            return "expired"

        # Low cohesion
        if coalition.cohesion < self.min_cohesion:
            return "low_cohesion"

        # Low effectiveness (only check after coalition has had time to act)
        if age > 20 and coalition.effectiveness < self.min_effectiveness:
            return "ineffective"

        # Check trust breakdown among members
        trust_issue = self._check_trust_breakdown(coalition, social_memories)
        if trust_issue:
            return trust_issue

        # Coalition is healthy
        return None

    def _check_trust_breakdown(
        self,
        coalition: Coalition,
        social_memories: dict[str, SocialMemory],
    ) -> str | None:
        """Check if trust has broken down among coalition members.

        Args:
            coalition: Coalition to evaluate
            social_memories: Map from agent_id to SocialMemory

        Returns:
            "trust_breakdown" if trust is too low, None otherwise
        """
        trust_values = []

        # Check all pairwise trust relationships
        members = list(coalition.members)
        for i, agent_id in enumerate(members):
            social_mem = social_memories.get(agent_id)
            if social_mem is None:
                continue

            for other_id in members[i + 1 :]:
                rel = social_mem.get(other_id)
                if rel is not None:
                    trust_values.append(rel.trust)

        if not trust_values:
            return None

        # Check if average trust is too low
        avg_trust = sum(trust_values) / len(trust_values)
        if avg_trust < self.min_trust_threshold:
            return "trust_breakdown"

        # Check if any member was attacked by another
        for agent_id in members:
            social_mem = social_memories.get(agent_id)
            if social_mem is None:
                continue

            for other_id in members:
                if other_id == agent_id:
                    continue

                rel = social_mem.get(other_id)
                if rel and rel.was_attacked_by:
                    return "internal_conflict"

        return None

    def predict_dissolution_risk(
        self,
        coalition: Coalition,
        tick: int,
        social_memories: dict[str, SocialMemory],
    ) -> float:
        """Predict risk of coalition dissolution.

        Args:
            coalition: Coalition to evaluate
            tick: Current simulation tick
            social_memories: Map from agent_id to SocialMemory

        Returns:
            Risk score from 0.0 (stable) to 1.0 (imminent dissolution)
        """
        risk = 0.0

        # Size risk
        if coalition.size < 2:
            return 1.0
        elif coalition.size == 2:
            risk += 0.3

        # Age risk
        age = tick - coalition.formation_tick
        if age > self.max_age_ticks * 0.8:
            risk += 0.3

        # Cohesion risk
        if coalition.cohesion < self.min_cohesion * 1.5:
            risk += (1.0 - coalition.cohesion / (self.min_cohesion * 1.5)) * 0.3

        # Effectiveness risk
        if age > 20 and coalition.effectiveness < self.min_effectiveness * 1.5:
            risk += (1.0 - coalition.effectiveness / (self.min_effectiveness * 1.5)) * 0.2

        # Trust risk
        trust_values = []
        members = list(coalition.members)
        for i, agent_id in enumerate(members):
            social_mem = social_memories.get(agent_id)
            if social_mem is None:
                continue

            for other_id in members[i + 1 :]:
                rel = social_mem.get(other_id)
                if rel is not None:
                    trust_values.append(rel.trust)

        if trust_values:
            avg_trust = sum(trust_values) / len(trust_values)
            if avg_trust < self.min_trust_threshold * 1.5:
                risk += (1.0 - avg_trust / (self.min_trust_threshold * 1.5)) * 0.2

        return min(1.0, risk)
