"""Coalition formation decision logic.

Determines when agents should propose or accept coalitions based on
personality traits, social relationships, and current needs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.identity import PersonalityTraits
    from src.awareness.types import AgentSummary
    from src.memory.social import SocialMemory


class CoalitionFormation:
    """Decision logic for coalition formation.

    Uses personality traits, social trust, and context to decide
    when to propose coalitions and whether to accept proposals.
    """

    def __init__(self, sociability_threshold: float = 0.6):
        """Initialize formation logic.

        Args:
            sociability_threshold: Minimum sociability to consider coalitions
        """
        self.sociability_threshold = sociability_threshold

    def should_propose(
        self,
        agent_traits: PersonalityTraits,
        visible_agents: list[AgentSummary],
        social_memory: SocialMemory,
        current_needs: dict[str, float],
    ) -> bool:
        """Decide if agent should propose a coalition.

        Args:
            agent_traits: Agent's personality traits
            visible_agents: Agents visible to proposer
            social_memory: Social relationship memory
            current_needs: Current need levels (0-100 scale)

        Returns:
            True if agent should attempt to form a coalition
        """
        # Must have high sociability
        if agent_traits.sociability < self.sociability_threshold:
            return False

        # Need at least 1 other agent visible
        if len(visible_agents) == 0:
            return False

        # More likely if cooperative and sees potential allies
        if agent_traits.cooperation_tendency < 0.4:
            return False

        # Check if we have trusted agents nearby
        trusted_count = 0
        for agent_summary in visible_agents:
            rel = social_memory.get(agent_summary.agent_id)
            if rel and rel.trust >= 0.6:
                trusted_count += 1

        # Propose if we have at least 1 trusted agent
        return trusted_count >= 1

    def should_accept(
        self,
        agent_traits: PersonalityTraits,
        proposer_id: str,
        social_memory: SocialMemory,
        proposed_goal: str,
        current_needs: dict[str, float],
    ) -> bool:
        """Decide if agent should accept a coalition proposal.

        Args:
            agent_traits: Agent's personality traits
            proposer_id: ID of agent proposing the coalition
            social_memory: Social relationship memory
            proposed_goal: Coalition's shared goal
            current_needs: Current need levels (0-100 scale)

        Returns:
            True if agent should accept the proposal
        """
        # Check trust with proposer
        rel = social_memory.get(proposer_id)
        if rel is None:
            # Unknown agent - accept based on cooperation tendency
            return agent_traits.cooperation_tendency >= 0.6

        # Don't join if proposer attacked us
        if rel.was_attacked_by:
            return False

        # Accept if trust is high
        if rel.trust >= 0.7:
            return True

        # Accept if trust is moderate and we're cooperative
        if rel.trust >= 0.5 and agent_traits.cooperation_tendency >= 0.6:
            return True

        # Consider goal urgency
        if proposed_goal in ("hunt", "gather"):
            # Accept if we're hungry and proposer is trusted
            hunger = current_needs.get("hunger", 100.0)
            if hunger < 40 and rel.trust >= 0.5:
                return True

        if proposed_goal == "defend":
            # Accept if health is low and proposer is trusted
            health = current_needs.get("health", 100.0)
            if health < 60 and rel.trust >= 0.5:
                return True

        # Default: reject if trust is low
        return False

    def select_coalition_targets(
        self,
        agent_id: str,
        visible_agents: list[AgentSummary],
        social_memory: SocialMemory,
        max_targets: int = 3,
    ) -> list[str]:
        """Select which agents to invite to a coalition.

        Args:
            agent_id: ID of agent proposing coalition
            visible_agents: Agents visible to proposer
            social_memory: Social relationship memory
            max_targets: Maximum number of agents to invite

        Returns:
            List of agent IDs to invite (sorted by trust, highest first)
        """
        candidates = []

        for agent_summary in visible_agents:
            other_id = str(agent_summary.agent_id)
            if other_id == agent_id:
                continue

            rel = social_memory.get(other_id)
            trust = 0.5 if rel is None else rel.trust  # Neutral trust for unknown agents

            # Filter out enemies
            if trust < 0.3:
                continue

            candidates.append((other_id, trust))

        # Sort by trust descending, take top N
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [agent_id for agent_id, _ in candidates[:max_targets]]

    def suggest_coalition_goal(
        self,
        agent_traits: PersonalityTraits,
        current_needs: dict[str, float],
        visible_agents: list[AgentSummary],
    ) -> str:
        """Suggest a goal for a new coalition based on context.

        Args:
            agent_traits: Agent's personality traits
            current_needs: Current need levels (0-100 scale)
            visible_agents: Agents visible to proposer

        Returns:
            Suggested coalition goal
        """
        hunger = current_needs.get("hunger", 100.0)
        health = current_needs.get("health", 100.0)

        # Survival goals based on urgent needs
        if hunger < 40:
            return "hunt"
        if health < 60:
            return "defend"

        # Social goals based on traits
        if agent_traits.curiosity >= 0.7:
            return "explore"
        if agent_traits.cooperation_tendency >= 0.7:
            return "trade"

        # Default goal
        return "cooperate"
