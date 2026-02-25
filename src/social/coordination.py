"""Coalition coordination: role assignment and action suggestions.

Coordinates coalition members by assigning roles based on traits
and suggesting coordinated actions for each member.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.identity import PersonalityTraits
    from src.awareness.types import Sensation
    from src.social.coalition import Coalition


class CoalitionCoordinator:
    """Coordinates actions within a coalition.

    Assigns roles to members based on traits and suggests
    coordinated actions that align with coalition goals.
    """

    def assign_roles(
        self,
        coalition: Coalition,
        member_traits: dict[str, PersonalityTraits],
    ) -> dict[str, str]:
        """Assign roles to coalition members based on traits.

        Args:
            coalition: Coalition to assign roles for
            member_traits: Map from agent_id to PersonalityTraits

        Returns:
            Dict mapping agent_id to role name
        """
        roles = {}

        # Leader gets their role automatically
        roles[coalition.leader_id] = "leader"

        # Assign roles to other members
        for agent_id in coalition.members:
            if agent_id == coalition.leader_id:
                continue

            traits = member_traits.get(agent_id)
            if traits is None:
                roles[agent_id] = "member"
                continue

            # Role assignment based on dominant trait
            if traits.aggression >= 0.7:
                role = "enforcer"
            elif traits.curiosity >= 0.7:
                role = "scout"
            elif traits.resource_sharing >= 0.7:
                role = "gatherer"
            elif traits.cooperation_tendency >= 0.7:
                role = "diplomat"
            else:
                role = "member"

            roles[agent_id] = role

        # Update coalition roles
        coalition.roles = roles.copy()

        return roles

    def suggest_action(
        self,
        coalition: Coalition,
        agent_id: str,
        role: str,
        sensation: Sensation,
    ) -> str | None:
        """Suggest a coordinated action for a coalition member.

        Args:
            coalition: Coalition the agent belongs to
            agent_id: Agent requesting action suggestion
            role: Agent's role in the coalition
            sensation: Agent's current perception

        Returns:
            Suggested action name, or None if no specific suggestion
        """
        goal = coalition.shared_goal

        if goal == "hunt":
            return self._suggest_hunt_action(role, sensation)
        elif goal == "gather":
            return self._suggest_gather_action(role, sensation)
        elif goal == "defend":
            return self._suggest_defend_action(role, sensation)
        elif goal == "explore":
            return self._suggest_explore_action(role, sensation)
        else:
            return None

    def _suggest_hunt_action(self, role: str, sensation: Sensation) -> str | None:
        """Suggest action for hunting goal."""
        if role == "scout":
            return "explore"  # Find resources
        elif role == "gatherer":
            return "gather"  # Collect food
        elif role == "enforcer":
            # Protect gatherers or engage threats
            if sensation.visible_agents:
                return "guard"
            return "gather"
        else:
            return "gather"

    def _suggest_gather_action(self, role: str, sensation: Sensation) -> str | None:
        """Suggest action for gathering goal."""
        if role == "scout":
            return "explore"
        else:
            return "gather"

    def _suggest_defend_action(self, role: str, sensation: Sensation) -> str | None:
        """Suggest action for defense goal."""
        if role == "enforcer":
            return "guard"
        elif role == "scout":
            return "explore"  # Watch for threats
        else:
            return "stay"  # Hold position

    def _suggest_explore_action(self, role: str, sensation: Sensation) -> str | None:
        """Suggest action for exploration goal."""
        if role == "scout" or role == "leader":
            return "explore"
        else:
            return "follow"  # Follow the scout

    def calculate_cohesion(
        self,
        coalition: Coalition,
        member_positions: dict[str, tuple[int, int]],
    ) -> float:
        """Calculate coalition cohesion based on spatial proximity.

        Args:
            coalition: Coalition to evaluate
            member_positions: Map from agent_id to (x, y) position

        Returns:
            Cohesion score from 0.0 (scattered) to 1.0 (tightly grouped)
        """
        if coalition.size < 2:
            return 1.0

        positions = [
            member_positions[agent_id]
            for agent_id in coalition.members
            if agent_id in member_positions
        ]

        if len(positions) < 2:
            return 0.5

        # Calculate average distance between members
        total_distance = 0.0
        pair_count = 0

        for i, pos1 in enumerate(positions):
            for pos2 in positions[i + 1 :]:
                distance = ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
                total_distance += distance
                pair_count += 1

        avg_distance = total_distance / pair_count if pair_count > 0 else 0

        # Convert distance to cohesion (closer = higher cohesion)
        # Distance 0-5: cohesion 1.0-0.5
        # Distance 5-20: cohesion 0.5-0.0
        if avg_distance <= 5:
            cohesion = 1.0 - (avg_distance / 10)
        else:
            cohesion = max(0.0, 0.5 - (avg_distance - 5) / 30)

        return max(0.0, min(1.0, cohesion))

    def update_effectiveness(
        self,
        coalition: Coalition,
        recent_successes: int,
        recent_failures: int,
    ) -> float:
        """Update coalition effectiveness based on recent outcomes.

        Args:
            coalition: Coalition to update
            recent_successes: Number of successful coordinated actions
            recent_failures: Number of failed coordinated actions

        Returns:
            Updated effectiveness score (0.0-1.0)
        """
        total = recent_successes + recent_failures
        if total == 0:
            return coalition.effectiveness

        success_rate = recent_successes / total

        # Smooth update: blend with previous effectiveness
        new_effectiveness = 0.7 * coalition.effectiveness + 0.3 * success_rate

        coalition.effectiveness = max(0.0, min(1.0, new_effectiveness))

        return coalition.effectiveness
