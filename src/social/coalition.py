"""Coalition data structures and lifecycle management.

A Coalition represents a group of agents with shared goals and role assignments.
The CoalitionManager handles creation, membership changes, and dissolution.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class Coalition:
    """A group of agents working toward a shared goal.

    Attributes:
        id: Unique 8-character coalition identifier
        name: Human-readable coalition name
        leader_id: Agent ID of the coalition leader
        members: Set of agent IDs in the coalition
        shared_goal: Optional shared objective (e.g., "hunt", "defend_territory")
        formation_tick: Tick when coalition was formed
        roles: Mapping from agent_id to role name
        cohesion: Measure of group stability (0.0-1.0)
        effectiveness: Measure of goal achievement (0.0-1.0)
    """

    id: str
    name: str
    leader_id: str
    members: set[str] = field(default_factory=set)
    shared_goal: str | None = None
    formation_tick: int = 0
    roles: dict[str, str] = field(default_factory=dict)
    cohesion: float = 1.0
    effectiveness: float = 0.0

    @property
    def size(self) -> int:
        """Number of members in the coalition."""
        return len(self.members)

    def add_member(self, agent_id: str) -> None:
        """Add an agent to the coalition."""
        self.members.add(agent_id)

    def remove_member(self, agent_id: str) -> None:
        """Remove an agent from the coalition."""
        self.members.discard(agent_id)
        if agent_id in self.roles:
            del self.roles[agent_id]

    def assign_role(self, agent_id: str, role: str) -> None:
        """Assign a role to a coalition member."""
        if agent_id in self.members:
            self.roles[agent_id] = role

    def get_role(self, agent_id: str) -> str | None:
        """Get the role of a coalition member."""
        return self.roles.get(agent_id)


class CoalitionManager:
    """Manages all active coalitions and membership.

    Responsibilities:
    - Track all active coalitions
    - Handle coalition proposal, acceptance, and rejection
    - Manage agent membership across coalitions
    - Record coalition history
    - Handle coalition dissolution
    """

    def __init__(self):
        self._coalitions: dict[str, Coalition] = {}
        self._agent_coalition: dict[str, str] = {}  # agent_id -> coalition_id
        self._pending: dict[str, dict] = {}  # coalition_id -> pending proposal data
        self._history: list[dict] = []  # Historical record of all coalitions

    def propose(
        self, proposer_id: str, target_ids: list[str], goal: str, name: str | None = None
    ) -> str:
        """Create a coalition proposal from proposer to targets.

        Args:
            proposer_id: Agent proposing the coalition
            target_ids: List of agent IDs to invite
            goal: Shared goal for the coalition
            name: Optional name (auto-generated if None)

        Returns:
            Coalition ID for the pending proposal
        """
        coalition_id = uuid.uuid4().hex[:8]
        coalition_name = name or f"Coalition-{coalition_id}"

        # Create pending proposal
        self._pending[coalition_id] = {
            "proposer_id": proposer_id,
            "target_ids": set(target_ids),
            "accepted": {proposer_id},  # Proposer auto-accepts
            "goal": goal,
            "name": coalition_name,
        }

        return coalition_id

    def accept(self, coalition_id: str, agent_id: str) -> bool:
        """Agent accepts a coalition proposal.

        Args:
            coalition_id: ID of the coalition to join
            agent_id: Agent accepting the proposal

        Returns:
            True if accepted successfully, False if invalid
        """
        if coalition_id not in self._pending:
            return False

        proposal = self._pending[coalition_id]

        # Verify agent was invited
        if agent_id not in proposal["target_ids"] and agent_id != proposal["proposer_id"]:
            return False

        # Accept the proposal
        proposal["accepted"].add(agent_id)

        # Check if all targets have accepted
        all_invited = proposal["target_ids"] | {proposal["proposer_id"]}
        if proposal["accepted"] == all_invited:
            # Form the coalition
            self._form_coalition(coalition_id, proposal)

        return True

    def reject(self, coalition_id: str, agent_id: str) -> bool:
        """Agent rejects a coalition proposal.

        Args:
            coalition_id: ID of the coalition to reject
            agent_id: Agent rejecting the proposal

        Returns:
            True if rejected successfully, False if invalid
        """
        if coalition_id not in self._pending:
            return False

        proposal = self._pending[coalition_id]

        # Verify agent was invited
        if agent_id not in proposal["target_ids"] and agent_id != proposal["proposer_id"]:
            return False

        # Rejection dissolves the pending proposal
        del self._pending[coalition_id]

        self._history.append(
            {
                "coalition_id": coalition_id,
                "event": "rejected",
                "agent_id": agent_id,
            }
        )

        return True

    def _form_coalition(self, coalition_id: str, proposal: dict) -> None:
        """Form a coalition from an accepted proposal.

        Args:
            coalition_id: Coalition ID
            proposal: Proposal data dict
        """
        from src.simulation.engine import SimulationEngine

        current_tick = getattr(SimulationEngine, "_current_tick", 0)

        coalition = Coalition(
            id=coalition_id,
            name=proposal["name"],
            leader_id=proposal["proposer_id"],
            members=proposal["accepted"].copy(),
            shared_goal=proposal["goal"],
            formation_tick=current_tick,
        )

        self._coalitions[coalition_id] = coalition

        # Update agent -> coalition mapping
        for agent_id in coalition.members:
            self._agent_coalition[agent_id] = coalition_id

        # Remove from pending
        del self._pending[coalition_id]

        self._history.append(
            {
                "coalition_id": coalition_id,
                "event": "formed",
                "tick": current_tick,
                "members": list(coalition.members),
                "goal": coalition.shared_goal,
            }
        )

    def leave(self, coalition_id: str, agent_id: str) -> bool:
        """Agent voluntarily leaves a coalition.

        Args:
            coalition_id: Coalition to leave
            agent_id: Agent leaving

        Returns:
            True if left successfully, False if invalid
        """
        if coalition_id not in self._coalitions:
            return False

        coalition = self._coalitions[coalition_id]

        if agent_id not in coalition.members:
            return False

        coalition.remove_member(agent_id)

        if agent_id in self._agent_coalition:
            del self._agent_coalition[agent_id]

        self._history.append(
            {
                "coalition_id": coalition_id,
                "event": "member_left",
                "agent_id": agent_id,
            }
        )

        # Dissolve if too small or leader left
        if coalition.size < 2 or agent_id == coalition.leader_id:
            self.dissolve(
                coalition_id, reason="insufficient_members" if coalition.size < 2 else "leader_left"
            )

        return True

    def dissolve(self, coalition_id: str, reason: str) -> None:
        """Dissolve a coalition.

        Args:
            coalition_id: Coalition to dissolve
            reason: Reason for dissolution (for history tracking)
        """
        if coalition_id not in self._coalitions:
            return

        coalition = self._coalitions[coalition_id]

        # Remove all agent mappings
        for agent_id in list(coalition.members):
            if agent_id in self._agent_coalition:
                del self._agent_coalition[agent_id]

        self._history.append(
            {
                "coalition_id": coalition_id,
                "event": "dissolved",
                "reason": reason,
                "final_members": list(coalition.members),
            }
        )

        del self._coalitions[coalition_id]

    def get_coalition(self, agent_id: str) -> Coalition | None:
        """Get the coalition an agent belongs to.

        Args:
            agent_id: Agent to query

        Returns:
            Coalition object or None if not in any coalition
        """
        coalition_id = self._agent_coalition.get(agent_id)
        if coalition_id is None:
            return None
        return self._coalitions.get(coalition_id)

    def get_coalition_by_id(self, coalition_id: str) -> Coalition | None:
        """Get a coalition by its ID.

        Args:
            coalition_id: Coalition ID to query

        Returns:
            Coalition object or None if not found
        """
        return self._coalitions.get(coalition_id)

    def all_coalitions(self) -> list[Coalition]:
        """Get all active coalitions.

        Returns:
            List of all Coalition objects
        """
        return list(self._coalitions.values())

    def pending_proposals(self) -> dict[str, dict]:
        """Get all pending coalition proposals.

        Returns:
            Dict mapping coalition_id to proposal data
        """
        return self._pending.copy()

    def coalition_history(self) -> list[dict]:
        """Get the complete coalition history.

        Returns:
            List of historical events
        """
        return self._history.copy()
