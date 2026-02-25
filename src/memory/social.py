"""Social memory: trust-based relationship tracking between agents."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Relationship:
    """One agent's perceived relationship with another.

    Trust builds slowly and drops quickly (asymmetric).
    """

    other_agent_id: object  # AgentID at runtime
    trust: float = 0.5  # 0=enemy, 1=trusted ally
    interaction_count: int = 0
    last_interaction_tick: int = 0
    net_resources_given: int = 0  # Positive = we gave more
    was_attacked_by: bool = False
    was_helped_by: bool = False

    def update_trust(self, event: str, positive: bool, tick: int = 0) -> None:
        """Update trust based on interaction.

        Positive events: +0.05 trust
        Negative events: -0.15 trust (asymmetric — trust is hard to build, easy to break)
        """
        delta = 0.05 if positive else -0.15
        self.trust = max(0.0, min(1.0, self.trust + delta))
        self.interaction_count += 1
        self.last_interaction_tick = tick

        if event == "attacked":
            self.was_attacked_by = True
        elif event in ("helped", "gave_resource", "shared_info"):
            self.was_helped_by = True

    def update_resource_flow(self, amount: int) -> None:
        """Track resource flow. Positive = we gave, negative = they gave."""
        self.net_resources_given += amount


class SocialMemory:
    """Tracks relationships with all known agents."""

    def __init__(self):
        self._relationships: dict[object, Relationship] = {}

    def get_or_create(self, other_id: object) -> Relationship:
        """Get existing or create new relationship entry."""
        if other_id not in self._relationships:
            self._relationships[other_id] = Relationship(other_agent_id=other_id)
        return self._relationships[other_id]

    def get(self, other_id: object) -> Relationship | None:
        """Get relationship if it exists."""
        return self._relationships.get(other_id)

    def most_trusted(self) -> Relationship | None:
        """Agent we trust most."""
        if not self._relationships:
            return None
        return max(self._relationships.values(), key=lambda r: r.trust)

    def least_trusted(self) -> Relationship | None:
        """Agent we trust least."""
        if not self._relationships:
            return None
        return min(self._relationships.values(), key=lambda r: r.trust)

    def allies(self, threshold: float = 0.7) -> list[Relationship]:
        """Agents with trust above threshold."""
        return [r for r in self._relationships.values() if r.trust >= threshold]

    def enemies(self, threshold: float = 0.3) -> list[Relationship]:
        """Agents with trust below threshold."""
        return [r for r in self._relationships.values() if r.trust <= threshold]

    def known_agents(self) -> list[object]:
        """All agent IDs we have relationships with."""
        return list(self._relationships.keys())

    @property
    def relationship_count(self) -> int:
        return len(self._relationships)
