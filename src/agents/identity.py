"""Agent identity: IDs, personality traits, and profiles."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field


class AgentID:
    """Opaque agent identifier wrapping a short UUID."""

    def __init__(self, value: str | None = None):
        self.value = value or str(uuid.uuid4())[:8]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AgentID) and self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)

    def __repr__(self) -> str:
        return f"AgentID({self.value})"

    def __str__(self) -> str:
        return self.value


@dataclass
class PersonalityTraits:
    """Personality bias vector. Each trait is 0.0-1.0.

    These bias decisions — they don't determine them.
    Initialized from archetype, evolved by experience.
    """

    cooperation_tendency: float = 0.5  # 0=selfish, 1=altruistic
    curiosity: float = 0.5  # 0=stay put, 1=explore
    risk_tolerance: float = 0.5  # 0=cautious, 1=bold
    resource_sharing: float = 0.5  # 0=hoard, 1=generous
    aggression: float = 0.5  # 0=pacifist, 1=aggressive
    sociability: float = 0.5  # 0=loner, 1=seeks groups

    def as_dict(self) -> dict[str, float]:
        """Return traits as a flat dictionary."""
        return {
            "cooperation_tendency": self.cooperation_tendency,
            "curiosity": self.curiosity,
            "risk_tolerance": self.risk_tolerance,
            "resource_sharing": self.resource_sharing,
            "aggression": self.aggression,
            "sociability": self.sociability,
        }

    def shift_trait(self, trait: str, delta: float) -> None:
        """Shift a trait by delta, clamping to [0.0, 1.0]."""
        current = getattr(self, trait)
        setattr(self, trait, max(0.0, min(1.0, current + delta)))

    def copy(self) -> PersonalityTraits:
        """Return a deep copy of these traits."""
        return PersonalityTraits(**self.as_dict())


@dataclass
class AgentProfile:
    """Complete agent identity: ID + name + personality + archetype label."""

    agent_id: AgentID
    name: str
    archetype: str  # "gatherer", "explorer", etc. — informational label
    traits: PersonalityTraits = field(default_factory=PersonalityTraits)
