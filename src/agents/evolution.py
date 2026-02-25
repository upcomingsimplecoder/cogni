"""Trait evolution based on experience outcomes."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.identity import PersonalityTraits


@dataclass
class TraitNudgeRecord:
    """Record of a single trait nudge event."""

    tick: int
    agent_id: str
    event_type: str
    was_positive: bool
    traits_affected: list[tuple[str, float, float]]  # (trait_name, old_value, new_value)
    viewed_at: int | None = None  # tick when scored by effectiveness engine


# Maps event types to which traits they affect
SIGNAL_MAP: dict[str, list[str]] = {
    "shared_resource": ["cooperation_tendency", "resource_sharing", "sociability"],
    "explored_new_area": ["curiosity"],
    "attacked_agent": ["aggression", "risk_tolerance"],
    "received_help": ["sociability", "cooperation_tendency"],
    "trade_completed": ["cooperation_tendency", "sociability"],
    "survived_low_health": ["risk_tolerance"],
    "was_attacked": ["aggression", "risk_tolerance"],
    "found_resource": ["curiosity", "resource_sharing"],
    "social_interaction": ["sociability", "cooperation_tendency"],
    "sent_message": ["sociability"],
    "received_message": ["sociability"],
    "joined_coalition": ["cooperation_tendency", "sociability"],
}


class TraitEvolution:
    """Evolves personality traits based on experience outcomes.

    Traits drift slowly — ~50 consistent signals to move from 0.5 to 1.0.
    """

    def __init__(self, learning_rate: float = 0.01, history_max: int = 500):
        self.learning_rate = learning_rate
        self._history: deque[TraitNudgeRecord] = deque(maxlen=history_max)

    def process_outcome(
        self,
        traits: PersonalityTraits,
        event_type: str,
        was_positive: bool,
        agent_id: str = "",
        tick: int = 0,
    ) -> list[tuple[str, float]]:
        """Shift traits based on an interaction outcome.

        Returns list of (trait_name, new_value) for tracking.
        """
        affected = SIGNAL_MAP.get(event_type, [])
        if not affected:
            return []

        direction = 1.0 if was_positive else -1.0
        delta = self.learning_rate * direction

        changes = []
        traits_affected = []
        for trait_name in affected:
            old_val = getattr(traits, trait_name)
            traits.shift_trait(trait_name, delta)
            new_val = getattr(traits, trait_name)
            changes.append((trait_name, new_val))
            traits_affected.append((trait_name, old_val, new_val))

        # Record the nudge event
        record = TraitNudgeRecord(
            tick=tick,
            agent_id=agent_id,
            event_type=event_type,
            was_positive=was_positive,
            traits_affected=traits_affected,
        )
        self._history.append(record)

        return changes

    def export_history(self) -> list[dict]:
        """Export nudge history for serialization."""
        return [
            {
                "tick": r.tick,
                "agent_id": r.agent_id,
                "event_type": r.event_type,
                "was_positive": r.was_positive,
                "traits_affected": r.traits_affected,
                "viewed_at": r.viewed_at,
            }
            for r in self._history
        ]

    def import_history(self, data: list[dict]) -> None:
        """Import nudge history from serialized data."""
        self._history.clear()
        for entry in data:
            record = TraitNudgeRecord(
                tick=entry["tick"],
                agent_id=entry["agent_id"],
                event_type=entry["event_type"],
                was_positive=entry["was_positive"],
                traits_affected=[tuple(t) for t in entry["traits_affected"]],
                viewed_at=entry.get("viewed_at"),
            )
            self._history.append(record)

    @property
    def history(self) -> list[TraitNudgeRecord]:
        """Get nudge history."""
        return list(self._history)
