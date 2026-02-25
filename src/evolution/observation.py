"""Behavioral observation system for cultural transmission.

This module implements observation recording for agents to learn from each other's
actions. Observers record what actors do, the context, and the outcome fitness delta.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

# Re-export ContextTag from shared module for backward compatibility
from src.awareness.context import ContextTag  # noqa: F401


@dataclass(frozen=True)
class BehaviorObservation:
    """Record of one agent observing another's action.

    Captures what was done, the context it happened in, and whether it succeeded.
    Used for cultural transmission — observers learn from actors' outcomes.
    """

    observer_id: str
    actor_id: str
    action_type: str  # ActionType.value
    context_tag: str
    outcome_success: bool
    outcome_fitness_delta: float  # sum of needs_delta values
    tick: int
    actor_position: tuple[int, int]


class ObservationMemory:
    """Rolling buffer of behavioral observations for one agent.

    Stores what this agent has seen other agents do and how it turned out.
    Supports retrieval by actor, context, and recency for imitation learning.
    """

    MAX_OBSERVATIONS = 500
    MAX_PER_ACTOR = 100

    def __init__(self):
        self._observations: deque[BehaviorObservation] = deque(maxlen=self.MAX_OBSERVATIONS)
        self._by_actor: dict[str, list[BehaviorObservation]] = {}

    def record(self, obs: BehaviorObservation) -> None:
        """Append an observation to memory.

        Args:
            obs: The behavior observation to record.
        """
        self._observations.append(obs)

        # Update actor index
        if obs.actor_id not in self._by_actor:
            self._by_actor[obs.actor_id] = []
        self._by_actor[obs.actor_id].append(obs)

        # Trim per-actor list if needed
        if len(self._by_actor[obs.actor_id]) > self.MAX_PER_ACTOR:
            self._by_actor[obs.actor_id] = self._by_actor[obs.actor_id][-self.MAX_PER_ACTOR :]

    def recent(self, n: int = 50) -> list[BehaviorObservation]:
        """Retrieve the last N observations.

        Args:
            n: Number of recent observations to retrieve.

        Returns:
            List of up to N most recent observations.
        """
        return list(self._observations)[-n:]

    def observations_of(self, actor_id: str) -> list[BehaviorObservation]:
        """Retrieve all observations of a specific actor.

        Args:
            actor_id: The actor whose observations to retrieve.

        Returns:
            List of all observations where this actor was observed.
        """
        return self._by_actor.get(actor_id, [])

    def observations_in_context(self, context_tag: str, n: int = 50) -> list[BehaviorObservation]:
        """Retrieve recent observations matching a context tag.

        Args:
            context_tag: The context to filter by.
            n: Maximum number of observations to return.

        Returns:
            List of up to N recent observations in the specified context.
        """
        matching = [obs for obs in self._observations if obs.context_tag == context_tag]
        return matching[-n:]

    @property
    def count(self) -> int:
        """Total number of observations currently stored."""
        return len(self._observations)
