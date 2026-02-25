"""Episodic memory: rolling window of action outcomes and interactions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class Episode:
    """A remembered event with outcome evaluation."""

    tick: int
    action_type: str
    target: str | None
    success: bool
    needs_delta: dict[str, float] = field(default_factory=dict)
    location: tuple[int, int] = (0, 0)
    involved_agent: object | None = None  # AgentID at runtime


@dataclass
class InteractionOutcome:
    """Result of a past interaction with another agent."""

    other_agent_id: object  # AgentID at runtime
    tick: int
    was_positive: bool
    interaction_type: str  # "trade", "shared_info", "attacked", "was_helped", "gave_resource"


class EpisodicMemory:
    """Rolling window of remembered episodes for one agent.

    Used by ReflectionModule to evaluate past decisions.
    """

    MAX_EPISODES = 200
    MAX_INTERACTIONS = 100

    def __init__(self):
        self._episodes: deque[Episode] = deque(maxlen=self.MAX_EPISODES)
        self._interaction_log: deque[InteractionOutcome] = deque(maxlen=self.MAX_INTERACTIONS)

    def record_episode(
        self,
        tick: int,
        action_type: str,
        target: str | None,
        success: bool,
        needs_delta: dict[str, float] | None = None,
        location: tuple[int, int] = (0, 0),
        involved_agent: object | None = None,
    ) -> None:
        """Record an action outcome as an episode."""
        self._episodes.append(
            Episode(
                tick=tick,
                action_type=action_type,
                target=target,
                success=success,
                needs_delta=needs_delta or {},
                location=location,
                involved_agent=involved_agent,
            )
        )

    def record_interaction(self, outcome: InteractionOutcome) -> None:
        """Record a social interaction outcome."""
        self._interaction_log.append(outcome)

    def recent_episodes(self, n: int = 10) -> list[Episode]:
        """Last N episodes."""
        return list(self._episodes)[-n:]

    def recent_interactions(self, n: int = 5) -> list[InteractionOutcome]:
        """Last N interaction outcomes."""
        return list(self._interaction_log)[-n:]

    def success_rate(self, action_type: str, window: int = 20) -> float:
        """Success rate for a specific action type in recent episodes.

        Returns 0.5 (neutral) if no data available.
        """
        recent = list(self._episodes)[-window:]
        relevant = [e for e in recent if e.action_type == action_type]
        if not relevant:
            return 0.5
        return sum(1 for e in relevant if e.success) / len(relevant)

    def episodes_at_location(self, location: tuple[int, int], radius: int = 2) -> list[Episode]:
        """Episodes that occurred near a location."""
        return [
            e
            for e in self._episodes
            if abs(e.location[0] - location[0]) <= radius
            and abs(e.location[1] - location[1]) <= radius
        ]

    def interactions_with(self, agent_id: object) -> list[InteractionOutcome]:
        """All interactions with a specific agent."""
        return [i for i in self._interaction_log if i.other_agent_id == agent_id]

    @property
    def episode_count(self) -> int:
        return len(self._episodes)

    @property
    def interaction_count(self) -> int:
        return len(self._interaction_log)
