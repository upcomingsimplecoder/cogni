"""Goal representation with status, priority, and success conditions."""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field


class GoalStatus(enum.Enum):
    """Current state of a goal."""

    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class GoalPriority(enum.IntEnum):
    """Priority levels for goals (lower value = higher priority)."""

    SURVIVAL = 0
    URGENT_NEED = 1
    NEED = 2
    TACTICAL = 3
    STRATEGIC = 4
    OPPORTUNISTIC = 5


@dataclass
class Goal:
    """A goal the agent wants to achieve.

    Goals can be hierarchical (parent_goal_id, sub_goals) and have
    success/failure conditions that determine when they're complete.
    """

    id: str
    description: str  # "satisfy_hunger", "establish_territory", etc.
    priority: GoalPriority
    status: GoalStatus = GoalStatus.PENDING
    success_condition: str | None = None
    failure_condition: str | None = None
    parent_goal_id: str | None = None
    sub_goals: list[str] = field(default_factory=list)
    created_tick: int = 0
    deadline_tick: int | None = None
    planned_actions: list[str] = field(default_factory=list)
    progress: float = 0.0  # 0.0 to 1.0
    attempts: int = 0
    max_attempts: int = 10

    @classmethod
    def create(
        cls,
        description: str,
        priority: GoalPriority,
        created_tick: int = 0,
        deadline_ticks: int | None = None,
    ) -> Goal:
        """Factory method to create a new goal with auto-generated ID."""
        goal_id = str(uuid.uuid4())[:8]
        deadline_tick = None
        if deadline_ticks is not None:
            deadline_tick = created_tick + deadline_ticks

        return cls(
            id=goal_id,
            description=description,
            priority=priority,
            created_tick=created_tick,
            deadline_tick=deadline_tick,
        )
