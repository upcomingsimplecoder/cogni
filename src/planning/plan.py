"""Plan representation: sequence of steps to achieve a goal."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PlanStep:
    """A single step in a plan."""

    action_type: str  # "move", "gather", "eat", "rest", etc.
    target: str | None = None  # resource kind, item name
    target_position: tuple[int, int] | None = None
    target_agent_id: str | None = None
    executed: bool = False
    succeeded: bool = False
    tick_executed: int | None = None
    requires: dict | None = None  # Preconditions {"inventory": {"berries": 1}}


@dataclass
class Plan:
    """A plan to achieve a goal: ordered sequence of steps."""

    goal_id: str
    steps: list[PlanStep] = field(default_factory=list)
    created_tick: int = 0
    current_step_index: int = 0
    status: str = "active"  # "active", "completed", "failed"
    revision_count: int = 0

    @property
    def current_step(self) -> PlanStep | None:
        """Get the current step to execute."""
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    @property
    def progress(self) -> float:
        """Fraction of plan completed (0.0 to 1.0)."""
        if not self.steps:
            return 1.0
        return self.current_step_index / len(self.steps)

    def advance(self) -> None:
        """Move to next step."""
        self.current_step_index += 1
        if self.current_step_index >= len(self.steps):
            self.status = "completed"

    def mark_step_executed(self, tick: int, succeeded: bool) -> None:
        """Mark current step as executed."""
        step = self.current_step
        if step:
            step.executed = True
            step.succeeded = succeeded
            step.tick_executed = tick
