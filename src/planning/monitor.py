"""Plan execution monitoring and metrics tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.planning.goal import Goal
    from src.planning.plan import Plan


class PlanMonitor:
    """Tracks plan execution metrics for emergence detection and trajectory recording."""

    def __init__(self):
        self._plan_history: list[dict] = []  # Completed/failed plans
        self._replanning_count: int = 0
        self._interrupt_count: int = 0
        self._completed_goals: int = 0
        self._failed_goals: int = 0

    def record_plan_outcome(self, plan: Plan, goal: Goal) -> None:
        """Record completed or failed plan for analysis."""
        self._plan_history.append(
            {
                "goal_id": goal.id,
                "goal_description": goal.description,
                "goal_priority": goal.priority.name,
                "goal_status": goal.status.name,
                "plan_steps": len(plan.steps),
                "plan_progress": plan.progress,
                "plan_revisions": plan.revision_count,
                "plan_status": plan.status,
            }
        )

        if goal.status.name == "COMPLETED":
            self._completed_goals += 1
        elif goal.status.name == "FAILED":
            self._failed_goals += 1

    def record_interrupt(self, interrupted_goal: Goal, new_goal: Goal) -> None:
        """Record when a plan was interrupted."""
        self._interrupt_count += 1

    def record_replan(self, old_plan: Plan, new_plan: Plan, reason: str) -> None:
        """Record when replanning occurred."""
        self._replanning_count += 1

    @property
    def planning_stats(self) -> dict:
        """Summary stats for trajectory recording."""
        total_goals = self._completed_goals + self._failed_goals
        completion_rate = self._completed_goals / total_goals if total_goals > 0 else 0.0

        return {
            "total_plans": len(self._plan_history),
            "completion_rate": completion_rate,
            "replan_count": self._replanning_count,
            "interrupt_count": self._interrupt_count,
            "completed_goals": self._completed_goals,
            "failed_goals": self._failed_goals,
        }
