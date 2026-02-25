"""Hierarchical planner: generates and executes multi-step plans."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.planning.goal import Goal, GoalPriority, GoalStatus
from src.planning.plan import Plan, PlanStep

if TYPE_CHECKING:
    from src.awareness.types import Reflection, Sensation


class HierarchicalPlanner:
    """Plans and executes multi-step goals for an agent.

    The planner:
    1. Proposes goals based on current state (needs, opportunities, threats)
    2. Creates multi-step plans to achieve goals
    3. Executes plans step-by-step
    4. Monitors for interrupts and triggers replanning when needed
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._goals: dict[str, Goal] = {}
        self._plans: dict[str, Plan] = {}
        self._active_goal_id: str | None = None
        self._interrupt_stack: list[str] = []  # Stack of interrupted goal IDs
        self._failed_steps_count: int = 0

    def propose_goals(
        self,
        sensation: Sensation,
        reflection: Reflection,
        traits: dict[str, float],
    ) -> list[Goal]:
        """Generate candidate goals based on current state.

        Goals are prioritized by urgency:
        - SURVIVAL: health critical
        - URGENT_NEED: any need critically low
        - NEED: important needs below threshold
        - TACTICAL: resource gathering
        - STRATEGIC: territory/alliances
        - OPPORTUNISTIC: exploration
        """
        goals = []
        tick = sensation.tick
        needs = sensation.own_needs

        # SURVIVAL: Critical health
        health = needs.get("health", 100)
        if health < 20:
            goal = Goal.create(
                description="restore_health",
                priority=GoalPriority.SURVIVAL,
                created_tick=tick,
                deadline_ticks=20,
            )
            goals.append(goal)

        # URGENT_NEED: Any need critically low
        for need_name in ["hunger", "thirst", "energy"]:
            need_value = needs.get(need_name, 100)
            if need_value < 15:
                goal = Goal.create(
                    description=f"satisfy_{need_name}",
                    priority=GoalPriority.URGENT_NEED,
                    created_tick=tick,
                    deadline_ticks=10,
                )
                goals.append(goal)

        # NEED: Most urgent need below threshold
        survival_needs = {k: v for k, v in needs.items() if k in ["hunger", "thirst", "energy"]}
        if survival_needs:
            most_urgent: str = min(survival_needs, key=lambda k: survival_needs[k])
            if survival_needs[most_urgent] < 40:
                goal = Goal.create(
                    description=f"satisfy_{most_urgent}",
                    priority=GoalPriority.NEED,
                    created_tick=tick,
                    deadline_ticks=30,
                )
                goals.append(goal)

        # TACTICAL: Low inventory
        total_inventory = sum(sensation.own_inventory.values())
        if total_inventory < 3:
            goal = Goal.create(
                description="gather_resources",
                priority=GoalPriority.TACTICAL,
                created_tick=tick,
            )
            goals.append(goal)

        # STRATEGIC: Territory establishment (high aggression)
        aggression = traits.get("aggression", 0.5)
        if aggression > 0.6:
            goal = Goal.create(
                description="establish_territory",
                priority=GoalPriority.STRATEGIC,
                created_tick=tick,
            )
            goals.append(goal)

        # STRATEGIC: Build alliances (high sociability, few nearby agents)
        sociability = traits.get("sociability", 0.5)
        visible_agents = len(sensation.visible_agents)
        if sociability > 0.6 and visible_agents < 2:
            goal = Goal.create(
                description="build_alliances",
                priority=GoalPriority.STRATEGIC,
                created_tick=tick,
            )
            goals.append(goal)

        # OPPORTUNISTIC: Exploration (high curiosity + all needs satisfied)
        curiosity = traits.get("curiosity", 0.5)
        all_needs_ok = all(v > 50 for k, v in needs.items() if k in ["hunger", "thirst", "energy"])
        if curiosity > 0.7 and all_needs_ok:
            goal = Goal.create(
                description="explore_new_area",
                priority=GoalPriority.OPPORTUNISTIC,
                created_tick=tick,
            )
            goals.append(goal)

        return goals

    # ------------------------------------------------------------------
    # Plan creation
    # ------------------------------------------------------------------

    def create_plan(
        self,
        goal: Goal,
        sensation: Sensation,
        reflection: Reflection,
    ) -> Plan:
        """Decompose a goal into a sequence of PlanSteps.

        Uses goal description to dispatch to specialised plan builders.
        """
        builders = {
            "satisfy_hunger": self._plan_satisfy_hunger,
            "satisfy_thirst": self._plan_satisfy_thirst,
            "satisfy_energy": self._plan_satisfy_energy,
            "restore_health": self._plan_restore_health,
            "gather_resources": self._plan_gather_resources,
            "establish_territory": self._plan_establish_territory,
            "build_alliances": self._plan_build_alliances,
            "explore_new_area": self._plan_explore,
        }

        builder = builders.get(goal.description, self._plan_explore)
        steps = builder(sensation, reflection)

        plan = Plan(
            goal_id=goal.id,
            steps=steps,
            created_tick=sensation.tick,
        )
        return plan

    # ------------------------------------------------------------------
    # Core execution loop
    # ------------------------------------------------------------------

    def get_next_action(
        self,
        sensation: Sensation,
        reflection: Reflection,
        traits: dict[str, float],
        tick: int,
    ) -> tuple[str, Plan | None]:
        """Main entry-point: decide what to do this tick.

        Returns ``(action_type_str, plan)`` where *plan* is the active plan
        or ``None`` when no planning is needed (fall-back to reactive).
        """
        # 1. Check for high-priority interrupts
        interrupt_goal = self._check_interrupts(sensation, traits, tick)
        if interrupt_goal is not None:
            # Push current goal onto the interrupt stack
            if self._active_goal_id:
                self._interrupt_stack.append(self._active_goal_id)
                old_goal = self._goals.get(self._active_goal_id)
                if old_goal:
                    old_goal.status = GoalStatus.INTERRUPTED
            # Activate the interrupt goal
            self._goals[interrupt_goal.id] = interrupt_goal
            interrupt_goal.status = GoalStatus.ACTIVE
            self._active_goal_id = interrupt_goal.id
            plan = self.create_plan(interrupt_goal, sensation, reflection)
            self._plans[interrupt_goal.id] = plan

        # 2. If no active goal, propose and select one
        if self._active_goal_id is None or self._active_goal_id not in self._goals:
            # Try to pop an interrupted goal first
            if self._interrupt_stack:
                resumed_id = self._interrupt_stack.pop()
                if resumed_id in self._goals:
                    resumed_goal = self._goals[resumed_id]
                    resumed_goal.status = GoalStatus.ACTIVE
                    self._active_goal_id = resumed_id
                else:
                    self._active_goal_id = None

            if self._active_goal_id is None:
                proposed = self.propose_goals(sensation, reflection, traits)
                if not proposed:
                    return ("wait", None)
                best = self._select_best_goal(proposed)
                self._goals[best.id] = best
                best.status = GoalStatus.ACTIVE
                self._active_goal_id = best.id
                plan = self.create_plan(best, sensation, reflection)
                self._plans[best.id] = plan

        # 3. Get active plan
        active_plan: Plan | None = self._plans.get(self._active_goal_id)  # type: ignore[arg-type]
        goal = self._goals.get(self._active_goal_id)  # type: ignore[arg-type]

        if active_plan is None or goal is None:
            self._active_goal_id = None
            return ("wait", None)

        # 4. Check if plan is completed
        if active_plan.status == "completed":
            goal.status = GoalStatus.COMPLETED
            self._active_goal_id = None
            return ("wait", None)

        # 5. Check if we need to replan
        if self._should_replan(active_plan, goal, sensation):
            active_plan = self.create_plan(goal, sensation, reflection)
            active_plan.revision_count += 1
            self._plans[goal.id] = active_plan
            self._failed_steps_count = 0

        # 6. Return current step's action
        step = active_plan.current_step
        if step is None:
            goal.status = GoalStatus.COMPLETED
            self._active_goal_id = None
            return ("wait", None)

        return (step.action_type, active_plan)

    def report_action_result(self, action_succeeded: bool, tick: int) -> None:
        """Report the outcome of the last action to advance or retry the plan."""
        if self._active_goal_id is None:
            return

        plan = self._plans.get(self._active_goal_id)
        goal = self._goals.get(self._active_goal_id)
        if plan is None or goal is None:
            return

        plan.mark_step_executed(tick, action_succeeded)

        if action_succeeded:
            plan.advance()
            self._failed_steps_count = 0
            goal.progress = plan.progress

            if plan.status == "completed":
                goal.status = GoalStatus.COMPLETED
                self._active_goal_id = None
        else:
            goal.attempts += 1
            self._failed_steps_count += 1

            # Too many failures → fail the goal
            if goal.attempts >= goal.max_attempts:
                goal.status = GoalStatus.FAILED
                plan.status = "failed"
                self._active_goal_id = None

    # ------------------------------------------------------------------
    # Interrupt & replan logic
    # ------------------------------------------------------------------

    def _check_interrupts(
        self,
        sensation: Sensation,
        traits: dict[str, float],
        tick: int,
    ) -> Goal | None:
        """Detect if a survival-level interrupt should pre-empt the active goal."""
        needs = sensation.own_needs
        health = needs.get("health", 100)

        # Only interrupt for SURVIVAL-level emergencies
        if health < 20:
            # Don't interrupt if we're already pursuing restore_health
            active_goal = self._goals.get(self._active_goal_id) if self._active_goal_id else None
            if active_goal and active_goal.description == "restore_health":
                return None
            return Goal.create(
                description="restore_health",
                priority=GoalPriority.SURVIVAL,
                created_tick=tick,
                deadline_ticks=20,
            )

        # Interrupt for critically low needs
        for need_name in ["hunger", "thirst", "energy"]:
            if needs.get(need_name, 100) < 10:
                active_goal = (
                    self._goals.get(self._active_goal_id) if self._active_goal_id else None
                )
                if active_goal and active_goal.description == f"satisfy_{need_name}":
                    return None
                return Goal.create(
                    description=f"satisfy_{need_name}",
                    priority=GoalPriority.URGENT_NEED,
                    created_tick=tick,
                    deadline_ticks=10,
                )

        return None

    def _should_replan(self, plan: Plan, goal: Goal, sensation: Sensation) -> bool:
        """Determine if the current plan needs revision."""
        # Replan after 3 consecutive failures
        if self._failed_steps_count >= 3:
            return True

        # Replan if deadline is approaching and progress is poor
        if goal.deadline_tick is not None:
            remaining = goal.deadline_tick - sensation.tick
            if remaining < 5 and plan.progress < 0.5:
                return True

        return False

    def _select_best_goal(self, candidates: list[Goal]) -> Goal:
        """Pick the highest-priority goal (lowest GoalPriority value)."""
        return min(candidates, key=lambda g: g.priority.value)

    # ------------------------------------------------------------------
    # Plan builders — one per goal description
    # ------------------------------------------------------------------

    def _plan_satisfy_hunger(self, sensation: Sensation, reflection: Reflection) -> list[PlanStep]:
        """Plan steps: find food → gather → eat."""
        steps: list[PlanStep] = []

        # If we already have berries, just eat
        if sensation.own_inventory.get("berries", 0) > 0:
            steps.append(PlanStep(action_type="eat"))
            return steps

        # Find nearest berry bush
        target = self._find_nearest_resource(sensation, "berry_bush")
        if target:
            steps.append(PlanStep(action_type="move", target_position=target))
            steps.append(PlanStep(action_type="gather", target="berry_bush"))
        else:
            # No known food — explore
            steps.append(
                PlanStep(action_type="move", target_position=self._explore_target(sensation))
            )

        steps.append(PlanStep(action_type="eat"))
        return steps

    def _plan_satisfy_thirst(self, sensation: Sensation, reflection: Reflection) -> list[PlanStep]:
        """Plan steps: find water → gather → drink."""
        steps: list[PlanStep] = []

        if sensation.own_inventory.get("water", 0) > 0:
            steps.append(PlanStep(action_type="drink"))
            return steps

        target = self._find_nearest_resource(sensation, "water_source")
        if target:
            steps.append(PlanStep(action_type="move", target_position=target))
            steps.append(PlanStep(action_type="gather", target="water_source"))
        else:
            steps.append(
                PlanStep(action_type="move", target_position=self._explore_target(sensation))
            )

        steps.append(PlanStep(action_type="drink"))
        return steps

    def _plan_satisfy_energy(self, sensation: Sensation, reflection: Reflection) -> list[PlanStep]:
        """Plan steps: rest."""
        return [PlanStep(action_type="rest")]

    def _plan_restore_health(self, sensation: Sensation, reflection: Reflection) -> list[PlanStep]:
        """Plan steps: rest repeatedly to recover health."""
        return [
            PlanStep(action_type="rest"),
            PlanStep(action_type="rest"),
            PlanStep(action_type="rest"),
        ]

    def _plan_gather_resources(
        self, sensation: Sensation, reflection: Reflection
    ) -> list[PlanStep]:
        """Plan steps: find any resource → gather multiple."""
        steps: list[PlanStep] = []

        # Try berries first, then water
        for kind in ["berry_bush", "water_source"]:
            target = self._find_nearest_resource(sensation, kind)
            if target:
                steps.append(PlanStep(action_type="move", target_position=target))
                steps.append(PlanStep(action_type="gather", target=kind))
                steps.append(PlanStep(action_type="gather", target=kind))
                return steps

        # Nothing found — explore
        steps.append(PlanStep(action_type="move", target_position=self._explore_target(sensation)))
        return steps

    def _plan_establish_territory(
        self, sensation: Sensation, reflection: Reflection
    ) -> list[PlanStep]:
        """Plan steps: patrol area, threaten intruders."""
        steps: list[PlanStep] = []
        # Move around current position
        x, y = sensation.own_position
        steps.append(PlanStep(action_type="move", target_position=(x + 2, y)))
        steps.append(PlanStep(action_type="move", target_position=(x, y + 2)))
        steps.append(PlanStep(action_type="move", target_position=(x - 2, y)))
        steps.append(PlanStep(action_type="move", target_position=(x, y - 2)))

        # Threaten any visible agents
        for agent_summary in sensation.visible_agents:
            steps.append(
                PlanStep(
                    action_type="send_message",
                    target_agent_id=str(agent_summary.agent_id),
                )
            )
        return steps

    def _plan_build_alliances(self, sensation: Sensation, reflection: Reflection) -> list[PlanStep]:
        """Plan steps: find agents → send friendly messages."""
        steps: list[PlanStep] = []

        if sensation.visible_agents:
            for agent_summary in sensation.visible_agents[:2]:
                steps.append(
                    PlanStep(
                        action_type="send_message",
                        target_agent_id=str(agent_summary.agent_id),
                    )
                )
        else:
            # Move to find agents
            steps.append(
                PlanStep(action_type="move", target_position=self._explore_target(sensation))
            )
        return steps

    def _plan_explore(self, sensation: Sensation, reflection: Reflection) -> list[PlanStep]:
        """Plan steps: move in a spiral pattern to discover territory."""
        steps: list[PlanStep] = []
        x, y = sensation.own_position
        # Simple exploration: move outward in 4 directions
        offsets = [(3, 0), (0, 3), (-3, 0), (0, -3)]
        for dx, dy in offsets:
            steps.append(PlanStep(action_type="move", target_position=(x + dx, y + dy)))
        return steps

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_nearest_resource(
        self, sensation: Sensation, resource_kind: str
    ) -> tuple[int, int] | None:
        """Find the nearest tile containing the given resource kind."""
        best_pos: tuple[int, int] | None = None
        best_dist = float("inf")
        ox, oy = sensation.own_position

        for tile in sensation.visible_tiles:
            for kind, qty in tile.resources:
                if kind == resource_kind and qty > 0:
                    dist = abs(tile.x - ox) + abs(tile.y - oy)
                    if dist < best_dist:
                        best_dist = dist
                        best_pos = (tile.x, tile.y)

        return best_pos

    def _explore_target(self, sensation: Sensation) -> tuple[int, int]:
        """Pick a target position for exploration away from current pos."""
        x, y = sensation.own_position
        # Default: move east and south
        return (x + 3, y + 3)
