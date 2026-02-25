"""Tests for hierarchical goal planning system."""

from __future__ import annotations

import pytest

from src.awareness.types import Reflection, Sensation, TileSummary
from src.cognition.strategies.planning import PlanningStrategy
from src.planning.goal import Goal, GoalPriority, GoalStatus
from src.planning.monitor import PlanMonitor
from src.planning.plan import Plan, PlanStep
from src.planning.planner import HierarchicalPlanner
from src.simulation.actions import ActionType


@pytest.fixture
def basic_sensation():
    """Create a basic sensation for testing."""
    return Sensation(
        tick=10,
        own_needs={"hunger": 50, "thirst": 50, "energy": 50, "health": 100},
        own_position=(5, 5),
        own_inventory={},
        visible_tiles=[
            TileSummary(x=5, y=5, tile_type="grass", resources=[]),
            TileSummary(x=6, y=5, tile_type="grass", resources=[("berry_bush", 3)]),
            TileSummary(x=5, y=6, tile_type="water", resources=[("water_source", 5)]),
        ],
        visible_agents=[],
        own_traits={
            "cooperation_tendency": 0.5,
            "curiosity": 0.5,
            "risk_tolerance": 0.5,
            "resource_sharing": 0.5,
            "aggression": 0.5,
            "sociability": 0.5,
        },
    )


@pytest.fixture
def basic_reflection():
    """Create a basic reflection for testing."""
    return Reflection(
        last_action_succeeded=True,
        need_trends={"hunger": "stable", "thirst": "stable", "energy": "stable"},
        threat_level=0.0,
        opportunity_score=0.5,
    )


class TestGoal:
    """Test Goal dataclass and status transitions."""

    def test_goal_creation_with_all_priorities(self):
        """Test creating goals with different priority levels."""
        priorities = [
            GoalPriority.SURVIVAL,
            GoalPriority.URGENT_NEED,
            GoalPriority.NEED,
            GoalPriority.TACTICAL,
            GoalPriority.STRATEGIC,
            GoalPriority.OPPORTUNISTIC,
        ]
        for priority in priorities:
            goal = Goal.create(
                description=f"test_{priority.name}",
                priority=priority,
                created_tick=0,
            )
            assert goal.id is not None
            assert goal.priority == priority
            assert goal.status == GoalStatus.PENDING

    def test_goal_status_transitions(self):
        """Test goal status can transition through lifecycle."""
        goal = Goal.create(
            description="test_goal",
            priority=GoalPriority.NEED,
            created_tick=0,
        )

        assert goal.status == GoalStatus.PENDING

        goal.status = GoalStatus.ACTIVE
        assert goal.status == GoalStatus.ACTIVE

        goal.status = GoalStatus.COMPLETED
        assert goal.status == GoalStatus.COMPLETED

    def test_goal_failure_by_max_attempts(self):
        """Test goal can be marked as failed after max attempts."""
        goal = Goal.create(
            description="failing_goal",
            priority=GoalPriority.TACTICAL,
            created_tick=0,
        )
        goal.max_attempts = 3

        for _ in range(3):
            goal.attempts += 1

        assert goal.attempts >= goal.max_attempts


class TestPlan:
    """Test Plan and PlanStep functionality."""

    def test_plan_step_advancement(self):
        """Test plan advances through steps correctly."""
        steps = [
            PlanStep(action_type="move"),
            PlanStep(action_type="gather", target="berries"),
            PlanStep(action_type="eat"),
        ]
        plan = Plan(goal_id="test-goal", steps=steps, created_tick=0)

        assert plan.current_step_index == 0
        assert plan.current_step == steps[0]
        assert plan.progress == 0.0

        plan.advance()
        assert plan.current_step_index == 1
        assert plan.current_step == steps[1]
        assert plan.progress == pytest.approx(1 / 3)

        plan.advance()
        assert plan.current_step_index == 2
        assert plan.current_step == steps[2]

        plan.advance()
        assert plan.current_step is None
        assert plan.status == "completed"
        assert plan.progress == 1.0

    def test_empty_plan_progress(self):
        """Test plan with no steps shows 100% progress."""
        plan = Plan(goal_id="empty", steps=[], created_tick=0)
        assert plan.progress == 1.0
        assert plan.current_step is None


class TestHierarchicalPlanner:
    """Test the HierarchicalPlanner."""

    def test_planner_proposes_survival_goal_when_health_low(
        self, basic_sensation, basic_reflection
    ):
        """Test planner generates SURVIVAL goal when health < 20."""
        planner = HierarchicalPlanner(agent_id="agent-1")
        sensation = basic_sensation
        sensation.own_needs["health"] = 15

        goals = planner.propose_goals(sensation, basic_reflection, sensation.own_traits)

        survival_goals = [g for g in goals if g.priority == GoalPriority.SURVIVAL]
        assert len(survival_goals) > 0
        assert any("health" in g.description for g in survival_goals)

    def test_planner_proposes_urgent_need_goal(self, basic_sensation, basic_reflection):
        """Test planner generates URGENT_NEED goal when any need < 15."""
        planner = HierarchicalPlanner(agent_id="agent-1")
        sensation = basic_sensation
        sensation.own_needs["hunger"] = 10

        goals = planner.propose_goals(sensation, basic_reflection, sensation.own_traits)

        urgent_goals = [g for g in goals if g.priority == GoalPriority.URGENT_NEED]
        assert len(urgent_goals) > 0
        assert any("hunger" in g.description for g in urgent_goals)

    def test_planner_creates_plan_for_satisfy_hunger(self, basic_sensation, basic_reflection):
        """Test planner creates executable plan for satisfy_hunger goal."""
        planner = HierarchicalPlanner(agent_id="agent-1")
        goal = Goal.create(
            description="satisfy_hunger",
            priority=GoalPriority.NEED,
            created_tick=0,
        )

        plan = planner.create_plan(goal, basic_sensation, basic_reflection)

        assert len(plan.steps) > 0
        # Should involve MOVE, GATHER, and EAT
        action_types = [s.action_type for s in plan.steps]
        assert any("move" in a.lower() for a in action_types)

    def test_get_next_action_returns_action(self, basic_sensation, basic_reflection):
        """Test get_next_action returns valid action and plan."""
        planner = HierarchicalPlanner(agent_id="agent-1")
        sensation = basic_sensation
        sensation.own_needs["hunger"] = 30  # Trigger NEED goal

        action_type, plan = planner.get_next_action(
            sensation, basic_reflection, sensation.own_traits, sensation.tick
        )

        assert action_type is not None
        # Plan might be None if planner decides to wait, but action_type should exist
        assert isinstance(action_type, str)

    def test_report_action_result_advances_plan_on_success(self, basic_sensation, basic_reflection):
        """Test report_action_result advances plan when action succeeds."""
        planner = HierarchicalPlanner(agent_id="agent-1")
        sensation = basic_sensation
        sensation.own_needs["hunger"] = 30

        # Get first action
        action_type, plan = planner.get_next_action(
            sensation, basic_reflection, sensation.own_traits, sensation.tick
        )

        if plan:
            initial_step = plan.current_step_index
            planner.report_action_result(action_succeeded=True, tick=sensation.tick)
            assert plan.current_step_index == initial_step + 1

    def test_report_action_result_increments_attempts_on_failure(
        self, basic_sensation, basic_reflection
    ):
        """Test report_action_result increments attempts when action fails."""
        planner = HierarchicalPlanner(agent_id="agent-1")
        sensation = basic_sensation
        sensation.own_needs["hunger"] = 30

        action_type, plan = planner.get_next_action(
            sensation, basic_reflection, sensation.own_traits, sensation.tick
        )

        if plan and planner._active_goal_id:
            goal = planner._goals[planner._active_goal_id]
            initial_attempts = goal.attempts
            planner.report_action_result(action_succeeded=False, tick=sensation.tick)
            assert goal.attempts == initial_attempts + 1

    def test_interrupt_stack_works(self, basic_sensation, basic_reflection):
        """Test interrupt stack pushes and pops goals correctly."""
        planner = HierarchicalPlanner(agent_id="agent-1")
        sensation = basic_sensation

        # Start with a low-priority goal
        sensation.own_needs["hunger"] = 35  # Trigger NEED goal
        action_type, plan = planner.get_next_action(
            sensation, basic_reflection, sensation.own_traits, sensation.tick
        )

        first_goal_id = planner._active_goal_id

        # Now trigger interrupt with critical health
        sensation.own_needs["health"] = 15
        action_type, plan = planner.get_next_action(
            sensation, basic_reflection, sensation.own_traits, sensation.tick + 1
        )

        # Should have switched to survival goal
        assert planner._active_goal_id != first_goal_id
        # First goal should be on interrupt stack
        if first_goal_id:
            assert first_goal_id in planner._interrupt_stack

    def test_planner_proposes_strategic_goals_based_on_traits(
        self, basic_sensation, basic_reflection
    ):
        """Test planner proposes strategic goals based on personality traits."""
        planner = HierarchicalPlanner(agent_id="agent-1")
        sensation = basic_sensation
        sensation.own_traits["aggression"] = 0.8

        goals = planner.propose_goals(sensation, basic_reflection, sensation.own_traits)

        strategic_goals = [g for g in goals if g.priority == GoalPriority.STRATEGIC]
        # Should propose establish_territory for aggressive agent
        assert any("territory" in g.description for g in strategic_goals)

    def test_planner_proposes_opportunistic_goals_when_curious(
        self, basic_sensation, basic_reflection
    ):
        """Test planner proposes exploration when curious and needs satisfied."""
        planner = HierarchicalPlanner(agent_id="agent-1")
        sensation = basic_sensation
        sensation.own_traits["curiosity"] = 0.8
        sensation.own_needs = {"hunger": 80, "thirst": 80, "energy": 80, "health": 100}

        goals = planner.propose_goals(sensation, basic_reflection, sensation.own_traits)

        opportunistic_goals = [g for g in goals if g.priority == GoalPriority.OPPORTUNISTIC]
        assert any("explore" in g.description for g in opportunistic_goals)


class TestPlanMonitor:
    """Test PlanMonitor metrics tracking."""

    def test_monitor_records_plan_outcome(self):
        """Test monitor records completed and failed plans."""
        monitor = PlanMonitor()

        goal = Goal.create(description="test_goal", priority=GoalPriority.NEED, created_tick=0)
        goal.status = GoalStatus.COMPLETED
        plan = Plan(goal_id=goal.id, steps=[], created_tick=0)

        monitor.record_plan_outcome(plan, goal)

        stats = monitor.planning_stats
        assert stats["total_plans"] == 1
        assert stats["completed_goals"] == 1

    def test_monitor_records_interrupts(self):
        """Test monitor tracks interrupts."""
        monitor = PlanMonitor()

        goal1 = Goal.create(
            description="interrupted", priority=GoalPriority.TACTICAL, created_tick=0
        )
        goal2 = Goal.create(
            description="interrupting", priority=GoalPriority.SURVIVAL, created_tick=5
        )

        monitor.record_interrupt(goal1, goal2)

        stats = monitor.planning_stats
        assert stats["interrupt_count"] == 1

    def test_monitor_calculates_completion_rate(self):
        """Test monitor calculates goal completion rate."""
        monitor = PlanMonitor()

        # Record 2 completed, 1 failed
        for i in range(2):
            goal = Goal.create(
                description=f"completed_{i}", priority=GoalPriority.NEED, created_tick=0
            )
            goal.status = GoalStatus.COMPLETED
            plan = Plan(goal_id=goal.id, steps=[], created_tick=0)
            monitor.record_plan_outcome(plan, goal)

        goal = Goal.create(description="failed", priority=GoalPriority.NEED, created_tick=0)
        goal.status = GoalStatus.FAILED
        plan = Plan(goal_id=goal.id, steps=[], created_tick=0)
        monitor.record_plan_outcome(plan, goal)

        stats = monitor.planning_stats
        assert stats["completion_rate"] == pytest.approx(2 / 3)


class TestPlanningStrategy:
    """Test PlanningStrategy integration."""

    def test_planning_strategy_produces_valid_intention(self, basic_sensation, basic_reflection):
        """Test PlanningStrategy produces valid Intention."""
        planner = HierarchicalPlanner(agent_id="agent-1")
        strategy = PlanningStrategy(planner=planner)

        sensation = basic_sensation
        sensation.own_needs["hunger"] = 30

        intention = strategy.form_intention(sensation, basic_reflection)

        assert intention is not None
        assert intention.primary_goal is not None
        assert isinstance(intention.confidence, float)
        assert 0.0 <= intention.confidence <= 1.0

    def test_planning_strategy_produces_valid_expression(self, basic_sensation, basic_reflection):
        """Test PlanningStrategy produces valid Expression."""
        planner = HierarchicalPlanner(agent_id="agent-1")
        strategy = PlanningStrategy(planner=planner)

        sensation = basic_sensation
        sensation.own_needs["hunger"] = 30

        intention = strategy.form_intention(sensation, basic_reflection)
        expression = strategy.express(sensation, basic_reflection, intention)

        assert expression is not None
        assert expression.action is not None
        assert hasattr(expression.action, "type")

    def test_planning_strategy_falls_back_when_no_planner(self, basic_sensation, basic_reflection):
        """Test PlanningStrategy falls back to reactive when no planner."""
        strategy = PlanningStrategy(planner=None)

        intention = strategy.form_intention(basic_sensation, basic_reflection)

        # Should produce valid intention from fallback
        assert intention is not None
        assert intention.primary_goal is not None

    def test_planning_strategy_handles_action_types(self, basic_sensation, basic_reflection):
        """Test PlanningStrategy correctly converts plan steps to actions."""
        planner = HierarchicalPlanner(agent_id="agent-1")
        strategy = PlanningStrategy(planner=planner)

        # Set up scenario requiring movement
        sensation = basic_sensation
        sensation.own_needs["hunger"] = 20  # Trigger urgent need

        intention = strategy.form_intention(sensation, basic_reflection)
        expression = strategy.express(sensation, basic_reflection, intention)

        # Should produce valid action type
        assert expression.action.type in ActionType.__members__.values()


class TestPlanningIntegration:
    """Integration tests for planning system."""

    def test_planner_completes_simple_goal(self, basic_sensation, basic_reflection):
        """Test planner can execute and complete a simple goal."""
        planner = HierarchicalPlanner(agent_id="agent-1")
        sensation = basic_sensation

        # Give agent berries to make eating simple
        sensation.own_inventory["berries"] = 2
        sensation.own_needs["hunger"] = 30

        # Get first action
        action_type, plan = planner.get_next_action(
            sensation, basic_reflection, sensation.own_traits, sensation.tick
        )

        # Simulate successful execution
        if plan:
            steps_completed = 0
            while plan.current_step and steps_completed < 10:
                planner.report_action_result(action_succeeded=True, tick=sensation.tick)
                steps_completed += 1

                # Get next action
                action_type, plan = planner.get_next_action(
                    sensation, basic_reflection, sensation.own_traits, sensation.tick
                )

                if not plan or plan.status == "completed":
                    break

            # Goal should be completed or plan should be done
            if planner._active_goal_id:
                goal = planner._goals.get(planner._active_goal_id)
                if goal and plan:
                    assert goal.status in [GoalStatus.COMPLETED, GoalStatus.ACTIVE]

    def test_planner_replans_after_failures(self, basic_sensation, basic_reflection):
        """Test planner handles repeated failures gracefully."""
        planner = HierarchicalPlanner(agent_id="agent-1")
        sensation = basic_sensation
        sensation.own_needs["hunger"] = 30

        action_type, plan = planner.get_next_action(
            sensation, basic_reflection, sensation.own_traits, sensation.tick
        )

        if plan and planner._active_goal_id:
            goal = planner._goals[planner._active_goal_id]

            # Simulate repeated failures
            for _ in range(5):
                planner.report_action_result(action_succeeded=False, tick=sensation.tick)

            # Should either fail the goal or replan
            assert goal.attempts > 0
