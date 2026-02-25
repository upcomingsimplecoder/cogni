"""Planning-driven decision strategy with multi-tick goal pursuit."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.awareness.types import Expression, Intention, Reflection, Sensation
from src.cognition.strategies.personality import PersonalityStrategy
from src.communication.protocol import MessageType, create_message
from src.simulation.actions import ActionType

if TYPE_CHECKING:
    from src.planning.planner import HierarchicalPlanner


class PlanningStrategy:
    """DecisionStrategy that uses HierarchicalPlanner for multi-tick planning.

    Implements DecisionStrategy protocol:
    - form_intention: queries planner for current goal → Intention
    - express: converts plan step to Action → Expression

    Falls back to PersonalityStrategy when planner returns None or is unavailable.
    """

    def __init__(self, planner: HierarchicalPlanner | None = None):
        self._planner = planner
        self._fallback = PersonalityStrategy()  # For reactive fallback

    def form_intention(self, sensation: Sensation, reflection: Reflection) -> Intention:
        """Get intention from planner, or fall back to reactive strategy."""
        if self._planner is None:
            return self._fallback.form_intention(sensation, reflection)

        traits = sensation.own_traits
        action_type, plan = self._planner.get_next_action(
            sensation, reflection, traits, sensation.tick
        )

        # No plan available — fall back to reactive
        if plan is None:
            return self._fallback.form_intention(sensation, reflection)

        # Build intention from plan
        goal = self._planner._goals.get(plan.goal_id)
        goal_description = goal.description if goal else action_type

        return Intention(
            primary_goal=goal_description,
            target_position=(
                plan.current_step.target_position
                if plan.current_step and plan.current_step.target_position
                else None
            ),
            target_agent_id=(
                plan.current_step.target_agent_id
                if plan.current_step and plan.current_step.target_agent_id
                else None
            ),
            planned_actions=[s.action_type for s in plan.steps[plan.current_step_index :]]
            if plan
            else [action_type],
            confidence=1.0 - (plan.revision_count * 0.1) if plan else 0.5,
        )

    def express(
        self, sensation: Sensation, reflection: Reflection, intention: Intention
    ) -> Expression:
        """Convert plan step to concrete action.

        Delegates to personality strategy for action generation guided by
        the planner's chosen goal and target.
        """
        # If we have a planner and intention is from planner, convert directly
        if self._planner and self._planner._active_goal_id:
            plan = self._planner._plans.get(self._planner._active_goal_id)
            if plan and plan.current_step:
                return self._express_plan_step(plan.current_step, sensation, intention)

        # Otherwise delegate to personality strategy
        return self._fallback.express(sensation, reflection, intention)

    def _express_plan_step(self, step, sensation: Sensation, intention: Intention) -> Expression:
        """Convert a PlanStep to an Expression."""
        from src.simulation.actions import Action

        action_type_str = step.action_type

        # Convert string to ActionType enum
        try:
            action_enum = ActionType(action_type_str)
        except ValueError:
            action_enum = ActionType.WAIT

        # Build action based on type
        match action_enum:
            case ActionType.MOVE:
                # Determine direction from target position
                if step.target_position:
                    direction = self._direction_toward(sensation.own_position, step.target_position)
                    if direction:
                        return Expression(
                            action=Action(type=action_enum, direction=direction),
                            internal_monologue=f"Moving toward {step.target_position}",
                        )
                # No target, explore
                return self._fallback._express_explore(sensation)

            case ActionType.GATHER:
                return Expression(
                    action=Action(type=action_enum, target=step.target),
                    internal_monologue=f"Gathering {step.target}",
                )

            case ActionType.EAT:
                return Expression(action=Action(type=action_enum))

            case ActionType.DRINK:
                return Expression(action=Action(type=action_enum))

            case ActionType.REST:
                return Expression(action=Action(type=action_enum))

            case ActionType.GIVE:
                if step.target_agent_id and step.target:
                    return Expression(
                        action=Action(
                            type=action_enum,
                            target=step.target,
                            target_agent_id=step.target_agent_id,
                            quantity=1,
                        ),
                        internal_monologue=f"Giving {step.target} to {step.target_agent_id}",
                    )

            case ActionType.ATTACK:
                if step.target_agent_id:
                    return Expression(
                        action=Action(
                            type=action_enum,
                            target_agent_id=step.target_agent_id,
                        ),
                        internal_monologue=f"Attacking {step.target_agent_id}",
                    )

            case ActionType.SEND_MESSAGE:
                if step.target_agent_id:
                    # Determine message type from goal
                    msg_type = MessageType.INFORM
                    content = "Hello"
                    if intention.primary_goal == "establish_territory":
                        msg_type = MessageType.THREAT
                        content = "This is my territory!"
                    elif intention.primary_goal == "build_alliances":
                        msg_type = MessageType.INFORM
                        content = "Let's work together"

                    msg = create_message(
                        tick=sensation.tick,
                        sender_id=None,  # Filled by awareness loop
                        receiver_id=step.target_agent_id,
                        message_type=msg_type,
                        content=content,
                    )
                    return Expression(
                        action=Action(type=action_enum),
                        message=msg,
                        internal_monologue=f"Sending {msg_type.name} to {step.target_agent_id}",
                    )

            case _:
                return Expression(action=Action(type=ActionType.WAIT))

        return Expression(action=Action(type=ActionType.WAIT))

    def _direction_toward(self, from_pos: tuple[int, int], to_pos: tuple[int, int]):
        """Get cardinal direction toward target."""
        from src.simulation.actions import Direction

        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        if dx == 0 and dy == 0:
            return None
        if abs(dx) >= abs(dy):
            return Direction.EAST if dx > 0 else Direction.WEST
        else:
            return Direction.SOUTH if dy > 0 else Direction.NORTH
