"""Culturally-modulated decision strategy.

Wraps any DecisionStrategy to check cultural repertoire before falling
back to personality-based behavior. This is the integration point between
cultural transmission (learned behaviors) and the SRIE cognitive pipeline.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from src.awareness.types import Expression, Intention, Reflection, Sensation
from src.simulation.actions import Action, ActionType, Direction

if TYPE_CHECKING:
    pass


class CulturallyModulatedStrategy:
    """Wraps any DecisionStrategy to check cultural repertoire first.

    Before the inner strategy forms an intention, this wrapper checks
    if the agent has an adopted cultural variant for the current context.
    If yes (and the probabilistic override fires), it short-circuits
    to the culturally learned behavior. Otherwise, it falls back to the
    inner strategy.

    Satisfies the DecisionStrategy protocol (form_intention + express).
    """

    def __init__(
        self,
        inner_strategy: Any,
        repertoire: Any,
        override_probability: float = 0.7,
    ):
        """Initialize the culturally-modulated strategy.

        Args:
            inner_strategy: The base strategy to wrap (e.g., PersonalityStrategy)
            repertoire: This agent's BehavioralRepertoire
            override_probability: Probability that cultural variant overrides
                personality-based decision (0.0 = never, 1.0 = always)
        """
        self._inner = inner_strategy
        self._repertoire = repertoire
        self._override_probability = override_probability

    def form_intention(self, sensation: Sensation, reflection: Reflection) -> Intention:
        """Check repertoire first, fall back to inner strategy.

        Extracts the current context tag from sensation, looks up any
        adopted variant, and probabilistically overrides the inner
        strategy's intention.

        Args:
            sensation: Current perception
            reflection: Current evaluation

        Returns:
            Intention from cultural repertoire or inner strategy
        """
        from src.evolution.observation import ContextTag

        context = ContextTag.extract_primary(sensation)
        variant = self._repertoire.lookup(context)

        if variant is not None and random.random() < self._override_probability:
            # Cultural override: use learned variant
            result_intention: Intention = Intention(
                primary_goal=f"cultural_{variant.action_type}",
                planned_actions=[variant.action_type],
                confidence=min(0.9, variant.observed_success_rate),
            )
            return result_intention

        # Fall back to personality-based decision
        return self._inner.form_intention(sensation, reflection)

    def express(
        self, sensation: Sensation, reflection: Reflection, intention: Intention
    ) -> Expression:
        """Convert intention to concrete action.

        If the intention was culturally derived (prefixed with "cultural_"),
        maps the action type string to a concrete Action. Otherwise
        delegates to the inner strategy.

        Args:
            sensation: Current perception
            reflection: Current evaluation
            intention: Intention to express

        Returns:
            Expression with action and optional message
        """
        if intention.primary_goal.startswith("cultural_"):
            action_type_str = intention.primary_goal.removeprefix("cultural_")
            try:
                action_type = ActionType(action_type_str)
            except ValueError:
                # Invalid action type — fall through to inner strategy
                return self._inner.express(sensation, reflection, intention)

            action = self._build_cultural_action(action_type, sensation)
            result_expression: Expression = Expression(
                action=action,
                internal_monologue=f"Cultural: {action_type_str} (learned behavior)",
            )
            return result_expression

        return self._inner.express(sensation, reflection, intention)

    def _build_cultural_action(self, action_type: ActionType, sensation: Sensation) -> Action:
        """Build a concrete Action from a cultural action type.

        Maps ActionType to a fully-specified Action with appropriate
        targets/directions based on current sensation.

        Args:
            action_type: The action type to build
            sensation: Current perception for context

        Returns:
            Concrete Action instance
        """
        match action_type:
            case ActionType.GATHER:
                # Gather whatever resource is on the current tile
                target = None
                for tile in sensation.visible_tiles:
                    if (tile.x, tile.y) == sensation.own_position and tile.resources:
                        target = tile.resources[0][0]
                        break
                return Action(type=ActionType.GATHER, target=target)

            case ActionType.EAT:
                return Action(type=ActionType.EAT)

            case ActionType.DRINK:
                return Action(type=ActionType.DRINK)

            case ActionType.REST:
                return Action(type=ActionType.REST)

            case ActionType.WAIT:
                return Action(type=ActionType.WAIT)

            case ActionType.MOVE:
                # Move in a tick-based rotating direction, avoiding water
                directions = list(Direction)
                idx = sensation.tick % len(directions)
                ax, ay = sensation.own_position

                for i in range(len(directions)):
                    d = directions[(idx + i) % len(directions)]
                    dx, dy = d.value
                    tx, ty = ax + dx, ay + dy
                    for tile in sensation.visible_tiles:
                        if (tile.x, tile.y) == (tx, ty) and tile.tile_type != "water":
                            return Action(type=ActionType.MOVE, direction=d)

                return Action(type=ActionType.WAIT)

            case _:
                # For action types without special logic, pass through
                return Action(type=action_type)
