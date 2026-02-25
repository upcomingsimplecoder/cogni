"""Metacognitive strategy wrapper: blends FOK into confidence.

The outermost strategy wrapper that integrates feeling-of-knowing (FOK)
into the decision pipeline. Wraps any DecisionStrategy and adjusts the
confidence of its intentions based on metacognitive awareness.
"""

from __future__ import annotations

from typing import Any

from src.awareness.types import Expression, Intention, Reflection, Sensation


class MetacognitiveStrategy:
    """Strategy wrapper that blends feeling-of-knowing into intention confidence.

    Satisfies DecisionStrategy protocol (form_intention + express).
    Adjusts confidence = 0.5 * base_confidence + 0.5 * FOK.

    The engine sets FOK via set_fok() before each SRIE cycle.
    """

    def __init__(
        self,
        inner_strategy: Any,
        strategy_name: str = "personality",
        fok_enabled: bool = True,
    ):
        """Initialize the metacognitive strategy wrapper.

        Args:
            inner_strategy: The wrapped strategy (often CulturallyModulatedStrategy)
            strategy_name: Name of the currently active base strategy
            fok_enabled: Whether to blend FOK into confidence
        """
        self._inner = inner_strategy
        self._active_strategy_name = strategy_name
        self._current_fok: float = 0.5
        self._fok_enabled = fok_enabled

    def form_intention(self, sensation: Sensation, reflection: Reflection) -> Intention:
        """Form intention with FOK-adjusted confidence.

        Delegates to inner strategy, then blends FOK into the confidence
        if enabled. Returns a new Intention with adjusted confidence.

        Args:
            sensation: Current perception
            reflection: Current evaluation

        Returns:
            Intention with potentially adjusted confidence
        """
        # Get base intention from inner strategy
        base_intention: Intention = self._inner.form_intention(sensation, reflection)

        # Adjust confidence if FOK is enabled
        if self._fok_enabled:
            adjusted_confidence = 0.5 * base_intention.confidence + 0.5 * self._current_fok
            # Return new Intention with adjusted confidence
            return Intention(
                primary_goal=base_intention.primary_goal,
                target_position=base_intention.target_position,
                target_agent_id=base_intention.target_agent_id,
                planned_actions=base_intention.planned_actions,
                confidence=adjusted_confidence,
            )

        return base_intention

    def express(
        self, sensation: Sensation, reflection: Reflection, intention: Intention
    ) -> Expression:
        """Convert intention to action.

        Pure delegation to inner strategy.

        Args:
            sensation: Current perception
            reflection: Current evaluation
            intention: Intention to express

        Returns:
            Expression with action and optional message
        """
        result: Expression = self._inner.express(sensation, reflection, intention)
        return result

    def set_fok(self, fok: float) -> None:
        """Set the current feeling-of-knowing value.

        Called by the engine before each SRIE cycle.

        Args:
            fok: Feeling of knowing (0-1), clamped to valid range
        """
        self._current_fok = max(0.0, min(1.0, fok))

    @property
    def active_strategy_name(self) -> str:
        """Get the name of the currently active base strategy."""
        return self._active_strategy_name

    @property
    def inner(self) -> Any:
        """Get the inner strategy (for engine to access/rebuild chain)."""
        return self._inner
