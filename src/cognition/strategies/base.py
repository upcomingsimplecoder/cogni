"""Base protocol for decision-making strategies."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.awareness.types import Expression, Intention, Reflection, Sensation


@runtime_checkable
class DecisionStrategy(Protocol):
    """Protocol that any decision-making strategy must implement.

    Strategies can be:
    - HardcodedStrategy (existing priority-based logic)
    - PersonalityStrategy (trait-biased decisions)
    - LLMStrategy (future — call language model)
    """

    def form_intention(self, sensation: Sensation, reflection: Reflection) -> Intention:
        """Given perception and evaluation, what does the agent want?"""
        ...

    def express(
        self, sensation: Sensation, reflection: Reflection, intention: Intention
    ) -> Expression:
        """Given intention, what action does the agent take?"""
        ...
