"""Protocols for pluggable awareness pipeline components."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from src.awareness.types import Reflection, Sensation

if TYPE_CHECKING:
    from typing import Any

    from src.simulation.entities import Agent


@runtime_checkable
class PerceptionStrategy(Protocol):
    """Pluggable sensation — how an agent perceives the world."""

    def perceive(self, agent: Agent, engine: Any) -> Sensation:
        """Build a Sensation snapshot for this agent."""
        ...


@runtime_checkable
class EvaluationStrategy(Protocol):
    """Pluggable reflection — how an agent evaluates its situation."""

    def evaluate(self, agent: Agent, engine: Any, sensation: Sensation) -> Reflection:
        """Evaluate recent experience and current perception."""
        ...


@runtime_checkable
class DeliberationStrategy(Protocol):
    """Optional deliberation layer between reflection and intention.

    Returns a refined reflection OR triggers System 2 escalation.
    """

    def should_escalate(self, sensation: Sensation, reflection: Reflection) -> bool:
        """Determine if deliberation (System 2) should activate."""
        ...

    def deliberate(self, agent: Agent, sensation: Sensation, reflection: Reflection) -> Reflection:
        """Perform deeper analysis, returning refined reflection."""
        ...
