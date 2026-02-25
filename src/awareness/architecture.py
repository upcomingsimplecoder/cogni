"""Cognitive architecture registry: named configurations of SRIE pipeline components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.awareness.deliberation import (
    ConsensusDeliberation,
    ThresholdDeliberation,
)
from src.awareness.protocols import DeliberationStrategy, EvaluationStrategy, PerceptionStrategy
from src.awareness.reflection import ReflectionModule
from src.awareness.reflection_variants import (
    OptimisticReflection,
    PessimisticReflection,
    SocialReflection,
)
from src.awareness.sensation import SensationModule

if TYPE_CHECKING:
    from src.awareness.loop import AwarenessLoop
    from src.simulation.entities import Agent


@dataclass
class CognitiveArchitecture:
    """Named configuration of SRIE pipeline components."""

    name: str
    perception: PerceptionStrategy
    evaluation: EvaluationStrategy
    deliberation: DeliberationStrategy | None
    strategy_factory: type  # Class that creates DecisionStrategy (e.g., PersonalityStrategy)
    description: str = ""


def _build_architectures() -> dict[str, CognitiveArchitecture]:
    """Build the registry of pre-configured architectures.

    Import PersonalityStrategy here to avoid circular imports.
    """
    from src.cognition.strategies.personality import PersonalityStrategy
    from src.cognition.strategies.planning import PlanningStrategy

    return {
        "reactive": CognitiveArchitecture(
            name="reactive",
            perception=SensationModule(),
            evaluation=ReflectionModule(),
            deliberation=None,
            strategy_factory=PersonalityStrategy,
            description="System 1 only. Fast reactive decisions.",
        ),
        "cautious": CognitiveArchitecture(
            name="cautious",
            perception=SensationModule(),
            evaluation=PessimisticReflection(),
            deliberation=ThresholdDeliberation(threat_threshold=0.5),
            strategy_factory=PersonalityStrategy,
            description="Pessimistic + low escalation threshold. Careful.",
        ),
        "optimistic": CognitiveArchitecture(
            name="optimistic",
            perception=SensationModule(),
            evaluation=OptimisticReflection(),
            deliberation=None,
            strategy_factory=PersonalityStrategy,
            description="Optimistic evaluation. Takes more risks.",
        ),
        "social": CognitiveArchitecture(
            name="social",
            perception=SensationModule(),
            evaluation=SocialReflection(),
            deliberation=ConsensusDeliberation(),
            strategy_factory=PersonalityStrategy,
            description="Social-first evaluation. Group-oriented.",
        ),
        "dual_process": CognitiveArchitecture(
            name="dual_process",
            perception=SensationModule(),
            evaluation=ReflectionModule(),
            deliberation=ThresholdDeliberation(),
            strategy_factory=PersonalityStrategy,
            description="System 1/2 with threat-based escalation.",
        ),
        "planning": CognitiveArchitecture(
            name="planning",
            perception=SensationModule(),
            evaluation=ReflectionModule(),
            deliberation=ThresholdDeliberation(),
            strategy_factory=PlanningStrategy,
            description="Multi-tick hierarchical planning with goal pursuit.",
        ),
        "metacognitive": CognitiveArchitecture(
            name="metacognitive",
            perception=SensationModule(),
            evaluation=ReflectionModule(),
            deliberation=ThresholdDeliberation(),
            strategy_factory=PersonalityStrategy,
            description="Full System 1/2 with metacognitive monitoring and control.",
        ),
    }


# Lazy initialization to avoid import issues
_ARCHITECTURES: dict[str, CognitiveArchitecture] | None = None


def get_architectures() -> dict[str, CognitiveArchitecture]:
    """Get the architecture registry, initializing it if needed."""
    global _ARCHITECTURES
    if _ARCHITECTURES is None:
        _ARCHITECTURES = _build_architectures()
    return _ARCHITECTURES


def build_awareness_loop(
    agent: Agent,
    architecture_name: str,
    strategy_override: Any | None = None,
) -> AwarenessLoop:
    """Create an AwarenessLoop from a named architecture.

    Args:
        agent: The agent to create the loop for
        architecture_name: Name of the architecture to use (e.g., "reactive", "cautious")
        strategy_override: Optional DecisionStrategy to use instead of the architecture's default

    Returns:
        Configured AwarenessLoop instance
    """
    from src.awareness.loop import AwarenessLoop

    architectures = get_architectures()
    if architecture_name not in architectures:
        raise ValueError(
            f"Unknown architecture: {architecture_name}. "
            f"Available: {', '.join(architectures.keys())}"
        )

    arch = architectures[architecture_name]

    # Use override strategy if provided, otherwise create from factory
    strategy = strategy_override if strategy_override is not None else arch.strategy_factory()

    return AwarenessLoop(
        agent=agent,
        strategy=strategy,
        perception=arch.perception,
        evaluation=arch.evaluation,
        deliberation=arch.deliberation,
    )
