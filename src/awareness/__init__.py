"""Awareness system: Sensation-Reflection-Intention-Expression loop."""

from src.awareness.architecture import (
    CognitiveArchitecture,
    build_awareness_loop,
    get_architectures,
)
from src.awareness.protocols import (
    DeliberationStrategy,
    EvaluationStrategy,
    PerceptionStrategy,
)
from src.awareness.reflection import ReflectionModule
from src.awareness.sensation import SensationModule
from src.awareness.types import (
    AgentSummary,
    Expression,
    Intention,
    InteractionOutcome,
    Reflection,
    Sensation,
    TileSummary,
)

__all__ = [
    "Sensation",
    "Reflection",
    "Intention",
    "Expression",
    "TileSummary",
    "AgentSummary",
    "InteractionOutcome",
    "SensationModule",
    "ReflectionModule",
    "PerceptionStrategy",
    "EvaluationStrategy",
    "DeliberationStrategy",
    "CognitiveArchitecture",
    "build_awareness_loop",
    "get_architectures",
]
