"""Central registry for all AUTOCOG extension points.

Uses a class-level registry pattern for global access without singleton instantiation.
Supports decorators for registering strategies, actions, archetypes, and metrics.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Central registry for all extension points.

    Class-level registries — all methods are classmethods for global access.
    """

    _strategies: dict[str, type] = {}
    _actions: dict[str, Callable] = {}
    _archetypes: dict[str, dict] = {}
    _evaluations: dict[str, type] = {}
    _metrics: dict[str, Callable] = {}

    @classmethod
    def register_strategy(cls, name: str):
        """Decorator to register a new DecisionStrategy.

        Usage:
            @PluginRegistry.register_strategy("my_strategy")
            class MyStrategy:
                def form_intention(self, sensation, reflection) -> Intention: ...
                def express(self, sensation, reflection, intention) -> Expression: ...
        """

        def decorator(strategy_cls: type) -> type:
            if name in cls._strategies:
                logger.warning(f"Strategy '{name}' already registered, overriding")
            cls._strategies[name] = strategy_cls
            logger.debug(f"Registered strategy: {name}")
            return strategy_cls

        return decorator

    @classmethod
    def register_action(cls, name: str):
        """Decorator to register a new action handler.

        Usage:
            @PluginRegistry.register_action("my_action")
            def my_action_handler(action, agent, world, registry) -> ActionResult:
                ...
        """

        def decorator(handler: Callable) -> Callable:
            if name in cls._actions:
                logger.warning(f"Action '{name}' already registered, overriding")
            cls._actions[name] = handler
            logger.debug(f"Registered action: {name}")
            return handler

        return decorator

    @classmethod
    def register_archetype(
        cls, name: str, traits: dict[str, Any], color: str = "white", symbol: str = "?"
    ):
        """Register a new agent archetype.

        Args:
            name: Archetype identifier
            traits: Dict of personality trait values (cooperation_tendency, curiosity, etc.)
            color: Display color for visualization
            symbol: Single-character symbol for display
        """
        if name in cls._archetypes:
            logger.warning(f"Archetype '{name}' already registered, overriding")
        cls._archetypes[name] = {
            "traits": traits,
            "color": color,
            "symbol": symbol,
        }
        logger.debug(f"Registered archetype: {name}")

    @classmethod
    def register_evaluation(cls, name: str):
        """Decorator for evaluation strategy.

        Usage:
            @PluginRegistry.register_evaluation("my_eval")
            class MyEvaluation:
                def evaluate(self, agent, engine, sensation) -> Reflection: ...
        """

        def decorator(eval_cls: type) -> type:
            if name in cls._evaluations:
                logger.warning(f"Evaluation '{name}' already registered, overriding")
            cls._evaluations[name] = eval_cls
            logger.debug(f"Registered evaluation: {name}")
            return eval_cls

        return decorator

    @classmethod
    def register_metric(cls, name: str):
        """Decorator for custom metric extractors.

        Usage:
            @PluginRegistry.register_metric("my_metric")
            def my_metric(engine) -> float:
                ...
        """

        def decorator(metric_fn: Callable) -> Callable:
            if name in cls._metrics:
                logger.warning(f"Metric '{name}' already registered, overriding")
            cls._metrics[name] = metric_fn
            logger.debug(f"Registered metric: {name}")
            return metric_fn

        return decorator

    @classmethod
    def get_strategy(cls, name: str) -> type | None:
        """Get a registered strategy class by name."""
        return cls._strategies.get(name)

    @classmethod
    def get_action_handler(cls, name: str) -> Callable | None:
        """Get a registered action handler by name."""
        return cls._actions.get(name)

    @classmethod
    def get_archetype(cls, name: str) -> dict[str, Any] | None:
        """Get a registered archetype by name."""
        return cls._archetypes.get(name)

    @classmethod
    def get_evaluation(cls, name: str) -> type | None:
        """Get a registered evaluation strategy by name."""
        return cls._evaluations.get(name)

    @classmethod
    def get_metric(cls, name: str) -> Callable | None:
        """Get a registered metric extractor by name."""
        return cls._metrics.get(name)

    @classmethod
    def all_strategies(cls) -> dict[str, type]:
        """Get all registered strategies."""
        return dict(cls._strategies)

    @classmethod
    def all_archetypes(cls) -> dict[str, dict[str, Any]]:
        """Merge built-in archetypes with plugins.

        Returns a unified dict of all archetypes, with plugin-registered ones
        taking precedence over built-ins if there's a name collision.
        """
        from src.agents.archetypes import ARCHETYPES

        merged: dict[str, dict[str, Any]] = {}

        # Add built-in archetypes first
        for name, arch in ARCHETYPES.items():
            traits_dict = (
                arch["traits"].as_dict()
                if hasattr(arch["traits"], "as_dict")
                else (arch["traits"] if isinstance(arch["traits"], dict) else {})
            )
            merged[name] = {
                "traits": traits_dict,
                "color": arch.get("color", "white"),
                "symbol": arch.get("symbol", "?"),
            }

        # Plugin archetypes override built-ins
        merged.update(cls._archetypes)

        return merged

    @classmethod
    def clear(cls):
        """Clear all registrations. Useful for testing."""
        cls._strategies.clear()
        cls._actions.clear()
        cls._archetypes.clear()
        cls._evaluations.clear()
        cls._metrics.clear()
        logger.debug("Cleared all plugin registrations")
