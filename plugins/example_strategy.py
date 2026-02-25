"""Example plugin: Cautious Gatherer Strategy.

This is an example plugin demonstrating how to extend AUTOCOG with custom
decision strategies. It implements a risk-averse gatherer that prioritizes
safety over resource collection.

To use this plugin, place it in the plugins/ directory and it will be
automatically discovered and loaded.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from src.plugins import PluginRegistry
from src.awareness.types import Intention, Expression
from src.simulation.actions import Action, ActionType, Direction

if TYPE_CHECKING:
    from src.awareness.types import Sensation, Reflection


@PluginRegistry.register_strategy("cautious_gatherer")
class CautiousGathererStrategy:
    """Conservative gatherer that avoids risk and prioritizes survival.

    Decision logic:
    - If health < 40%: REST to recover
    - If threat_level > 0.4: FLEE from danger
    - If hunger < 30%: Prioritize food gathering
    - If thirst < 30%: Prioritize water gathering
    - Otherwise: Gather berries cautiously
    """

    def __init__(self):
        """Initialize the strategy."""
        self.flee_threshold = 0.4
        self.critical_health = 40.0
        self.critical_hunger = 30.0
        self.critical_thirst = 30.0

    def form_intention(
        self, sensation: Sensation, reflection: Reflection
    ) -> Intention:
        """Decide what to do based on current situation.

        Args:
            sensation: Current perceptions
            reflection: Evaluated situation

        Returns:
            Intention with primary goal and confidence
        """
        # Critical health: must rest
        if sensation.own_needs.get("health", 100.0) < self.critical_health:
            return Intention(
                primary_goal="rest",
                confidence=0.95,
                planned_actions=["rest"],
            )

        # High threat: flee immediately
        if reflection.threat_level > self.flee_threshold:
            return Intention(
                primary_goal="flee",
                confidence=0.9,
                planned_actions=["flee", "rest"],
            )

        # Critical hunger: prioritize food
        hunger = sensation.own_needs.get("hunger", 100.0)
        if hunger < self.critical_hunger:
            # Check if we have food in inventory
            if sensation.own_inventory.get("berries", 0) > 0:
                return Intention(
                    primary_goal="eat",
                    confidence=0.85,
                    planned_actions=["eat"],
                )
            else:
                return Intention(
                    primary_goal="gather_food",
                    confidence=0.85,
                    planned_actions=["gather", "eat"],
                )

        # Critical thirst: prioritize water
        thirst = sensation.own_needs.get("thirst", 100.0)
        if thirst < self.critical_thirst:
            if sensation.own_inventory.get("water", 0) > 0:
                return Intention(
                    primary_goal="drink",
                    confidence=0.85,
                    planned_actions=["drink"],
                )
            else:
                return Intention(
                    primary_goal="gather_water",
                    confidence=0.85,
                    planned_actions=["gather", "drink"],
                )

        # Default: cautious resource gathering
        return Intention(
            primary_goal="gather_resources",
            confidence=0.6,
            planned_actions=["gather", "rest"],
        )

    def express(
        self,
        sensation: Sensation,
        reflection: Reflection,
        intention: Intention,
    ) -> Expression:
        """Convert intention into concrete action.

        Args:
            sensation: Current perceptions
            reflection: Evaluated situation
            intention: Desired goal

        Returns:
            Expression with action to execute
        """
        action: Action

        match intention.primary_goal:
            case "rest":
                action = Action(type=ActionType.REST)

            case "flee":
                # Flee in direction away from threats
                # For simplicity, flee north by default
                action = Action(type=ActionType.FLEE, direction=Direction.NORTH)

            case "eat":
                action = Action(type=ActionType.EAT)

            case "drink":
                action = Action(type=ActionType.DRINK)

            case "gather_food":
                action = Action(type=ActionType.GATHER, target="berry_bush")

            case "gather_water":
                action = Action(type=ActionType.GATHER, target="water_source")

            case "gather_resources":
                # Prefer berries as safe food source
                action = Action(type=ActionType.GATHER, target="berry_bush")

            case _:
                # Fallback: wait
                action = Action(type=ActionType.WAIT)

        # Add internal monologue for debugging/visualization
        monologue = self._generate_monologue(intention, sensation)

        return Expression(action=action, internal_monologue=monologue)

    def _generate_monologue(
        self, intention: Intention, sensation: Sensation
    ) -> str:
        """Generate internal monologue based on intention.

        Args:
            intention: Current intention
            sensation: Current perceptions

        Returns:
            Internal thought string
        """
        health = sensation.own_needs.get("health", 100.0)
        hunger = sensation.own_needs.get("hunger", 100.0)
        thirst = sensation.own_needs.get("thirst", 100.0)

        match intention.primary_goal:
            case "rest":
                return f"My health is low ({health:.0f}%). I need to rest and recover."
            case "flee":
                return "Danger! I need to get away from here immediately."
            case "eat":
                return f"I'm very hungry ({hunger:.0f}%). Time to eat."
            case "drink":
                return f"I'm very thirsty ({thirst:.0f}%). Time to drink."
            case "gather_food":
                return "I need food. Let me look for berry bushes."
            case "gather_water":
                return "I need water. Let me find a water source."
            case "gather_resources":
                return "Things are stable. Let me gather some resources."
            case _:
                return "I'll wait and observe for now."


# Plugin metadata (optional, for introspection)
PLUGIN_INFO = {
    "name": "Cautious Gatherer Strategy",
    "version": "1.0.0",
    "author": "AUTOCOG",
    "description": "Risk-averse gatherer that prioritizes safety and survival",
    "strategy_name": "cautious_gatherer",
}
