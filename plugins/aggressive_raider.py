"""Aggressive Raider Strategy.

This plugin implements a high-risk, combat-focused strategy that prioritizes
attacking other agents, stealing their resources, and only retreating when
critically wounded. Designed for competitive survival scenarios.

To use this plugin, place it in the plugins/ directory and it will be
automatically discovered and loaded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.awareness.types import Expression, Intention
from src.plugins import PluginRegistry
from src.simulation.actions import Action, ActionType, Direction

if TYPE_CHECKING:
    from src.awareness.types import Reflection, Sensation


@PluginRegistry.register_strategy("aggressive_raider")
class AggressiveRaiderStrategy:
    """Aggressive combatant that targets agents and steals resources.

    Decision logic:
    - If health < 25%: FLEE to safety
    - If visible agents and health > 40%: ATTACK nearest agent
    - If no agents visible and resources available: GATHER
    - If holding stolen loot and injured: REST
    - Otherwise: SCOUT for targets
    """

    def __init__(self):
        """Initialize the strategy."""
        self.critical_health = 25.0
        self.combat_threshold = 40.0
        self.aggression_level = 0.85

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
        health = sensation.own_needs.get("health", 100.0)

        # Critical health: must flee
        if health < self.critical_health:
            return Intention(
                primary_goal="flee",
                confidence=0.95,
                planned_actions=["flee", "rest"],
            )

        # Healthy enough to fight: target visible agents
        if len(sensation.visible_agents) > 0 and health > self.combat_threshold:
            # Find nearest agent to attack
            nearest_agent = self._find_nearest_agent(
                sensation.own_position, sensation.visible_agents
            )
            if nearest_agent:
                return Intention(
                    primary_goal="attack",
                    confidence=0.9,
                    target_agent_id=nearest_agent.agent_id,
                    target_position=nearest_agent.position,
                    planned_actions=["move_to_target", "attack", "loot"],
                )

        # No targets visible: gather resources for survival
        hunger = sensation.own_needs.get("hunger", 100.0)
        thirst = sensation.own_needs.get("thirst", 100.0)

        if hunger < 40.0:
            # Check if we have food
            if sensation.own_inventory.get("berries", 0) > 0:
                return Intention(
                    primary_goal="eat",
                    confidence=0.7,
                    planned_actions=["eat"],
                )
            else:
                return Intention(
                    primary_goal="gather_food",
                    confidence=0.7,
                    planned_actions=["gather", "eat"],
                )

        if thirst < 40.0:
            if sensation.own_inventory.get("water", 0) > 0:
                return Intention(
                    primary_goal="drink",
                    confidence=0.7,
                    planned_actions=["drink"],
                )
            else:
                return Intention(
                    primary_goal="gather_water",
                    confidence=0.7,
                    planned_actions=["gather", "drink"],
                )

        # Injured but holding resources: rest to recover
        if health < 70.0 and self._has_resources(sensation.own_inventory):
            return Intention(
                primary_goal="rest",
                confidence=0.65,
                planned_actions=["rest"],
            )

        # Default: scout for targets or gather opportunistically
        return Intention(
            primary_goal="scout",
            confidence=0.6,
            planned_actions=["scout", "gather"],
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
            case "flee":
                # Flee away from threats (default: random safe direction)
                flee_dir = self._choose_flee_direction(
                    sensation.own_position, sensation.visible_agents
                )
                action = Action(type=ActionType.FLEE, direction=flee_dir)

            case "attack":
                if intention.target_agent_id:
                    # Check if we need to move closer first
                    if self._is_adjacent(
                        sensation.own_position, intention.target_position
                    ):
                        action = Action(
                            type=ActionType.ATTACK,
                            target_agent_id=intention.target_agent_id,
                        )
                    else:
                        # Move toward target
                        direction = self._direction_to_target(
                            sensation.own_position, intention.target_position
                        )
                        action = Action(type=ActionType.MOVE, direction=direction)
                else:
                    action = Action(type=ActionType.WAIT)

            case "eat":
                action = Action(type=ActionType.EAT)

            case "drink":
                action = Action(type=ActionType.DRINK)

            case "gather_food":
                action = Action(type=ActionType.GATHER, target="berry_bush")

            case "gather_water":
                action = Action(type=ActionType.GATHER, target="water_source")

            case "rest":
                action = Action(type=ActionType.REST)

            case "scout":
                # Move in search of targets
                action = Action(type=ActionType.SCOUT)

            case _:
                action = Action(type=ActionType.WAIT)

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
        visible_count = len(sensation.visible_agents)

        match intention.primary_goal:
            case "flee":
                return f"Too wounded to fight ({health:.0f}%). Retreating!"
            case "attack":
                return (
                    f"Target acquired! Moving in for the attack. "
                    f"{visible_count} agent(s) visible."
                )
            case "eat":
                return "Need to maintain strength. Eating quickly."
            case "drink":
                return "Quick drink, then back to hunting."
            case "gather_food":
                return "No targets around. Gathering food for later."
            case "gather_water":
                return "No targets around. Gathering water for later."
            case "rest":
                return f"Recovering from wounds ({health:.0f}%). Will resume hunting soon."
            case "scout":
                return "Searching for targets and opportunities..."
            case _:
                return "Waiting for an opportunity to strike."

    def _find_nearest_agent(
        self, own_pos: tuple[int, int], visible_agents: list
    ) -> object | None:
        """Find the nearest visible agent."""
        if not visible_agents:
            return None

        min_dist = float("inf")
        nearest = None
        for agent in visible_agents:
            dist = self._manhattan_distance(own_pos, agent.position)
            if dist < min_dist:
                min_dist = dist
                nearest = agent
        return nearest

    def _manhattan_distance(
        self, pos1: tuple[int, int], pos2: tuple[int, int]
    ) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _is_adjacent(
        self, own_pos: tuple[int, int], target_pos: tuple[int, int] | None
    ) -> bool:
        """Check if target position is adjacent (same tile)."""
        if target_pos is None:
            return False
        return own_pos == target_pos

    def _direction_to_target(
        self, own_pos: tuple[int, int], target_pos: tuple[int, int] | None
    ) -> Direction:
        """Calculate direction to move toward target."""
        if target_pos is None:
            return Direction.NORTH

        dx = target_pos[0] - own_pos[0]
        dy = target_pos[1] - own_pos[1]

        # Prioritize larger delta
        if abs(dx) > abs(dy):
            return Direction.EAST if dx > 0 else Direction.WEST
        else:
            return Direction.SOUTH if dy > 0 else Direction.NORTH

    def _choose_flee_direction(
        self, own_pos: tuple[int, int], visible_agents: list
    ) -> Direction:
        """Choose direction to flee away from threats."""
        if not visible_agents:
            return Direction.NORTH  # Default

        # Flee away from nearest agent
        nearest = self._find_nearest_agent(own_pos, visible_agents)
        if nearest:
            dx = own_pos[0] - nearest.position[0]
            dy = own_pos[1] - nearest.position[1]

            if abs(dx) > abs(dy):
                return Direction.EAST if dx > 0 else Direction.WEST
            else:
                return Direction.SOUTH if dy > 0 else Direction.NORTH

        return Direction.NORTH

    def _has_resources(self, inventory: dict[str, int]) -> bool:
        """Check if inventory has any resources."""
        return any(qty > 0 for qty in inventory.values())


# Plugin metadata
PLUGIN_INFO = {
    "name": "Aggressive Raider Strategy",
    "version": "1.0.0",
    "author": "AUTOCOG",
    "description": "High-risk combat strategy that targets agents and steals resources",
    "strategy_name": "aggressive_raider",
}
