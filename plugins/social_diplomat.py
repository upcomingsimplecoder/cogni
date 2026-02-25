"""Social Diplomat Strategy.

This plugin implements a cooperation-first strategy that prioritizes sharing
resources, communicating with other agents, and avoiding all conflict. When
threatened, flees rather than fights. Designed for peaceful survival scenarios.

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


@PluginRegistry.register_strategy("social_diplomat")
class SocialDiplomatStrategy:
    """Cooperative diplomat that shares resources and avoids conflict.

    Decision logic:
    - If threatened (visible agents acting hostile): FLEE immediately
    - If agents nearby and have surplus resources: SHARE/GIVE
    - If agents nearby: SEND_MESSAGE to communicate
    - If needs critical: Gather resources
    - Otherwise: Gather and build surplus for sharing
    """

    def __init__(self):
        """Initialize the strategy."""
        self.flee_threshold = 0.2  # Lower threshold = more cautious
        self.sharing_threshold = 3  # Share if we have 3+ of an item
        self.critical_hunger = 30.0
        self.critical_thirst = 30.0
        self.cooperation_score = 0.9

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
        # Threat detected: flee immediately
        if reflection.threat_level > self.flee_threshold:
            return Intention(
                primary_goal="flee",
                confidence=0.95,
                planned_actions=["flee", "hide"],
            )

        # Check for hostile actions from visible agents
        for agent in sensation.visible_agents:
            if agent.last_action in ["attack", "fight"]:
                return Intention(
                    primary_goal="flee",
                    confidence=0.95,
                    planned_actions=["flee", "hide"],
                )

        # Agents nearby: prioritize social interaction
        if len(sensation.visible_agents) > 0:
            # Check if we have surplus resources to share
            shareable_items = self._find_shareable_items(sensation.own_inventory)
            if shareable_items:
                target_agent = self._find_neediest_agent(sensation.visible_agents)
                if target_agent:
                    item, qty = shareable_items[0]  # Share first available
                    return Intention(
                        primary_goal="share_resources",
                        confidence=0.85,
                        target_agent_id=target_agent.agent_id,
                        target_position=target_agent.position,
                        planned_actions=["move_to_target", "give"],
                    )

            # No surplus to share: send friendly message
            nearest_agent = self._find_nearest_agent(
                sensation.own_position, sensation.visible_agents
            )
            if nearest_agent:
                return Intention(
                    primary_goal="communicate",
                    confidence=0.75,
                    target_agent_id=nearest_agent.agent_id,
                    planned_actions=["send_message"],
                )

        # No agents around: focus on survival needs
        hunger = sensation.own_needs.get("hunger", 100.0)
        thirst = sensation.own_needs.get("thirst", 100.0)

        # Critical hunger
        if hunger < self.critical_hunger:
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

        # Critical thirst
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

        # Default: gather resources for future sharing
        return Intention(
            primary_goal="gather_surplus",
            confidence=0.7,
            planned_actions=["gather", "stockpile"],
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
                # Flee away from threats
                flee_dir = self._choose_flee_direction(
                    sensation.own_position, sensation.visible_agents
                )
                action = Action(type=ActionType.FLEE, direction=flee_dir)

            case "share_resources":
                if intention.target_agent_id:
                    # Check if we're adjacent to target
                    if self._is_adjacent(
                        sensation.own_position, intention.target_position
                    ):
                        # Give resources
                        shareable = self._find_shareable_items(sensation.own_inventory)
                        if shareable:
                            item, qty = shareable[0]
                            action = Action(
                                type=ActionType.GIVE,
                                target_agent_id=intention.target_agent_id,
                                target=item,
                                quantity=1,  # Share one at a time
                            )
                        else:
                            action = Action(type=ActionType.WAIT)
                    else:
                        # Move toward target
                        direction = self._direction_to_target(
                            sensation.own_position, intention.target_position
                        )
                        action = Action(type=ActionType.MOVE, direction=direction)
                else:
                    action = Action(type=ActionType.WAIT)

            case "communicate":
                if intention.target_agent_id:
                    action = Action(
                        type=ActionType.SEND_MESSAGE,
                        target_agent_id=intention.target_agent_id,
                    )
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

            case "gather_surplus":
                # Alternate between food and water gathering
                if sensation.own_inventory.get("berries", 0) < 5:
                    action = Action(type=ActionType.GATHER, target="berry_bush")
                else:
                    action = Action(type=ActionType.GATHER, target="water_source")

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
        visible_count = len(sensation.visible_agents)
        inventory_total = sum(sensation.own_inventory.values())

        match intention.primary_goal:
            case "flee":
                return "Danger! I need to get away peacefully. No violence."
            case "share_resources":
                return f"I have surplus ({inventory_total} items). Let me help someone nearby."
            case "communicate":
                return f"I see {visible_count} agent(s). Let me reach out and communicate."
            case "eat":
                return "Taking care of my hunger so I can help others later."
            case "drink":
                return "Staying hydrated so I can be useful to my community."
            case "gather_food":
                return "Gathering food. I'll share if anyone needs it."
            case "gather_water":
                return "Gathering water. Always good to have extra for sharing."
            case "gather_surplus":
                return "Building up resources to share with others who need help."
            case _:
                return "Observing peacefully, ready to help if needed."

    def _find_shareable_items(
        self, inventory: dict[str, int]
    ) -> list[tuple[str, int]]:
        """Find items in inventory that can be shared (surplus > threshold)."""
        shareable = []
        for item, qty in inventory.items():
            if qty >= self.sharing_threshold:
                shareable.append((item, qty))
        return shareable

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

    def _find_neediest_agent(self, visible_agents: list) -> object | None:
        """Find agent that appears most in need (heuristic: appears injured)."""
        for agent in visible_agents:
            if agent.apparent_health in ["injured", "critical"]:
                return agent
        # Default to nearest if no one appears needy
        return visible_agents[0] if visible_agents else None

    def _manhattan_distance(
        self, pos1: tuple[int, int], pos2: tuple[int, int]
    ) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _is_adjacent(
        self, own_pos: tuple[int, int], target_pos: tuple[int, int] | None
    ) -> bool:
        """Check if target position is adjacent (manhattan distance <= 1)."""
        if target_pos is None:
            return False
        return self._manhattan_distance(own_pos, target_pos) <= 1

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
            return Direction.SOUTH  # Default safe direction

        # Flee away from nearest agent
        nearest = self._find_nearest_agent(own_pos, visible_agents)
        if nearest:
            dx = own_pos[0] - nearest.position[0]
            dy = own_pos[1] - nearest.position[1]

            if abs(dx) > abs(dy):
                return Direction.EAST if dx > 0 else Direction.WEST
            else:
                return Direction.SOUTH if dy > 0 else Direction.NORTH

        return Direction.SOUTH


# Plugin metadata
PLUGIN_INFO = {
    "name": "Social Diplomat Strategy",
    "version": "1.0.0",
    "author": "AUTOCOG",
    "description": "Cooperation-first strategy that shares resources and avoids conflict",
    "strategy_name": "social_diplomat",
}
