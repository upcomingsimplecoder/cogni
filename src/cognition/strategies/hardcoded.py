"""Hardcoded reactive strategy — wraps existing HardcodedCognitiveLoop logic."""

from __future__ import annotations

import math

from src.awareness.types import (
    Expression,
    Intention,
    Reflection,
    Sensation,
)
from src.simulation.actions import Action, ActionType, Direction

# Map needs to what satisfies them (same as in loop.py)
NEED_TO_RESOURCE = {
    "hunger": "berry_bush",
    "thirst": "water_source",
}

NEED_TO_ITEM = {
    "hunger": "berries",
    "thirst": "water",
}

NEED_TO_ACTION = {
    "hunger": ActionType.EAT,
    "thirst": ActionType.DRINK,
}


class HardcodedStrategy:
    """Priority-based reactive strategy matching original HardcodedCognitiveLoop.

    This is the known-working baseline. Personality traits are IGNORED.
    Decision priority:
    1. Rest if energy critically low
    2. Consume from inventory if available
    3. Gather if on matching resource
    4. Move toward nearest matching resource
    5. Rest if energy is the urgent need
    6. Explore
    """

    def form_intention(self, sensation: Sensation, reflection: Reflection) -> Intention:
        """Map hardcoded priority logic to an Intention."""
        needs = sensation.own_needs

        # Priority 1: Rest if energy critical
        if needs.get("energy", 0) < 15:
            return Intention(
                primary_goal="rest",
                planned_actions=["rest"],
                confidence=0.9,
            )

        # Find most urgent need
        survival_needs = {k: v for k, v in needs.items() if k in ("hunger", "thirst", "energy")}
        urgent_need = (
            min(survival_needs, key=lambda k: survival_needs[k]) if survival_needs else "hunger"
        )

        # Priority 2: Consume from inventory
        item_needed = NEED_TO_ITEM.get(urgent_need)
        if item_needed and sensation.own_inventory.get(item_needed, 0) > 0:
            return Intention(
                primary_goal=f"consume_{urgent_need}",
                planned_actions=[f"eat_or_drink_{item_needed}"],
                confidence=0.9,
            )

        # Priority 3-5: Find and gather resource
        resource_kind = NEED_TO_RESOURCE.get(urgent_need)
        if resource_kind:
            # Check if standing on matching resource
            for tile in sensation.visible_tiles:
                if (tile.x, tile.y) == sensation.own_position:
                    for kind, qty in tile.resources:
                        if kind == resource_kind and qty > 0:
                            return Intention(
                                primary_goal=f"gather_{urgent_need}",
                                target_position=sensation.own_position,
                                planned_actions=["gather"],
                                confidence=0.85,
                            )

            # Check adjacent water tiles (special case: can't walk on water)
            if urgent_need == "thirst":
                ax, ay = sensation.own_position
                for tile in sensation.visible_tiles:
                    if tile.tile_type == "water":
                        dist = abs(tile.x - ax) + abs(tile.y - ay)
                        if dist == 1:
                            for kind, qty in tile.resources:
                                if kind == "water_source" and qty > 0:
                                    return Intention(
                                        primary_goal="drink_adjacent_water",
                                        target_position=(tile.x, tile.y),
                                        planned_actions=["drink_adjacent"],
                                        confidence=0.85,
                                    )

            # Move toward nearest resource
            target = self._find_nearest_resource(sensation, resource_kind)
            if target:
                return Intention(
                    primary_goal=f"seek_{urgent_need}",
                    target_position=target,
                    planned_actions=["move_toward_resource"],
                    confidence=0.6,
                )

        # Priority 6: Rest if energy is the issue
        if urgent_need == "energy":
            return Intention(
                primary_goal="rest",
                planned_actions=["rest"],
                confidence=0.7,
            )

        # Fallback: explore
        return Intention(
            primary_goal="explore",
            planned_actions=["move_explore"],
            confidence=0.3,
        )

    def express(
        self, sensation: Sensation, reflection: Reflection, intention: Intention
    ) -> Expression:
        """Convert intention to concrete action."""
        match intention.primary_goal:
            case "rest":
                return Expression(action=Action(type=ActionType.REST))

            case goal if goal.startswith("consume_"):
                need = goal.removeprefix("consume_")
                action_type = NEED_TO_ACTION.get(need, ActionType.WAIT)
                return Expression(action=Action(type=action_type))

            case goal if goal.startswith("gather_"):
                need = goal.removeprefix("gather_")
                resource_kind = NEED_TO_RESOURCE.get(need)
                return Expression(action=Action(type=ActionType.GATHER, target=resource_kind))

            case "drink_adjacent_water":
                # Special case: gather water from adjacent tile then drink
                return Expression(action=Action(type=ActionType.DRINK))

            case goal if goal.startswith("seek_"):
                if intention.target_position:
                    direction = self._direction_toward(
                        sensation.own_position, intention.target_position
                    )
                    if direction:
                        return Expression(action=Action(type=ActionType.MOVE, direction=direction))
                return Expression(action=Action(type=ActionType.WAIT))

            case "explore":
                return self._express_explore(sensation)

            case _:
                return Expression(action=Action(type=ActionType.WAIT))

    def _find_nearest_resource(
        self, sensation: Sensation, resource_kind: str
    ) -> tuple[int, int] | None:
        """Find nearest tile with the desired resource."""
        ax, ay = sensation.own_position
        best_dist = float("inf")
        best_pos = None

        for tile in sensation.visible_tiles:
            if tile.tile_type == "water" and resource_kind != "water_source":
                continue
            for kind, qty in tile.resources:
                if kind == resource_kind and qty > 0:
                    dist = math.sqrt((tile.x - ax) ** 2 + (tile.y - ay) ** 2)
                    if 0 < dist < best_dist:
                        best_dist = dist
                        best_pos = (tile.x, tile.y)

        return best_pos

    def _direction_toward(
        self, from_pos: tuple[int, int], to_pos: tuple[int, int]
    ) -> Direction | None:
        """Get cardinal direction toward target."""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]

        if dx == 0 and dy == 0:
            return None

        if abs(dx) >= abs(dy):
            return Direction.EAST if dx > 0 else Direction.WEST
        else:
            return Direction.SOUTH if dy > 0 else Direction.NORTH

    def _express_explore(self, sensation: Sensation) -> Expression:
        """Move in a direction to explore. Avoid water."""
        # Use tick to vary direction
        directions = list(Direction)
        idx = sensation.tick % len(directions)

        ax, ay = sensation.own_position
        for i in range(len(directions)):
            d = directions[(idx + i) % len(directions)]
            dx, dy = d.value
            target_x, target_y = ax + dx, ay + dy
            # Check if target tile is passable
            for tile in sensation.visible_tiles:
                if (tile.x, tile.y) == (target_x, target_y) and tile.tile_type != "water":
                    return Expression(action=Action(type=ActionType.MOVE, direction=d))

        return Expression(action=Action(type=ActionType.WAIT))
