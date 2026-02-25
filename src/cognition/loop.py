from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.simulation.engine import SimulationEngine

from src.simulation.actions import Action, ActionType, Direction
from src.simulation.world import TileType

# Map needs to what satisfies them
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


class HardcodedCognitiveLoop:
    """Phase 1 cognitive loop — simple reactive behavior.

    Strategy:
    1. If most urgent need can be satisfied from inventory → consume
    2. If standing on a matching resource → gather
    3. Otherwise → move toward nearest matching resource
    4. If energy is critically low → rest
    5. Fallback → wait
    """

    def decide(self, engine: SimulationEngine) -> Action:
        agent = engine.agent
        if agent is None:
            return Action(type=ActionType.WAIT)

        world = engine.world
        needs = agent.needs

        # Priority 1: Rest if energy is critically low
        if needs.energy < 15:
            return Action(type=ActionType.REST)

        # Find most urgent need
        urgent_need = needs.most_urgent_need()

        # Priority 2: Consume from inventory if possible
        item_needed = NEED_TO_ITEM.get(urgent_need)
        if item_needed and agent.has_item(item_needed):
            action_type = NEED_TO_ACTION[urgent_need]
            return Action(type=action_type)

        # Priority 3: Gather if standing on matching resource
        current_tile = world.get_tile(agent.x, agent.y)
        resource_kind = NEED_TO_RESOURCE.get(urgent_need)
        if current_tile and resource_kind:
            for resource in current_tile.resources:
                if resource.kind == resource_kind and resource.quantity > 0:
                    return Action(type=ActionType.GATHER, target=resource_kind)

        # Priority 4: Also gather water if on water-adjacent tile (can't walk on water)
        if urgent_need == "thirst":
            # Check adjacent tiles for water
            for direction in Direction:
                dx, dy = direction.value
                adj_tile = world.get_tile(agent.x + dx, agent.y + dy)
                if adj_tile and adj_tile.type == TileType.WATER:
                    # Gather water from adjacent water tile
                    for resource in adj_tile.resources:
                        if resource.kind == "water_source" and resource.quantity > 0:
                            resource.harvest(1)
                            agent.add_item("water", 1)
                            return Action(type=ActionType.DRINK)

        # Priority 5: Move toward nearest matching resource
        if resource_kind:
            found_direction = self._find_direction_to_resource(
                agent.x, agent.y, resource_kind, world, engine.config.vision_radius
            )
            if found_direction is not None:
                return Action(type=ActionType.MOVE, direction=found_direction)

        # Priority 6: If we need energy, just gather whatever is around (wood for later)
        if urgent_need == "energy":
            return Action(type=ActionType.REST)

        # Fallback: move in a random-ish direction to explore
        return self._explore(agent.x, agent.y, world, engine.state.tick)

    def _find_direction_to_resource(
        self, ax: int, ay: int, resource_kind: str, world, radius: int
    ) -> Direction | None:
        """Find direction toward the nearest tile with the desired resource."""
        best_dist = float("inf")
        best_tile = None

        tiles = world.get_tiles_in_radius(ax, ay, radius)
        for tile in tiles:
            # Skip water tiles for movement (except if looking for water, check adjacent)
            if tile.type == TileType.WATER and resource_kind != "water_source":
                continue
            for resource in tile.resources:
                if resource.kind == resource_kind and resource.quantity > 0:
                    dist = math.sqrt((tile.x - ax) ** 2 + (tile.y - ay) ** 2)
                    if dist < best_dist and dist > 0:
                        best_dist = dist
                        best_tile = tile

        if best_tile is None:
            return None

        # Pick the cardinal direction that moves us closest
        dx = best_tile.x - ax
        dy = best_tile.y - ay

        if abs(dx) >= abs(dy):
            return Direction.EAST if dx > 0 else Direction.WEST
        else:
            return Direction.SOUTH if dy > 0 else Direction.NORTH

    def _explore(self, ax: int, ay: int, world, tick: int) -> Action:
        """Move in a direction based on tick count to avoid getting stuck."""
        directions = list(Direction)
        direction = directions[tick % len(directions)]
        # Check if we can move there
        dx, dy = direction.value
        tile = world.get_tile(ax + dx, ay + dy)
        if tile and tile.type != TileType.WATER:
            return Action(type=ActionType.MOVE, direction=direction)
        # Try next direction
        for d in directions:
            ddx, ddy = d.value
            t = world.get_tile(ax + ddx, ay + ddy)
            if t and t.type != TileType.WATER:
                return Action(type=ActionType.MOVE, direction=d)
        return Action(type=ActionType.WAIT)
