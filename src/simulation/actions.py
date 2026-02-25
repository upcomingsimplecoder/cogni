"""Action definitions and execution for the survival simulation.

This module defines all available actions an agent can take, along with the
logic to execute them and compute their effects on the agent and world.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.simulation.entities import Agent
    from src.simulation.world import World


class ActionType(enum.Enum):
    """All possible action types in the simulation."""

    MOVE = "move"
    GATHER = "gather"
    EAT = "eat"
    DRINK = "drink"
    REST = "rest"
    BUILD_SHELTER = "build_shelter"
    CRAFT = "craft"
    LIGHT_FIRE = "light_fire"
    FLEE = "flee"
    FIGHT = "fight"
    SCOUT = "scout"
    WAIT = "wait"
    STORE = "store"
    GIVE = "give"
    SEND_MESSAGE = "send_message"
    ATTACK = "attack"


class Direction(enum.Enum):
    """Cardinal directions for movement."""

    NORTH = (0, -1)
    SOUTH = (0, 1)
    EAST = (1, 0)
    WEST = (-1, 0)


@dataclass
class Action:
    """An action to be executed by an agent."""

    type: ActionType
    target: str | None = None  # resource kind, item name, etc.
    direction: Direction | None = None  # for MOVE/FLEE
    target_agent_id: object | None = None
    quantity: int = 1


@dataclass
class ActionResult:
    """The result of executing an action."""

    action: Action
    success: bool
    message: str
    needs_delta: dict[str, float] | None = None  # changes to needs
    inventory_delta: dict[str, int] | None = None  # changes to inventory


def execute_action(
    action: Action, agent: Agent, world: World, registry: object | None = None
) -> ActionResult:
    """Execute an action and return the result."""
    match action.type:
        case ActionType.MOVE:
            return _execute_move(action, agent, world)
        case ActionType.GATHER:
            return _execute_gather(action, agent, world)
        case ActionType.EAT:
            return _execute_eat(action, agent)
        case ActionType.DRINK:
            return _execute_drink(action, agent)
        case ActionType.REST:
            return _execute_rest(action, agent)
        case ActionType.WAIT:
            return ActionResult(action=action, success=True, message="Waited.")
        case ActionType.GIVE:
            return _execute_give(action, agent, world, registry)
        case ActionType.ATTACK:
            return _execute_attack(action, agent, world, registry)
        case ActionType.SEND_MESSAGE:
            return _execute_send_message(action, agent)
        case _:
            return ActionResult(
                action=action,
                success=False,
                message=f"Action {action.type.value} not yet implemented.",
            )


def _execute_move(action: Action, agent: Agent, world: World) -> ActionResult:
    """Execute a MOVE action."""
    if action.direction is None:
        return ActionResult(action=action, success=False, message="No direction specified.")
    dx, dy = action.direction.value
    new_x, new_y = agent.x + dx, agent.y + dy
    tile = world.get_tile(new_x, new_y)
    if tile is None:
        return ActionResult(action=action, success=False, message="Cannot move out of bounds.")
    # Water tiles block movement (can't walk on water)
    from src.simulation.world import TileType

    if tile.type == TileType.WATER:
        return ActionResult(action=action, success=False, message="Cannot walk on water.")
    agent.x, agent.y = new_x, new_y
    # Moving costs energy
    return ActionResult(
        action=action,
        success=True,
        message=f"Moved {action.direction.name.lower()} to ({new_x}, {new_y}).",
        needs_delta={"energy": -1.0},
    )


def _execute_gather(action: Action, agent: Agent, world: World) -> ActionResult:
    """Execute a GATHER action.

    Checks the agent's tile first, then adjacent tiles. This allows
    gathering water from adjacent water tiles (agents can't stand on water).
    """
    target_kind = action.target

    # Check agent's own tile first
    tile = world.get_tile(agent.x, agent.y)
    if tile is not None:
        result = _try_harvest_from_tile(action, agent, tile, target_kind)
        if result is not None:
            return result

    # Check adjacent tiles (allows gathering water from shoreline)
    for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
        adj = world.get_tile(agent.x + dx, agent.y + dy)
        if adj is not None:
            result = _try_harvest_from_tile(action, agent, adj, target_kind)
            if result is not None:
                return result

    return ActionResult(action=action, success=False, message="Nothing to gather here or nearby.")


def _try_harvest_from_tile(
    action: Action, agent: Agent, tile: object, target_kind: str | None
) -> ActionResult | None:
    """Try to harvest a resource from a tile. Returns None if nothing found."""
    if not hasattr(tile, "resources"):
        return None
    for resource in tile.resources:
        if target_kind and resource.kind != target_kind:
            continue
        if resource.quantity > 0:
            taken = resource.harvest(1)
            if taken > 0:
                item_name = _resource_to_item(resource.kind)
                agent.add_item(item_name, taken)
                return ActionResult(
                    action=action,
                    success=True,
                    message=f"Gathered {taken} {item_name}.",
                    inventory_delta={item_name: taken},
                    needs_delta={"energy": -0.5},
                )
    return None


def _resource_to_item(kind: str) -> str:
    """Map resource kinds to inventory item names."""
    mapping = {
        "berry_bush": "berries",
        "wood": "wood",
        "stone": "stone",
        "water_source": "water",
    }
    return mapping.get(kind, kind)


def _execute_eat(action: Action, agent: Agent) -> ActionResult:
    """Execute an EAT action."""
    # Try to eat berries from inventory
    if agent.remove_item("berries", 1):
        return ActionResult(
            action=action,
            success=True,
            message="Ate berries.",
            needs_delta={"hunger": 15.0},
            inventory_delta={"berries": -1},
        )
    return ActionResult(action=action, success=False, message="No food in inventory.")


def _execute_drink(action: Action, agent: Agent) -> ActionResult:
    """Execute a DRINK action."""
    if agent.remove_item("water", 1):
        return ActionResult(
            action=action,
            success=True,
            message="Drank water.",
            needs_delta={"thirst": 20.0},
            inventory_delta={"water": -1},
        )
    return ActionResult(action=action, success=False, message="No water in inventory.")


def _execute_rest(action: Action, agent: Agent) -> ActionResult:
    """Execute a REST action."""
    return ActionResult(
        action=action,
        success=True,
        message="Rested for a tick.",
        needs_delta={"energy": 10.0},
    )


def _execute_give(action: Action, agent: Agent, world: World, registry: object) -> ActionResult:
    """Transfer items to an adjacent or co-located agent."""
    if not action.target_agent_id or registry is None:
        return ActionResult(action=action, success=False, message="No target agent specified.")

    if not hasattr(registry, "get"):
        return ActionResult(action=action, success=False, message="Invalid registry.")

    target_agent = registry.get(action.target_agent_id)  # type: ignore[attr-defined]
    if target_agent is None:
        return ActionResult(action=action, success=False, message="Target agent not found.")

    # Check adjacency (manhattan distance <= 1)
    dist = abs(agent.x - target_agent.x) + abs(agent.y - target_agent.y)
    if dist > 1:
        return ActionResult(action=action, success=False, message="Target agent too far away.")

    item = action.target
    qty = action.quantity
    if item and agent.remove_item(item, qty):
        target_agent.add_item(item, qty)
        return ActionResult(
            action=action,
            success=True,
            message=f"Gave {qty} {item} to agent {action.target_agent_id}.",
            inventory_delta={item: -qty},
        )
    return ActionResult(action=action, success=False, message="Don't have that item to give.")


def _execute_attack(action: Action, agent: Agent, world: World, registry: object) -> ActionResult:
    """Attack a co-located agent."""
    if not action.target_agent_id or registry is None:
        return ActionResult(action=action, success=False, message="No target for attack.")

    if not hasattr(registry, "get"):
        return ActionResult(action=action, success=False, message="Invalid registry.")

    target_agent = registry.get(action.target_agent_id)  # type: ignore[attr-defined]
    if target_agent is None:
        return ActionResult(action=action, success=False, message="Target agent not found.")

    # Must be on same tile
    if (agent.x, agent.y) != (target_agent.x, target_agent.y):
        return ActionResult(action=action, success=False, message="Target not on same tile.")

    # Deal damage to target
    target_agent.needs.health = max(0.0, target_agent.needs.health - 15.0)

    return ActionResult(
        action=action,
        success=True,
        message=f"Attacked agent {action.target_agent_id}. Dealt 15 damage.",
        needs_delta={"energy": -5.0, "health": -2.0},
    )


def _execute_send_message(action: Action, agent: Agent) -> ActionResult:
    """Send a message (energy cost only — actual routing via MessageBus)."""
    return ActionResult(
        action=action,
        success=True,
        message="Message queued for delivery.",
        needs_delta={"energy": -0.5},
    )
