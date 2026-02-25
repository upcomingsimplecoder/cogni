"""Tests for simulation actions: execute_action for all implemented types."""

from __future__ import annotations

import pytest

from src.simulation.actions import (
    Action,
    ActionType,
    Direction,
    execute_action,
)
from src.simulation.entities import Agent, AgentNeeds
from src.simulation.world import TileType, World


class MockRegistry:
    """Minimal registry mock for GIVE and ATTACK actions."""

    def __init__(self):
        self._agents = {}

    def get(self, agent_id):
        return self._agents.get(agent_id)

    def add(self, agent):
        self._agents[agent.agent_id] = agent


class TestMoveAction:
    """Tests for MOVE action."""

    def test_move_updates_position_and_costs_energy(self):
        """Successful move updates agent position and costs 1 energy."""
        world = World(16, 16, seed=42)

        # Find a non-water position with room to move north (not at y=0)
        pos = None
        for y in range(1, world.height):  # start from y=1 to allow north movement
            for x in range(world.width):
                tile = world.get_tile(x, y)
                north_tile = world.get_tile(x, y - 1)
                if (
                    tile
                    and tile.type != TileType.WATER
                    and north_tile
                    and north_tile.type != TileType.WATER
                ):
                    pos = (x, y)
                    break
            if pos:
                break

        assert pos is not None, "No valid position found for movement test"

        agent = Agent(x=pos[0], y=pos[1], needs=AgentNeeds(energy=50))

        action = Action(type=ActionType.MOVE, direction=Direction.NORTH)
        result = execute_action(action, agent, world)

        assert result.success is True
        assert agent.y == pos[1] - 1  # moved north
        assert agent.x == pos[0]  # x unchanged
        assert result.needs_delta == {"energy": -1.0}

    def test_move_fails_for_no_direction(self):
        """MOVE fails when no direction specified."""
        world = World(16, 16, seed=42)
        agent = Agent(x=8, y=8)

        action = Action(type=ActionType.MOVE, direction=None)
        result = execute_action(action, agent, world)

        assert result.success is False
        assert "No direction specified" in result.message

    def test_move_fails_when_out_of_bounds(self):
        """MOVE fails when moving out of world bounds."""
        world = World(16, 16, seed=42)
        agent = Agent(x=0, y=0)

        action = Action(type=ActionType.MOVE, direction=Direction.WEST)
        result = execute_action(action, agent, world)

        assert result.success is False
        assert "out of bounds" in result.message
        assert agent.x == 0  # position unchanged

    def test_move_fails_when_moving_onto_water(self):
        """MOVE fails when target tile is water."""
        world = World(16, 16, seed=42)

        # Find a water tile adjacent to a non-water tile
        water_pos = None
        adjacent_land_pos = None
        direction_to_water = None

        for y in range(1, world.height - 1):
            for x in range(1, world.width - 1):
                tile = world.get_tile(x, y)
                if tile.type != TileType.WATER:
                    # Check adjacent tiles for water
                    north = world.get_tile(x, y - 1)
                    if north and north.type == TileType.WATER:
                        adjacent_land_pos = (x, y)
                        water_pos = (x, y - 1)
                        direction_to_water = Direction.NORTH
                        break
                if water_pos:
                    break
            if water_pos:
                break

        # If no water found in this seed, skip test
        if not water_pos:
            pytest.skip("No water tiles found in test world with seed 42")

        agent = Agent(x=adjacent_land_pos[0], y=adjacent_land_pos[1])
        action = Action(type=ActionType.MOVE, direction=direction_to_water)
        result = execute_action(action, agent, world)

        assert result.success is False
        assert "Cannot walk on water" in result.message
        assert (agent.x, agent.y) == adjacent_land_pos  # position unchanged


class TestGatherAction:
    """Tests for GATHER action."""

    def test_gather_adds_to_inventory(self):
        """GATHER collects resource from tile and adds to inventory."""
        world = World(16, 16, seed=42)

        # Find a tile with berries
        berry_tile = None
        for row in world.tiles:
            for tile in row:
                for resource in tile.resources:
                    if resource.kind == "berry_bush" and resource.quantity > 0:
                        berry_tile = tile
                        break
                if berry_tile:
                    break
            if berry_tile:
                break

        if not berry_tile:
            pytest.skip("No berry tiles found in test world")

        agent = Agent(x=berry_tile.x, y=berry_tile.y)
        action = Action(type=ActionType.GATHER, target="berry_bush")

        _initial_quantity = None
        for res in berry_tile.resources:
            if res.kind == "berry_bush":
                _initial_quantity = res.quantity
                break

        result = execute_action(action, agent, world)

        assert result.success is True
        assert "berries" in agent.inventory
        assert agent.inventory["berries"] == 1
        assert result.inventory_delta == {"berries": 1}
        assert result.needs_delta == {"energy": -0.5}

    def test_gather_fails_when_nothing_to_gather(self):
        """GATHER fails when tile and adjacent tiles have no resources."""
        world = World(16, 16, seed=42)

        # Place agent on a tile and clear all resources from it AND adjacent tiles
        agent = Agent(x=8, y=8)
        world.tiles[8][8].resources = []
        for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
            adj = world.get_tile(8 + dx, 8 + dy)
            if adj:
                adj.resources = []

        action = Action(type=ActionType.GATHER)
        result = execute_action(action, agent, world)

        assert result.success is False
        assert "Nothing to gather" in result.message


class TestEatAction:
    """Tests for EAT action."""

    def test_eat_consumes_berries_and_restores_hunger(self):
        """EAT consumes berries from inventory and restores hunger."""
        agent = Agent()
        agent.add_item("berries", 5)
        agent.needs.hunger = 50.0

        action = Action(type=ActionType.EAT)
        result = execute_action(action, agent, None)

        assert result.success is True
        assert agent.inventory["berries"] == 4
        assert result.needs_delta == {"hunger": 15.0}
        assert result.inventory_delta == {"berries": -1}

    def test_eat_fails_with_no_food(self):
        """EAT fails when no berries in inventory."""
        agent = Agent()

        action = Action(type=ActionType.EAT)
        result = execute_action(action, agent, None)

        assert result.success is False
        assert "No food in inventory" in result.message


class TestDrinkAction:
    """Tests for DRINK action."""

    def test_drink_consumes_water_and_restores_thirst(self):
        """DRINK consumes water from inventory and restores thirst."""
        agent = Agent()
        agent.add_item("water", 3)
        agent.needs.thirst = 40.0

        action = Action(type=ActionType.DRINK)
        result = execute_action(action, agent, None)

        assert result.success is True
        assert agent.inventory["water"] == 2
        assert result.needs_delta == {"thirst": 20.0}
        assert result.inventory_delta == {"water": -1}

    def test_drink_fails_with_no_water(self):
        """DRINK fails when no water in inventory."""
        agent = Agent()

        action = Action(type=ActionType.DRINK)
        result = execute_action(action, agent, None)

        assert result.success is False
        assert "No water in inventory" in result.message


class TestRestAction:
    """Tests for REST action."""

    def test_rest_restores_energy(self):
        """REST restores energy."""
        agent = Agent()
        agent.needs.energy = 30.0

        action = Action(type=ActionType.REST)
        result = execute_action(action, agent, None)

        assert result.success is True
        assert result.needs_delta == {"energy": 10.0}
        assert "Rested" in result.message


class TestWaitAction:
    """Tests for WAIT action."""

    def test_wait_always_succeeds(self):
        """WAIT always succeeds."""
        agent = Agent()

        action = Action(type=ActionType.WAIT)
        result = execute_action(action, agent, None)

        assert result.success is True
        assert "Waited" in result.message


class TestGiveAction:
    """Tests for GIVE action."""

    def test_give_transfers_item_to_adjacent_agent(self):
        """GIVE transfers item to adjacent agent."""
        world = World(16, 16, seed=42)
        registry = MockRegistry()

        agent1 = Agent(x=8, y=8)
        agent1.add_item("wood", 5)
        registry.add(agent1)

        agent2 = Agent(x=9, y=8)  # adjacent (east)
        registry.add(agent2)

        action = Action(
            type=ActionType.GIVE,
            target="wood",
            target_agent_id=agent2.agent_id,
            quantity=2,
        )
        result = execute_action(action, agent1, world, registry)

        assert result.success is True
        assert agent1.inventory["wood"] == 3
        assert agent2.inventory["wood"] == 2
        assert result.inventory_delta == {"wood": -2}

    def test_give_fails_when_target_too_far(self):
        """GIVE fails when target agent is not adjacent."""
        world = World(16, 16, seed=42)
        registry = MockRegistry()

        agent1 = Agent(x=8, y=8)
        agent1.add_item("wood", 5)
        registry.add(agent1)

        agent2 = Agent(x=12, y=12)  # too far
        registry.add(agent2)

        action = Action(
            type=ActionType.GIVE,
            target="wood",
            target_agent_id=agent2.agent_id,
            quantity=1,
        )
        result = execute_action(action, agent1, world, registry)

        assert result.success is False
        assert "too far away" in result.message

    def test_give_fails_when_no_target_specified(self):
        """GIVE fails when no target specified."""
        world = World(16, 16, seed=42)
        registry = MockRegistry()

        agent = Agent(x=8, y=8)
        agent.add_item("wood", 5)

        action = Action(type=ActionType.GIVE, target="wood")
        result = execute_action(action, agent, world, registry)

        assert result.success is False
        assert "No target agent specified" in result.message

    def test_give_fails_when_item_not_in_inventory(self):
        """GIVE fails when item not in inventory."""
        world = World(16, 16, seed=42)
        registry = MockRegistry()

        agent1 = Agent(x=8, y=8)
        registry.add(agent1)

        agent2 = Agent(x=9, y=8)
        registry.add(agent2)

        action = Action(
            type=ActionType.GIVE,
            target="gold",
            target_agent_id=agent2.agent_id,
            quantity=1,
        )
        result = execute_action(action, agent1, world, registry)

        assert result.success is False
        assert "Don't have that item" in result.message


class TestAttackAction:
    """Tests for ATTACK action."""

    def test_attack_damages_colocated_agent(self):
        """ATTACK damages agent on same tile."""
        world = World(16, 16, seed=42)
        registry = MockRegistry()

        agent1 = Agent(x=8, y=8, needs=AgentNeeds(health=100, energy=50))
        registry.add(agent1)

        agent2 = Agent(x=8, y=8, needs=AgentNeeds(health=100))
        registry.add(agent2)

        action = Action(type=ActionType.ATTACK, target_agent_id=agent2.agent_id)
        result = execute_action(action, agent1, world, registry)

        assert result.success is True
        assert agent2.needs.health == 85.0  # took 15 damage
        assert result.needs_delta == {"energy": -5.0, "health": -2.0}
        assert "Dealt 15 damage" in result.message

    def test_attack_fails_when_target_not_on_same_tile(self):
        """ATTACK fails when target not on same tile."""
        world = World(16, 16, seed=42)
        registry = MockRegistry()

        agent1 = Agent(x=8, y=8)
        registry.add(agent1)

        agent2 = Agent(x=9, y=8)  # adjacent but not same tile
        registry.add(agent2)

        action = Action(type=ActionType.ATTACK, target_agent_id=agent2.agent_id)
        result = execute_action(action, agent1, world, registry)

        assert result.success is False
        assert "not on same tile" in result.message

    def test_attack_fails_when_no_target_specified(self):
        """ATTACK fails when no target specified."""
        world = World(16, 16, seed=42)
        registry = MockRegistry()

        agent = Agent(x=8, y=8)

        action = Action(type=ActionType.ATTACK)
        result = execute_action(action, agent, world, registry)

        assert result.success is False
        assert "No target for attack" in result.message


class TestSendMessageAction:
    """Tests for SEND_MESSAGE action."""

    def test_send_message_succeeds_and_costs_energy(self):
        """SEND_MESSAGE always succeeds and costs energy."""
        agent = Agent()

        action = Action(type=ActionType.SEND_MESSAGE, target="Hello!")
        result = execute_action(action, agent, None)

        assert result.success is True
        assert result.needs_delta == {"energy": -0.5}
        assert "Message queued" in result.message
