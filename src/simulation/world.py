"""World grid and tile-based environment for the survival simulation.

Provides a 64x64 tile-based world with different biomes (grass, forest, water, rock)
and regenerating resources (berries, wood, stone, water).
"""

from __future__ import annotations

import enum
import random
from dataclasses import dataclass, field


class TileType(enum.Enum):
    """Types of terrain tiles in the world."""

    GRASS = "grass"
    FOREST = "forest"
    WATER = "water"
    ROCK = "rock"
    SHELTER = "shelter"  # agent-built


@dataclass
class Resource:
    """A harvestable resource on a tile with regeneration capability."""

    kind: str  # "berry_bush", "wood", "stone", "water_source"
    quantity: int
    max_quantity: int
    regen_rate: float  # probability per tick of +1

    def try_regen(self) -> None:
        """Attempt to regenerate +1 resource based on regen_rate probability."""
        if self.quantity < self.max_quantity and random.random() < self.regen_rate:
            self.quantity += 1

    def harvest(self, amount: int = 1) -> int:
        """Harvest up to `amount` resources, returning actual amount taken."""
        taken = min(amount, self.quantity)
        self.quantity -= taken
        return taken


@dataclass
class Tile:
    """A single tile in the world grid."""

    type: TileType
    resources: list[Resource] = field(default_factory=list)
    x: int = 0
    y: int = 0
    occupants: set = field(default_factory=set)


class World:
    """The game world containing a grid of tiles with resources."""

    def __init__(self, width: int, height: int, seed: int = 42):
        """Initialize a new world with the given dimensions and random seed.

        Args:
            width: Width of the world grid
            height: Height of the world grid
            seed: Random seed for reproducible world generation
        """
        self.width = width
        self.height = height
        self.seed = seed
        self._rng = random.Random(seed)
        self.tiles: list[list[Tile]] = []
        self._generate()

    def _generate(self) -> None:
        """Generate terrain with coherent biomes and resources.

        Distribution: ~50% grass, ~25% forest, ~15% water, ~10% rock
        """
        for y in range(self.height):
            row = []
            for x in range(self.width):
                r = self._rng.random()
                if r < 0.50:
                    tile_type = TileType.GRASS
                elif r < 0.75:
                    tile_type = TileType.FOREST
                elif r < 0.90:
                    tile_type = TileType.WATER
                else:
                    tile_type = TileType.ROCK

                tile = Tile(type=tile_type, x=x, y=y)

                # Add resources based on tile type
                if tile_type == TileType.FOREST:
                    tile.resources.append(Resource("wood", self._rng.randint(3, 8), 10, 0.05))
                    if self._rng.random() < 0.4:
                        tile.resources.append(
                            Resource("berry_bush", self._rng.randint(2, 5), 5, 0.1)
                        )
                elif tile_type == TileType.GRASS:
                    if self._rng.random() < 0.15:
                        tile.resources.append(
                            Resource("berry_bush", self._rng.randint(1, 3), 3, 0.1)
                        )
                elif tile_type == TileType.WATER:
                    tile.resources.append(
                        Resource("water_source", 99, 99, 1.0)
                    )  # effectively infinite
                elif tile_type == TileType.ROCK:
                    tile.resources.append(Resource("stone", self._rng.randint(5, 15), 15, 0.02))

                row.append(tile)
            self.tiles.append(row)

    def get_tile(self, x: int, y: int) -> Tile | None:
        """Get the tile at the given coordinates, or None if out of bounds."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.tiles[y][x]
        return None

    def get_tiles_in_radius(self, cx: int, cy: int, radius: int) -> list[Tile]:
        """Get all tiles within a circular radius of the given center point.

        Args:
            cx: Center x coordinate
            cy: Center y coordinate
            radius: Radius in tiles (circular, not square)

        Returns:
            List of tiles within the radius
        """
        tiles = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:  # circular radius
                    tile = self.get_tile(cx + dx, cy + dy)
                    if tile is not None:
                        tiles.append(tile)
        return tiles

    def tick_resources(self) -> None:
        """Regenerate resources across the world for one tick."""
        for row in self.tiles:
            for tile in row:
                for resource in tile.resources:
                    resource.try_regen()

    def place_agent(self, agent_id: object, pos: tuple[int, int]) -> None:
        """Place an agent on a tile."""
        tile = self.get_tile(*pos)
        if tile is not None:
            tile.occupants.add(agent_id)

    def remove_agent(self, agent_id: object, pos: tuple[int, int]) -> None:
        """Remove an agent from a tile."""
        tile = self.get_tile(*pos)
        if tile is not None:
            tile.occupants.discard(agent_id)

    def move_agent(
        self, agent_id: object, from_pos: tuple[int, int], to_pos: tuple[int, int]
    ) -> None:
        """Update tile occupancy when an agent moves."""
        self.remove_agent(agent_id, from_pos)
        self.place_agent(agent_id, to_pos)

    def agents_on_tile(self, x: int, y: int) -> set:
        """Get all agent IDs on a tile."""
        tile = self.get_tile(x, y)
        return set(tile.occupants) if tile is not None else set()

    def agents_in_radius(self, cx: int, cy: int, radius: int) -> set:
        """Get all agent IDs within radius."""
        agents: set = set()
        for tile in self.get_tiles_in_radius(cx, cy, radius):
            agents.update(tile.occupants)
        return agents
