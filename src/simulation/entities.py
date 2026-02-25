"""Entities in the simulation: Agent and their needs.

The Agent represents the cognitive loop entity with physiological needs
that must be managed for survival.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.identity import AgentID, AgentProfile


@dataclass
class AgentNeeds:
    """Physiological needs tracking for the agent.

    All values range from 0-100 except temperature (Celsius).
    Critical thresholds cause health damage.
    """

    hunger: float = 80.0  # 0 = starving, 100 = full
    thirst: float = 80.0  # 0 = dehydrated, 100 = hydrated
    energy: float = 90.0  # 0 = exhausted, 100 = rested
    health: float = 100.0  # 0 = dead, 100 = healthy
    temperature: float = 37.0  # body temp (Celsius). 35-39 is safe range.

    def is_alive(self) -> bool:
        """Check if the agent is still alive."""
        return self.health > 0 and self.hunger > 0 and self.thirst > 0

    def most_urgent_need(self) -> str:
        """Identify which need is most critical (lowest value).

        Returns:
            Name of the most urgent need: "hunger", "thirst", or "energy"
        """
        needs = {
            "hunger": self.hunger,
            "thirst": self.thirst,
            "energy": self.energy,
        }
        return min(needs, key=lambda k: needs[k])

    def urgency(self, need: str) -> float:
        """Calculate urgency level for a specific need.

        Args:
            need: Name of the need to check

        Returns:
            Urgency from 0.0 (satisfied) to 1.0 (critical)
        """
        value: float = getattr(self, need)
        return max(0.0, 1.0 - value / 100.0)

    def decay(self, hunger_rate: float, thirst_rate: float, energy_rate: float) -> None:
        """Apply per-tick decay to needs and damage health if critical.

        Args:
            hunger_rate: Amount to decrease hunger
            thirst_rate: Amount to decrease thirst
            energy_rate: Amount to decrease energy
        """
        self.hunger = max(0.0, self.hunger - hunger_rate)
        self.thirst = max(0.0, self.thirst - thirst_rate)
        self.energy = max(0.0, self.energy - energy_rate)

        # Health damage from critical needs
        if self.hunger <= 0:
            self.health = max(0.0, self.health - 2.0)
        if self.thirst <= 0:
            self.health = max(0.0, self.health - 3.0)
        if self.energy <= 0:
            self.health = max(0.0, self.health - 0.5)


@dataclass
class Agent:
    """The cognitive agent that must survive in the world."""

    x: int = 32  # start center of world
    y: int = 32
    needs: AgentNeeds = field(default_factory=AgentNeeds)
    inventory: dict[str, int] = field(default_factory=dict)
    agent_id: AgentID = field(
        default_factory=lambda: __import__("src.agents.identity", fromlist=["AgentID"]).AgentID()
    )
    profile: AgentProfile | None = None
    alive: bool = True
    ticks_alive: int = 0
    color: str = "white"

    def add_item(self, item: str, count: int = 1) -> None:
        """Add items to the agent's inventory.

        Args:
            item: Name of the item
            count: Number to add (default 1)
        """
        self.inventory[item] = self.inventory.get(item, 0) + count

    def remove_item(self, item: str, count: int = 1) -> bool:
        """Remove items from the agent's inventory.

        Args:
            item: Name of the item
            count: Number to remove (default 1)

        Returns:
            True if items were removed, False if insufficient quantity
        """
        if self.inventory.get(item, 0) >= count:
            self.inventory[item] -= count
            if self.inventory[item] <= 0:
                del self.inventory[item]
            return True
        return False

    def has_item(self, item: str, count: int = 1) -> bool:
        """Check if the agent has at least `count` of an item.

        Args:
            item: Name of the item
            count: Minimum quantity required (default 1)

        Returns:
            True if agent has sufficient quantity
        """
        return self.inventory.get(item, 0) >= count

    def die(self) -> None:
        """Mark the agent as dead."""
        self.alive = False
