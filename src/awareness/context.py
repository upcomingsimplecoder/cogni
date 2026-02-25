"""Context tags for categorizing agent situations.

Provides a shared vocabulary of environmental contexts used by both
the cultural transmission system (Phase 2) and metacognition system (Phase 3).
Extracted from evolution.observation to avoid cross-module coupling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.awareness.types import Sensation


class ContextTag:
    """Context tags for categorizing behavioral observations.

    Each tag represents a salient situation in which an action was taken.
    Agents use these to retrieve relevant observations for their current context.
    """

    # Urgent needs
    LOW_HUNGER = "low_hunger"
    LOW_THIRST = "low_thirst"
    LOW_ENERGY = "low_energy"

    # Resource proximity
    NEAR_FOOD = "near_food"
    NEAR_WATER = "near_water"

    # Social contexts
    NEAR_AGENT = "near_agent"
    NEAR_HOSTILE = "near_hostile"
    NEAR_ALLY = "near_ally"

    # State contexts
    HAS_INVENTORY = "has_inventory"
    ALONE = "alone"
    CROWDED = "crowded"
    NIGHTTIME = "nighttime"

    @staticmethod
    def extract_primary(sensation: Sensation) -> str:
        """Extract the most salient context tag from a sensation.

        Priority order:
        1. Urgent needs (< 30)
        2. Social context (crowded vs near_agent vs alone)
        3. Resource context (near food/water)
        4. State context (has inventory)
        5. Default: ALONE

        Args:
            sensation: Current perception containing needs, visible agents, tiles, etc.

        Returns:
            The primary context tag string.
        """
        # Priority 1: Urgent needs
        hunger = sensation.own_needs.get("hunger", 100.0)
        thirst = sensation.own_needs.get("thirst", 100.0)
        energy = sensation.own_needs.get("energy", 100.0)

        if hunger < 30.0:
            return ContextTag.LOW_HUNGER
        if thirst < 30.0:
            return ContextTag.LOW_THIRST
        if energy < 30.0:
            return ContextTag.LOW_ENERGY

        # Priority 2: Social context
        visible_agent_count = len(sensation.visible_agents)
        if visible_agent_count >= 3:
            return ContextTag.CROWDED
        if visible_agent_count > 0:
            return ContextTag.NEAR_AGENT

        # Priority 3: Resource context
        own_x, own_y = sensation.own_position

        for tile in sensation.visible_tiles:
            dist = abs(tile.x - own_x) + abs(tile.y - own_y)
            if dist > 3:
                continue

            for kind, qty in tile.resources:
                if qty > 0:
                    if kind == "berry_bush":
                        return ContextTag.NEAR_FOOD
                    if kind == "water_source":
                        return ContextTag.NEAR_WATER

        # Priority 4: State context
        if sensation.own_inventory and sum(sensation.own_inventory.values()) > 0:
            return ContextTag.HAS_INVENTORY

        # Default
        return ContextTag.ALONE
