"""Sensation module: assembles perception from world state for an agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.awareness.types import AgentSummary, Sensation, TileSummary

if TYPE_CHECKING:
    from src.simulation.entities import Agent


class SensationModule:
    """Assembles a Sensation object from world state for a specific agent.

    This is the agent's 'eyes and ears' — what it can perceive each tick.
    """

    def perceive(self, agent: Agent, engine: Any) -> Sensation:
        """Build a Sensation snapshot for this agent.

        Perceives:
        - Own needs, position, inventory, traits
        - Visible tiles (within vision_radius) with resource info
        - Visible agents (on visible tiles) with rough health assessment
        - Incoming messages (drained from mailbox)
        - Time of day
        """
        visible_tiles = self._scan_tiles(agent, engine)
        visible_agents = self._scan_agents(agent, engine, visible_tiles)
        messages = self._drain_messages(agent, engine)

        return Sensation(
            tick=engine.state.tick,
            own_needs={
                "hunger": agent.needs.hunger,
                "thirst": agent.needs.thirst,
                "energy": agent.needs.energy,
                "health": agent.needs.health,
            },
            own_position=(agent.x, agent.y),
            own_inventory=dict(agent.inventory),
            visible_tiles=visible_tiles,
            visible_agents=visible_agents,
            incoming_messages=messages,
            time_of_day=engine.state.time_of_day,
            own_traits=agent.profile.traits.as_dict()
            if agent.profile and agent.profile.traits
            else {},
        )

    def _scan_tiles(self, agent: Agent, engine: Any) -> list[TileSummary]:
        """Get visible tile summaries within vision radius."""
        raw_tiles = engine.world.get_tiles_in_radius(agent.x, agent.y, engine.config.vision_radius)
        summaries = []
        for t in raw_tiles:
            summaries.append(
                TileSummary(
                    x=t.x,
                    y=t.y,
                    tile_type=t.type.value,
                    resources=[(r.kind, r.quantity) for r in t.resources if r.quantity > 0],
                    occupants=[oid for oid in t.occupants if oid != agent.agent_id],
                )
            )
        return summaries

    def _scan_agents(
        self, agent: Agent, engine: Any, tiles: list[TileSummary]
    ) -> list[AgentSummary]:
        """Identify visible agents and assess them."""
        visible = []
        seen_ids: set = set()

        for tile in tiles:
            for occ_id in tile.occupants:
                if occ_id in seen_ids:
                    continue
                seen_ids.add(occ_id)

                # Look up the agent from registry
                other = None
                if hasattr(engine, "registry"):
                    other = engine.registry.get(occ_id)

                if other is not None:
                    # Look up last action from engine's history buffer
                    last_action_data = engine._last_tick_actions.get(occ_id, ("", False, None))
                    action_type, success, target_id = last_action_data

                    visible.append(
                        AgentSummary(
                            agent_id=occ_id,
                            position=(other.x, other.y),
                            apparent_health=self._assess_health(other),
                            is_carrying_items=bool(other.inventory),
                            last_action=action_type if success else "",
                            last_action_target=target_id if success else None,
                        )
                    )

        return visible

    def _assess_health(self, agent: Agent) -> str:
        """Rough visual assessment of another agent's condition."""
        avg = (
            agent.needs.hunger + agent.needs.thirst + agent.needs.energy + agent.needs.health
        ) / 4
        if avg > 60:
            return "healthy"
        elif avg > 30:
            return "injured"
        return "critical"

    def _drain_messages(self, agent: Agent, engine: Any) -> list[Any]:
        """Get incoming messages for this agent from the message bus."""
        if hasattr(engine, "message_bus"):
            mailbox = engine.message_bus.get_or_create_mailbox(agent.agent_id)
            result: list[Any] = mailbox.drain_inbox()
            return result
        return []
