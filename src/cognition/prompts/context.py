"""Context builders for LLM prompts — assemble Sensation/Reflection into text."""

from __future__ import annotations

from src.awareness.types import Reflection, Sensation
from src.memory.episodic import EpisodicMemory
from src.memory.social import SocialMemory


class ContextBuilder:
    """Builds structured context strings from agent perception and memory.

    Formats data for LLM consumption with emphasis on:
    - Conciseness (token efficiency)
    - Relevance (prioritize actionable info)
    - Structure (consistent format for parsing)
    """

    def __init__(
        self,
        episodic_memory: EpisodicMemory | None = None,
        social_memory: SocialMemory | None = None,
    ):
        """Initialize with optional memory systems.

        Args:
            episodic_memory: Agent's episodic memory store
            social_memory: Agent's social relationship memory
        """
        self.episodic = episodic_memory
        self.social = social_memory

    def build_full_context(
        self,
        sensation: Sensation,
        reflection: Reflection,
        include_memory: bool = True,
    ) -> str:
        """Assemble complete context string from all available data.

        Args:
            sensation: Current perception
            reflection: Recent evaluation
            include_memory: Whether to include episodic/social memory sections

        Returns:
            Multi-section context string for LLM prompt
        """
        sections = [
            self._build_status_section(sensation),
            self._build_environment_section(sensation),
            self._build_agents_section(sensation),
            self._build_reflection_section(reflection),
        ]

        if include_memory:
            if self.episodic:
                sections.append(self._build_episodic_section())
            if self.social:
                sections.append(self._build_social_section(sensation))

        if include_memory and sensation.incoming_messages:
            sections.append(self._build_messages_section(sensation))

        return "\n\n".join(s for s in sections if s)

    def _build_status_section(self, sensation: Sensation) -> str:
        """Build agent status overview."""
        needs = sensation.own_needs
        lines = [
            f"## Your Status (Tick {sensation.tick})",
            f"Position: {sensation.own_position}",
            f"Time: {sensation.time_of_day}",
            f"Needs: Hunger={needs.get('hunger', 0):.0f}, "
            f"Thirst={needs.get('thirst', 0):.0f}, "
            f"Energy={needs.get('energy', 0):.0f}, "
            f"Health={needs.get('health', 100):.0f}",
        ]

        if sensation.own_inventory:
            inv_str = ", ".join(f"{k}×{v}" for k, v in sensation.own_inventory.items())
            lines.append(f"Inventory: {inv_str}")
        else:
            lines.append("Inventory: empty")

        return "\n".join(lines)

    def _build_environment_section(self, sensation: Sensation) -> str:
        """Build visible tiles and resources summary."""
        if not sensation.visible_tiles:
            return "## Visible Environment\nNo visibility."

        # Group by tile type
        tile_counts: dict[str, int] = {}
        resource_summary: dict[str, int] = {}

        for tile in sensation.visible_tiles:
            tile_counts[tile.tile_type] = tile_counts.get(tile.tile_type, 0) + 1
            for res_kind, qty in tile.resources:
                if qty > 0:
                    resource_summary[res_kind] = resource_summary.get(res_kind, 0) + qty

        lines = ["## Visible Environment"]
        lines.append(f"Tiles visible: {len(sensation.visible_tiles)}")

        if tile_counts:
            terrain = ", ".join(f"{k}×{v}" for k, v in sorted(tile_counts.items()))
            lines.append(f"Terrain: {terrain}")

        if resource_summary:
            resources = ", ".join(f"{k}×{v}" for k, v in sorted(resource_summary.items()))
            lines.append(f"Resources: {resources}")
        else:
            lines.append("Resources: none nearby")

        # Find nearest resource of each type
        own_x, own_y = sensation.own_position
        nearest: dict[str, tuple[tuple[int, int], int]] = {}  # kind -> (pos, dist)

        for tile in sensation.visible_tiles:
            dist = abs(tile.x - own_x) + abs(tile.y - own_y)
            for res_kind, qty in tile.resources:
                if qty > 0 and (res_kind not in nearest or dist < nearest[res_kind][1]):
                    nearest[res_kind] = ((tile.x, tile.y), dist)

        if nearest:
            near_lines = [
                f"{kind} at {pos} (dist {dist})"
                for kind, (pos, dist) in sorted(nearest.items(), key=lambda x: x[1][1])
            ]
            lines.append(f"Nearest resources: {'; '.join(near_lines)}")

        return "\n".join(lines)

    def _build_agents_section(self, sensation: Sensation) -> str:
        """Build visible agents summary."""
        if not sensation.visible_agents:
            return "## Nearby Agents\nNone visible."

        lines = ["## Nearby Agents"]
        own_x, own_y = sensation.own_position

        for agent in sensation.visible_agents:
            dist = abs(agent.position[0] - own_x) + abs(agent.position[1] - own_y)
            agent_id_str = str(agent.agent_id)[:8]  # Truncate for readability
            carrying = "carrying items" if agent.is_carrying_items else "empty-handed"
            lines.append(
                f"- Agent {agent_id_str} at {agent.position} "
                f"(dist {dist}): {agent.apparent_health}, {carrying}"
            )

        return "\n".join(lines)

    def _build_reflection_section(self, reflection: Reflection) -> str:
        """Build evaluation of recent experience."""
        lines = ["## Recent Evaluation"]
        lines.append(
            f"Last action: {'succeeded' if reflection.last_action_succeeded else 'failed'}"
        )

        if reflection.need_trends:
            trends = ", ".join(f"{k}={v}" for k, v in reflection.need_trends.items())
            lines.append(f"Need trends: {trends}")

        lines.append(f"Threat level: {reflection.threat_level:.1f}")
        lines.append(f"Opportunity score: {reflection.opportunity_score:.1f}")

        if reflection.recent_interaction_outcomes:
            interactions = [
                f"{'positive' if i.was_positive else 'negative'} {i.interaction_type}"
                for i in reflection.recent_interaction_outcomes[-3:]
            ]
            lines.append(f"Recent interactions: {', '.join(interactions)}")

        return "\n".join(lines)

    def _build_episodic_section(self) -> str:
        """Build episodic memory summary (recent successes/failures)."""
        if not self.episodic or self.episodic.episode_count == 0:
            return ""

        lines = ["## Recent Memory"]
        recent = self.episodic.recent_episodes(n=5)

        # Summarize action success rates
        action_types = {e.action_type for e in recent}
        rates = {act: self.episodic.success_rate(act, window=10) for act in action_types}
        if rates:
            rate_strs = [f"{act}={r:.0%}" for act, r in sorted(rates.items())]
            lines.append(f"Action success rates: {', '.join(rate_strs)}")

        # Notable recent episodes
        notable = [e for e in recent if not e.success or e.involved_agent is not None]
        if notable:
            episode_lines = []
            for e in notable[-3:]:
                result = "succeeded" if e.success else "failed"
                episode_lines.append(f"  - {e.action_type} {result} at {e.location}")
            lines.append("Recent events:\n" + "\n".join(episode_lines))

        return "\n".join(lines)

    def _build_social_section(self, sensation: Sensation) -> str:
        """Build social memory summary (relationships with visible agents)."""
        if not self.social or self.social.relationship_count == 0:
            return ""

        lines = ["## Known Relationships"]

        # Allies and enemies
        allies = self.social.allies(threshold=0.7)
        enemies = self.social.enemies(threshold=0.3)

        if allies:
            ally_ids = [str(r.other_agent_id)[:8] for r in allies]
            lines.append(f"Allies: {', '.join(ally_ids)}")

        if enemies:
            enemy_ids = [str(r.other_agent_id)[:8] for r in enemies]
            lines.append(f"Enemies: {', '.join(enemy_ids)}")

        # Relationships with visible agents
        visible_ids = {a.agent_id for a in sensation.visible_agents}
        visible_rels = [(aid, self.social.get(aid)) for aid in visible_ids]
        visible_rels = [(aid, rel) for aid, rel in visible_rels if rel is not None]

        if visible_rels:
            lines.append("Visible agents you know:")
            for aid, rel in visible_rels:
                aid_str = str(aid)[:8]
                trust_val = rel.trust if rel is not None else 0.5
                status = "ally" if trust_val >= 0.7 else "enemy" if trust_val <= 0.3 else "neutral"
                lines.append(f"  - {aid_str}: {status} (trust {trust_val:.1f})")

        return "\n".join(lines) if len(lines) > 1 else ""

    def _build_messages_section(self, sensation: Sensation) -> str:
        """Build incoming messages summary."""
        if not sensation.incoming_messages:
            return ""

        lines = ["## Messages Received"]
        for msg in sensation.incoming_messages[-5:]:  # Last 5 messages
            sender_str = str(msg.sender_id)[:8] if msg.sender_id else "unknown"
            msg_type = "INFO"
            if hasattr(msg, "message_type") and msg.message_type is not None:
                msg_type = (
                    msg.message_type.name
                    if hasattr(msg.message_type, "name")
                    else str(msg.message_type)
                )
            content = msg.content[:50] if hasattr(msg, "content") else str(msg)[:50]
            lines.append(f"- From {sender_str} ({msg_type}): {content}")

        return "\n".join(lines)
