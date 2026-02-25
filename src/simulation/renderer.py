"""Rich terminal renderer for the multi-agent simulation."""

from __future__ import annotations

import io
import sys
from typing import TYPE_CHECKING, Any

from rich.console import Console

if TYPE_CHECKING:
    from src.simulation.engine import SimulationEngine

from src.simulation.world import TileType


def _make_console() -> Console:
    """Create a Rich Console that works on Windows (force UTF-8)."""
    try:
        # Wrap stdout in UTF-8 to avoid cp1252 encoding errors on Windows
        if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
            utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
            return Console(file=utf8_stdout, force_terminal=True)
    except Exception:
        pass
    return Console()


# Tile characters for the map
TILE_CHARS = {
    TileType.GRASS: (".", "green"),
    TileType.FOREST: ("T", "dark_green"),
    TileType.WATER: ("~", "blue"),
    TileType.ROCK: ("#", "grey50"),
    TileType.SHELTER: ("H", "yellow"),
}

# Agent archetype symbols and colors
AGENT_DISPLAY = {
    "gatherer": ("G", "bold green"),
    "explorer": ("E", "bold cyan"),
    "diplomat": ("D", "bold yellow"),
    "aggressor": ("A", "bold red"),
    "survivalist": ("S", "bold white"),
}


class Renderer:
    """Rich terminal renderer for the simulation — supports multi-agent."""

    def __init__(self, viewport_size: int = 21):
        self.console = _make_console()
        self.viewport_size = viewport_size
        self._follow_agent_idx: int = 0  # Which agent to follow

    def set_follow(self, idx: int) -> None:
        """Set which agent index to follow (viewport centers on them)."""
        self._follow_agent_idx = idx

    def render_frame(self, engine: SimulationEngine) -> str:
        """Render one frame of the simulation as a string."""
        parts = []
        parts.append(self._render_header(engine))
        parts.append("")
        parts.append(self._render_map(engine))
        parts.append("")
        parts.append(self._render_needs(engine))
        parts.append("")
        parts.append(self._render_inventory(engine))
        parts.append("")

        # Last action
        if engine.state.history:
            last = engine.state.history[-1]
            if last.result:
                parts.append(f"  Action: {last.result.message}")

        return "\n".join(parts)

    def render_multi_frame(self, engine: SimulationEngine, tick_record: Any = None) -> str:
        """Render a multi-agent frame with agent summary, thoughts, and messages."""
        parts = []

        # Header
        parts.append(self._render_multi_header(engine))
        parts.append("")

        # Map with all agents
        parts.append(self._render_multi_map(engine))
        parts.append("")

        # Agent summary (all agents compact)
        parts.append(self._render_agent_summary(engine))
        parts.append("")

        # Selected agent detail
        followed = self._get_followed_agent(engine)
        if followed:
            parts.append(self._render_agent_detail(engine, followed))
            parts.append("")

        # Thought panel (last awareness loop output)
        if followed:
            parts.append(self._render_thoughts(engine, followed))
            parts.append("")

        # Recent messages
        parts.append(self._render_messages(engine))
        parts.append("")

        # Emergent events
        if tick_record and tick_record.emergent_events:
            parts.append(self._render_emergent_events(tick_record))
            parts.append("")

        return "\n".join(parts)

    # --- Header ---

    def _render_header(self, engine: SimulationEngine) -> str:
        """Legacy single-agent header."""
        tick = engine.state.tick
        day = engine.state.day
        time = engine.state.time_of_day
        agent = engine.agent
        pos = f"({agent.x}, {agent.y})" if agent else "(?)"
        return f"  Day {day} | {time.upper():>5} | Tick {tick:>5} | Pos {pos}"

    def _render_multi_header(self, engine: SimulationEngine) -> str:
        """Multi-agent header."""
        tick = engine.state.tick
        day = engine.state.day
        time = engine.state.time_of_day
        alive = engine.registry.count_living
        dead = engine.registry.count_dead
        return (
            f"  cogniarch | Day {day} | {time.upper():>5} | "
            f"Tick {tick:>5} | Alive: {alive} | Dead: {dead}"
        )

    # --- Map ---

    def _render_map(self, engine: SimulationEngine) -> str:
        """Legacy single-agent map viewport."""
        agent = engine.agent
        if not agent:
            return "  (no agent)"
        return self._render_viewport(engine, agent.x, agent.y, {})

    def _render_multi_map(self, engine: SimulationEngine) -> str:
        """Multi-agent map viewport centered on followed agent."""
        followed = self._get_followed_agent(engine)
        if not followed:
            return "  (no agents alive)"

        # Build agent position map
        agent_map: dict[tuple[int, int], tuple[str, str]] = {}
        for agent in engine.agents:
            archetype = agent.profile.archetype if agent.profile else "survivalist"
            symbol, color = AGENT_DISPLAY.get(archetype, ("?", "white"))
            pos = (agent.x, agent.y)
            if pos not in agent_map:  # First agent at position wins
                agent_map[pos] = (symbol, color)

        return self._render_viewport(engine, followed.x, followed.y, agent_map)

    def _render_viewport(
        self,
        engine: SimulationEngine,
        cx: int,
        cy: int,
        agent_map: dict[tuple[int, int], tuple[str, str]],
    ) -> str:
        """Render a viewport centered on (cx, cy) with optional agent markers."""
        world = engine.world
        half = self.viewport_size // 2
        lines = []

        for dy in range(-half, half + 1):
            row = "  "
            for dx in range(-half, half + 1):
                wx, wy = cx + dx, cy + dy
                pos = (wx, wy)

                if pos == (cx, cy) and not agent_map:
                    row += "@"  # Legacy: single agent marker
                elif pos in agent_map:
                    symbol, _ = agent_map[pos]
                    row += symbol
                elif 0 <= wx < world.width and 0 <= wy < world.height:
                    tile = world.tiles[wy][wx]
                    char, _ = TILE_CHARS.get(tile.type, ("?", "white"))
                    row += char
                else:
                    row += " "
            lines.append(row)

        return "\n".join(lines)

    # --- Agent panels ---

    def _render_needs(self, engine: SimulationEngine) -> str:
        """Legacy single-agent needs."""
        agent = engine.agent
        if not agent:
            return ""
        return self._render_needs_bars(agent)

    def _render_needs_bars(self, agent: Any) -> str:
        """Render need bars for one agent."""
        needs = agent.needs
        bars = []
        for name, value in [
            ("Hunger", needs.hunger),
            ("Thirst", needs.thirst),
            ("Energy", needs.energy),
            ("Health", needs.health),
        ]:
            bar_len = 20
            filled = int(value / 100 * bar_len)
            empty = bar_len - filled
            bar = f"  {name:>6}: [{'█' * filled}{'░' * empty}] {value:5.1f}"
            bars.append(bar)
        return "\n".join(bars)

    def _render_inventory(self, engine: SimulationEngine) -> str:
        """Legacy single-agent inventory."""
        agent = engine.agent
        if not agent:
            return ""
        inv = agent.inventory
        if not inv:
            return "  Inventory: (empty)"
        items = ", ".join(f"{k}: {v}" for k, v in sorted(inv.items()))
        return f"  Inventory: {items}"

    def _render_agent_summary(self, engine: SimulationEngine) -> str:
        """Compact summary of all agents."""
        lines = ["  === Agents ==="]

        # When following an agent, show them first with a star marker
        agents_to_show = list(enumerate(engine.agents))
        if self._follow_agent_idx < len(engine.agents):
            # Move followed agent to the front
            followed = agents_to_show.pop(self._follow_agent_idx)
            agents_to_show.insert(0, followed)

        for i, agent in agents_to_show:
            archetype = agent.profile.archetype if agent.profile else "?"
            name = agent.profile.name if agent.profile else f"agent_{i}"
            symbol, _ = AGENT_DISPLAY.get(archetype, ("?", "white"))

            h = agent.needs.health
            status = "OK" if h > 60 else "LOW" if h > 30 else "CRIT"

            follow_marker = " ★" if i == self._follow_agent_idx else ""
            lines.append(
                f"  [{symbol}] {name:>16} | HP:{h:4.0f} | "
                f"({agent.x:>3},{agent.y:>3}) | {status}{follow_marker}"
            )

        return "\n".join(lines)

    def _render_agent_detail(self, engine: SimulationEngine, agent: Any) -> str:
        """Detailed view of one agent."""
        lines = []
        name = agent.profile.name if agent.profile else "agent"
        archetype = agent.profile.archetype if agent.profile else "?"
        lines.append(f"  === {name} ({archetype}) ===")
        lines.append(self._render_needs_bars(agent))

        inv = agent.inventory
        if inv:
            items = ", ".join(f"{k}: {v}" for k, v in sorted(inv.items()))
            lines.append(f"  Inventory: {items}")
        else:
            lines.append("  Inventory: (empty)")

        # Show traits if available
        if agent.profile and agent.profile.traits:
            traits = agent.profile.traits.as_dict()
            trait_str = " | ".join(f"{k[:4]}:{v:.1f}" for k, v in traits.items())
            lines.append(f"  Traits: {trait_str}")

        return "\n".join(lines)

    def _render_thoughts(self, engine: SimulationEngine, agent: Any) -> str:
        """Show the agent's internal monologue from awareness loop."""
        loop = engine.registry.get_awareness_loop(agent.agent_id)
        if not loop:
            return "  Thoughts: (no awareness loop)"

        intention = loop.last_intention
        if intention:
            return f"  Thinking: {intention.primary_goal} (confidence: {intention.confidence:.1f})"
        return "  Thoughts: (none yet)"

    def _render_messages(self, engine: SimulationEngine) -> str:
        """Show recent messages."""
        recent = engine.message_bus.recent_messages(5)
        if not recent:
            return "  Messages: (none)"

        lines = ["  === Messages ==="]
        for msg in recent:
            sender = str(msg.sender_id)[:8]
            lines.append(f"  [{msg.message_type.value}] {sender}: {msg.content}")
        return "\n".join(lines)

    def _render_emergent_events(self, tick_record: Any) -> str:
        """Show emergent events from this tick."""
        lines = ["  === Emergent Events ==="]
        for event_str in tick_record.emergent_events[:3]:
            lines.append(f"  ! {event_str}")
        return "\n".join(lines)

    # --- Lifecycle ---

    def _get_followed_agent(self, engine: SimulationEngine) -> Any | None:
        """Get the agent we're currently following."""
        agents = engine.agents
        if not agents:
            return None
        idx = self._follow_agent_idx % len(agents)
        return agents[idx]

    def print_frame(self, engine: SimulationEngine) -> None:
        """Clear and print single-agent frame."""
        self.console.clear()
        self.console.print(self.render_frame(engine))

    def print_multi_frame(self, engine: SimulationEngine, tick_record: Any = None) -> None:
        """Clear and print multi-agent frame."""
        self.console.clear()
        self.console.print(self.render_multi_frame(engine, tick_record))

    def print_death(self, engine: SimulationEngine) -> None:
        """Print death summary (legacy)."""
        self.console.print("\n  [bold red]AGENT DIED[/bold red]")
        self.console.print(f"  Survived {engine.state.tick} ticks ({engine.state.day} days)")
        if engine.agent:
            needs = engine.agent.needs
            self.console.print(
                f"  Final needs: hunger={needs.hunger:.1f} thirst={needs.thirst:.1f} "
                f"energy={needs.energy:.1f} health={needs.health:.1f}"
            )

    def print_complete(self, engine: SimulationEngine) -> None:
        """Print completion summary (legacy)."""
        self.console.print("\n  [bold green]SIMULATION COMPLETE[/bold green]")
        self.console.print(f"  Survived all {engine.state.tick} ticks ({engine.state.day} days)")

    def print_multi_summary(self, engine: SimulationEngine) -> None:
        """Print multi-agent end-of-simulation summary."""
        self.console.print("\n  [bold cyan]═══ cogniarch — SIMULATION COMPLETE ═══[/bold cyan]")
        self.console.print(f"  Duration: {engine.state.tick} ticks ({engine.state.day} days)")
        self.console.print(f"  Agents alive: {engine.registry.count_living}")
        self.console.print(f"  Agents dead: {engine.registry.count_dead}")

        # Per-agent summary - show followed agent first
        self.console.print("\n  [bold]Agent Results:[/bold]")
        all_agents = list(engine.registry.all_agents())
        if self._follow_agent_idx < len(all_agents):
            # Move followed agent to the front
            followed = all_agents.pop(self._follow_agent_idx)
            all_agents.insert(0, followed)

        for idx, agent in enumerate(all_agents):
            name = agent.profile.name if agent.profile else "?"
            archetype = agent.profile.archetype if agent.profile else "?"
            status = "[green]ALIVE[/green]" if agent.alive else "[red]DEAD[/red]"
            cause = ""
            if not agent.alive:
                cause = f" ({engine.registry.death_cause(agent.agent_id)})"
            follow_marker = " ★" if idx == 0 and self._follow_agent_idx < len(all_agents) else ""
            self.console.print(
                f"    {name:>20} ({archetype:>12}) — {status} "
                f"| Survived {agent.ticks_alive} ticks{cause}{follow_marker}"
            )

        # Show trait evolution
        self.console.print("\n  [bold]Trait Evolution:[/bold]")
        for agent in engine.registry.all_agents():
            if not agent.profile or not agent.profile.traits:
                continue
            name = agent.profile.name if agent.profile else "?"
            archetype = agent.profile.archetype if agent.profile else "?"
            from src.agents.archetypes import ARCHETYPES

            original = ARCHETYPES.get(archetype, {}).get("traits")
            if original:
                current = agent.profile.traits.as_dict()
                orig_dict = original.as_dict()
                changes = []
                for k in current:
                    diff = current[k] - orig_dict[k]
                    if abs(diff) > 0.02:
                        arrow = "↑" if diff > 0 else "↓"
                        changes.append(f"{k[:4]}{arrow}{abs(diff):.2f}")
                if changes:
                    self.console.print(f"    {name:>20}: {', '.join(changes)}")

        # Emergent events summary
        events = engine.emergence_detector.all_events
        if events:
            self.console.print(f"\n  [bold]Emergent Events: {len(events)} detected[/bold]")
            # Count by type
            from collections import Counter

            type_counts = Counter(e.pattern_type for e in events)
            for ptype, count in type_counts.most_common():
                self.console.print(f"    {ptype}: {count}")

        # Metrics summary
        latest = engine.metrics_collector.latest()
        if latest:
            self.console.print("\n  [bold]Final Metrics:[/bold]")
            self.console.print(f"    Total messages: {engine.message_bus.total_messages}")
            self.console.print(
                f"    Cooperation trend: {engine.metrics_collector.trend('cooperation_events')}"
            )
            self.console.print(
                f"    Aggression trend: {engine.metrics_collector.trend('aggression_events')}"
            )
