"""Emergent behavior pattern detection from agent activity."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from src.emergence.events import EmergentEvent


@dataclass
class _ClusterTracker:
    """Internal state for tracking spatial clusters."""

    agents: frozenset  # frozenset of AgentIDs
    center: tuple[float, float]
    first_seen: int
    last_seen: int


@dataclass
class _TerritoryTracker:
    """Internal state for tracking territorial behavior."""

    agent_id: object
    center: tuple[float, float]
    positions: list[tuple[int, int]] = field(default_factory=list)
    first_seen: int = 0
    threats_sent: int = 0


class EmergenceDetector:
    """Detects emergent patterns from agent behavior.

    Patterns detected:
    1. Clustering — agents spatially grouping
    2. Sharing networks — repeated GIVE actions between same agents
    3. Territory formation — agent staying in bounded area + threats
    4. Specialization — agent actions concentrated in one category
    """

    def __init__(
        self,
        cluster_distance: int = 3,
        cluster_sustain: int = 5,
        territory_radius: int = 5,
        territory_sustain: int = 20,
    ):
        self.cluster_distance = cluster_distance
        self.cluster_sustain = cluster_sustain
        self.territory_radius = territory_radius
        self.territory_sustain = territory_sustain

        # Internal tracking state
        self._active_clusters: list[_ClusterTracker] = []
        self._sharing_log: list[tuple[int, object, object]] = []  # (tick, giver, receiver)
        self._agent_positions: dict[object, list[tuple[int, int]]] = defaultdict(list)
        self._agent_actions: dict[object, list[str]] = defaultdict(list)
        self._threats_sent: dict[object, int] = defaultdict(int)
        self._detected_events: list[EmergentEvent] = []

    def detect(
        self,
        tick: int,
        agents: list[Any],  # Agent objects with .agent_id, .x, .y
        tick_actions: list[tuple[object, str, bool, object | None]] | None = None,
        messages: list[Any] | None = None,
    ) -> list[EmergentEvent]:
        """Run all detectors. Returns newly detected events this tick.

        Args:
            tick: Current simulation tick
            agents: List of living Agent objects
            tick_actions: List of (agent_id, action_type, success, target_agent_id)
            messages: Messages delivered this tick
        """
        # Update tracking state
        self._update_tracking(tick, agents, tick_actions, messages)

        events: list[EmergentEvent] = []
        events.extend(self._detect_clusters(tick, agents))
        events.extend(self._detect_sharing_networks(tick))
        events.extend(self._detect_territory(tick))
        events.extend(self._detect_specialization(tick))

        self._detected_events.extend(events)
        return events

    def _update_tracking(
        self,
        tick: int,
        agents: list[Any],
        tick_actions: list[tuple[object, str, bool, object | None]] | None,
        messages: list[Any] | None,
    ) -> None:
        """Update internal tracking state from this tick's data."""
        # Track positions
        for agent in agents:
            self._agent_positions[agent.agent_id].append((agent.x, agent.y))
            # Keep last 50 positions
            if len(self._agent_positions[agent.agent_id]) > 50:
                self._agent_positions[agent.agent_id] = self._agent_positions[agent.agent_id][-50:]

        # Track actions
        if tick_actions:
            for agent_id, action_type, success, target_id in tick_actions:
                self._agent_actions[agent_id].append(action_type)
                if len(self._agent_actions[agent_id]) > 50:
                    self._agent_actions[agent_id] = self._agent_actions[agent_id][-50:]

                # Track sharing
                if action_type == "give" and success and target_id:
                    self._sharing_log.append((tick, agent_id, target_id))
                    # Keep last 200 sharing events
                    if len(self._sharing_log) > 200:
                        self._sharing_log = self._sharing_log[-200:]

        # Track threats from messages
        if messages:
            for msg in messages:
                if hasattr(msg, "message_type") and msg.message_type.value == "threat":
                    self._threats_sent[msg.sender_id] = self._threats_sent.get(msg.sender_id, 0) + 1

    def _detect_clusters(self, tick: int, agents: list[Any]) -> list[EmergentEvent]:
        """Detect spatial clustering: 2+ agents within distance threshold sustained."""
        if len(agents) < 2:
            return []

        events = []

        # Find current clusters using simple distance-based grouping
        clusters: list[set] = []
        agent_map = {a.agent_id: a for a in agents}
        used = set()

        for i, a1 in enumerate(agents):
            if a1.agent_id in used:
                continue
            cluster = {a1.agent_id}
            for j, a2 in enumerate(agents):
                if i == j or a2.agent_id in used:
                    continue
                dist = math.sqrt((a1.x - a2.x) ** 2 + (a1.y - a2.y) ** 2)
                if dist <= self.cluster_distance:
                    cluster.add(a2.agent_id)

            if len(cluster) >= 2:
                clusters.append(cluster)
                used.update(cluster)

        # Check against sustained clusters
        new_active = []
        for cluster in clusters:
            frozen = frozenset(cluster)
            # Calculate center
            xs = [agent_map[aid].x for aid in cluster if aid in agent_map]
            ys = [agent_map[aid].y for aid in cluster if aid in agent_map]
            center = (sum(xs) / len(xs), sum(ys) / len(ys)) if xs else (0, 0)

            # Check if this matches an existing tracked cluster
            matched = False
            for tracker in self._active_clusters:
                if len(tracker.agents & frozen) >= 2:  # Overlapping cluster
                    tracker.agents = frozen
                    tracker.center = center
                    tracker.last_seen = tick
                    new_active.append(tracker)
                    matched = True

                    # Check if sustained long enough
                    duration = tick - tracker.first_seen
                    if duration >= self.cluster_sustain and duration % self.cluster_sustain == 0:
                        events.append(
                            EmergentEvent(
                                tick=tick,
                                pattern_type="cluster",
                                agents_involved=list(frozen),
                                description=(
                                    f"{len(frozen)} agents clustered near "
                                    f"({center[0]:.0f}, {center[1]:.0f}) for {duration} ticks"
                                ),
                                data={"center": center, "duration": duration, "size": len(frozen)},
                            )
                        )
                    break

            if not matched:
                new_active.append(
                    _ClusterTracker(
                        agents=frozen,
                        center=center,
                        first_seen=tick,
                        last_seen=tick,
                    )
                )

        self._active_clusters = new_active
        return events

    def _detect_sharing_networks(self, tick: int) -> list[EmergentEvent]:
        """Detect repeated resource sharing between agent pairs."""
        events = []

        # Count sharing between pairs in last 50 ticks
        recent = [(t, g, r) for t, g, r in self._sharing_log if tick - t <= 50]
        pair_counts: dict[frozenset, int] = defaultdict(int)
        for _, giver, receiver in recent:
            pair = frozenset([giver, receiver])
            pair_counts[pair] += 1

        for pair, count in pair_counts.items():
            if count >= 3:
                agents = list(pair)
                events.append(
                    EmergentEvent(
                        tick=tick,
                        pattern_type="sharing_network",
                        agents_involved=agents,
                        description=f"Sharing network: {count} exchanges in last 50 ticks",
                        data={"exchange_count": count},
                    )
                )

        return events

    def _detect_territory(self, tick: int) -> list[EmergentEvent]:
        """Detect territorial behavior: staying in bounded area + threats."""
        events = []

        for agent_id, positions in self._agent_positions.items():
            if len(positions) < self.territory_sustain:
                continue

            recent = positions[-self.territory_sustain :]
            xs = [p[0] for p in recent]
            ys = [p[1] for p in recent]

            # Check if all positions within territory_radius of center
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            max_dist = max(math.sqrt((x - cx) ** 2 + (y - cy) ** 2) for x, y in recent)

            if max_dist <= self.territory_radius:
                threats = self._threats_sent.get(agent_id, 0)
                if threats > 0:
                    events.append(
                        EmergentEvent(
                            tick=tick,
                            pattern_type="territory",
                            agents_involved=[agent_id],
                            description=(
                                f"Territory at ({cx:.0f}, {cy:.0f}), radius {max_dist:.1f}, "
                                f"{threats} threats sent"
                            ),
                            data={"center": (cx, cy), "radius": max_dist, "threats": threats},
                        )
                    )

        return events

    def _detect_specialization(self, tick: int) -> list[EmergentEvent]:
        """Detect emergent role specialization from action patterns."""
        events = []

        # Action categories
        CATEGORIES = {
            "gathering": {"gather", "eat", "drink"},
            "social": {"give", "send_message"},
            "aggressive": {"attack", "fight"},
            "exploration": {"move", "scout"},
        }

        for agent_id, actions in self._agent_actions.items():
            if len(actions) < 30:
                continue

            recent = actions[-30:]
            total = len(recent)

            for category, action_types in CATEGORIES.items():
                count = sum(1 for a in recent if a in action_types)
                ratio = count / total
                if ratio > 0.6:
                    events.append(
                        EmergentEvent(
                            tick=tick,
                            pattern_type="specialization",
                            agents_involved=[agent_id],
                            description=f"Specialized as {category} ({ratio:.0%} of actions)",
                            data={"category": category, "ratio": ratio},
                        )
                    )

        return events

    @property
    def all_events(self) -> list[EmergentEvent]:
        """All detected emergent events."""
        return list(self._detected_events)

    @property
    def event_count(self) -> int:
        return len(self._detected_events)
