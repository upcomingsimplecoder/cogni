"""Cultural evolution metrics and analysis.

Tracks cultural dynamics across the simulation:
- Variant frequency over time (analogous to allele frequency in genetics)
- Cultural diversity via Shannon entropy
- Cultural group detection via Jaccard similarity clustering
- Transmission event statistics

Designed for both real-time dashboards and post-hoc research analysis.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


@dataclass
class CulturalSnapshot:
    """Cultural state at a single tick for trajectory recording."""

    tick: int
    variant_frequencies: dict[str, int]  # variant_id -> count of adopters
    transmission_events_count: int
    dominant_variants: list[str]  # top 5 most adopted
    cultural_diversity: float  # Shannon entropy of variant distribution
    bias_distribution: dict[str, int]  # bias_type -> adoption count


class CulturalEvolutionAnalyzer:
    """Analyzes cultural evolution across the simulation.

    Tracks:
    - Variant frequency over time (analogous to allele frequency)
    - Cultural diversity (Shannon entropy of variant distribution)
    - Cultural group detection (agents sharing similar repertoires)
    - Transmission fidelity and complexity metrics
    """

    def __init__(self) -> None:
        """Initialize the analyzer."""
        self._history: list[CulturalSnapshot] = []

    def record_tick(
        self,
        cultural_engine: Any,
        tick: int,
    ) -> CulturalSnapshot:
        """Record cultural state for this tick.

        Args:
            cultural_engine: CulturalTransmissionEngine instance
            tick: Current tick number

        Returns:
            CulturalSnapshot with current cultural state
        """
        # Count variant adoption across all agents
        variant_adopter_count: dict[str, int] = defaultdict(int)

        for _agent_id, rep in cultural_engine._repertoires.items():
            for variant in rep.adopted_variants():
                variant_adopter_count[variant.variant_id] += 1

        # Sort by adoption count (descending)
        sorted_variants = sorted(
            variant_adopter_count.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Cultural diversity: Shannon entropy
        total_adoptions = sum(variant_adopter_count.values())
        diversity = 0.0
        if total_adoptions > 0:
            for count in variant_adopter_count.values():
                p = count / total_adoptions
                if p > 0:
                    diversity -= p * math.log2(p)

        # Recent transmission events for this tick
        recent_events = [e for e in cultural_engine.transmission_events if e.tick == tick]

        bias_dist: dict[str, int] = {}
        for e in recent_events:
            if e.adopted:
                bias_dist[e.bias_type] = bias_dist.get(e.bias_type, 0) + 1

        snapshot = CulturalSnapshot(
            tick=tick,
            variant_frequencies=dict(variant_adopter_count),
            transmission_events_count=len(recent_events),
            dominant_variants=[vid for vid, _ in sorted_variants[:5]],
            cultural_diversity=round(diversity, 3),
            bias_distribution=bias_dist,
        )

        self._history.append(snapshot)
        return snapshot

    def detect_cultural_groups(
        self,
        cultural_engine: Any,
    ) -> list[set[str]]:
        """Detect groups of agents with similar behavioral repertoires.

        Uses Jaccard similarity on adopted variant sets. Agents with
        similarity > 0.5 are clustered into the same cultural group.

        Args:
            cultural_engine: CulturalTransmissionEngine instance

        Returns:
            List of sets, each set containing agent_ids in the same group
        """
        # Build adopted variant sets per agent
        agent_variants: dict[str, set[str]] = {}
        for agent_id, rep in cultural_engine._repertoires.items():
            adopted = {v.variant_id for v in rep.adopted_variants()}
            if adopted:  # only include agents with at least one adoption
                agent_variants[agent_id] = adopted

        if not agent_variants:
            return []

        # Greedy clustering via Jaccard similarity
        agents = list(agent_variants.keys())
        assigned: set[str] = set()
        groups: list[set[str]] = []

        for agent in agents:
            if agent in assigned:
                continue

            group = {agent}
            assigned.add(agent)

            for other in agents:
                if other in assigned:
                    continue

                # Jaccard similarity: |A ∩ B| / |A ∪ B|
                a_set = agent_variants[agent]
                b_set = agent_variants[other]
                intersection = len(a_set & b_set)
                union = len(a_set | b_set)

                if union > 0 and intersection / union > 0.5:
                    group.add(other)
                    assigned.add(other)

            groups.append(group)

        return groups

    def variant_frequency_timeline(self, variant_id: str) -> list[tuple[int, int]]:
        """Get (tick, adopter_count) timeline for a specific variant.

        Args:
            variant_id: The variant to track

        Returns:
            List of (tick, count) tuples
        """
        return [(snap.tick, snap.variant_frequencies.get(variant_id, 0)) for snap in self._history]

    def get_history(self) -> list[CulturalSnapshot]:
        """Get full cultural snapshot history.

        Returns:
            Copy of all recorded snapshots
        """
        return self._history.copy()
