"""Behavioral repertoire system for cultural transmission.

This module implements the repertoire of learned behaviors that agents maintain:
- CulturalVariant: A context → action mapping with accumulated evidence
- BehavioralRepertoire: An agent's library of learned context-action mappings

Agents build and refine their repertoires through observation and experience,
adopting strategies that prove effective in specific contexts.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CulturalVariant:
    """A context → action mapping with accumulated evidence.

    This is the unit of cultural information:
    "In context X, do action Y" with evidence of how well it works.

    Tracks both observational evidence (learned from others) and
    personal experience (own use outcomes).
    """

    variant_id: str  # f"{context_tag}:{action_type}" — unique key
    context_tag: str  # e.g., "low_hunger"
    action_type: str  # e.g., "gather" (ActionType.value)

    # Evidence tracking (from observation)
    times_observed: int = 0
    times_succeeded: int = 0
    total_fitness_delta: float = 0.0

    # Adoption tracking
    adopted: bool = False
    adoption_tick: int = -1
    times_used: int = 0
    own_success_count: int = 0

    # Source tracking
    learned_from: list[str] = field(default_factory=list)

    @property
    def observed_success_rate(self) -> float:
        """Success rate based on observations of others.

        Returns:
            Success rate in [0.0, 1.0], or 0.5 if no observations
            (uninformative prior).
        """
        if self.times_observed == 0:
            return 0.5
        return self.times_succeeded / self.times_observed

    @property
    def avg_fitness_delta(self) -> float:
        """Average fitness change per observed use.

        Returns:
            Mean fitness delta, or 0.0 if no observations.
        """
        if self.times_observed == 0:
            return 0.0
        return self.total_fitness_delta / self.times_observed

    @property
    def own_success_rate(self) -> float:
        """Success rate based on own usage.

        Returns:
            Success rate in [0.0, 1.0], or 0.5 if never used
            (uninformative prior).
        """
        if self.times_used == 0:
            return 0.5
        return self.own_success_count / self.times_used


class BehavioralRepertoire:
    """An agent's library of learned context → action mappings.

    Maintains a collection of cultural variants that the agent has observed
    or learned. Agents adopt variants to use in decision-making and refine
    them based on personal experience.

    Maximum capacity enforced to prevent unbounded growth.
    """

    MAX_VARIANTS = 100

    def __init__(self):
        """Initialize an empty behavioral repertoire."""
        self._variants: dict[str, CulturalVariant] = {}

    def get_or_create(self, context_tag: str, action_type: str) -> CulturalVariant:
        """Get existing variant or create new one.

        Args:
            context_tag: Context identifier (e.g., "low_hunger")
            action_type: Action type identifier (e.g., "gather")

        Returns:
            Existing or newly created CulturalVariant
        """
        variant_id = f"{context_tag}:{action_type}"
        if variant_id not in self._variants:
            self._variants[variant_id] = CulturalVariant(
                variant_id=variant_id,
                context_tag=context_tag,
                action_type=action_type,
            )
        return self._variants[variant_id]

    def update_from_observation(
        self,
        context_tag: str,
        action_type: str,
        success: bool,
        fitness_delta: float,
        actor_id: str,
    ) -> CulturalVariant:
        """Update variant evidence from observing another agent.

        Creates the variant if it doesn't exist. Increments observation count,
        success count if applicable, accumulates fitness delta, and records
        the actor as a source.

        Args:
            context_tag: Context in which action was observed
            action_type: Action that was observed
            success: Whether the action succeeded
            fitness_delta: Fitness change resulting from the action
            actor_id: ID of the agent who performed the action

        Returns:
            Updated CulturalVariant
        """
        variant = self.get_or_create(context_tag, action_type)
        variant.times_observed += 1
        if success:
            variant.times_succeeded += 1
        variant.total_fitness_delta += fitness_delta

        # Record actor as source if not already present
        if actor_id not in variant.learned_from:
            variant.learned_from.append(actor_id)

        return variant

    def record_own_use(self, context_tag: str, action_type: str, success: bool) -> None:
        """Record the outcome of personally using a variant.

        Only updates existing variants — no-op if variant doesn't exist.

        Args:
            context_tag: Context in which action was used
            action_type: Action that was used
            success: Whether the action succeeded
        """
        variant_id = f"{context_tag}:{action_type}"
        if variant_id in self._variants:
            variant = self._variants[variant_id]
            variant.times_used += 1
            if success:
                variant.own_success_count += 1

    def adopt(self, context_tag: str, action_type: str, tick: int) -> None:
        """Mark a variant as adopted for use in decision-making.

        Args:
            context_tag: Context of the variant
            action_type: Action of the variant
            tick: Tick at which adoption occurred
        """
        variant = self.get_or_create(context_tag, action_type)
        variant.adopted = True
        variant.adoption_tick = tick

    def unadopt(self, context_tag: str, action_type: str) -> None:
        """Mark a variant as no longer adopted.

        Args:
            context_tag: Context of the variant
            action_type: Action of the variant
        """
        variant_id = f"{context_tag}:{action_type}"
        if variant_id in self._variants:
            self._variants[variant_id].adopted = False

    def lookup(self, context_tag: str) -> CulturalVariant | None:
        """Find the best adopted variant for a context.

        Searches all adopted variants matching the context and returns
        the one with the highest observed success rate. Ties broken by
        number of observations (more observations = more confident).

        Args:
            context_tag: Context to query

        Returns:
            Best matching CulturalVariant, or None if no adopted variants match
        """
        candidates = [
            v for v in self._variants.values() if v.adopted and v.context_tag == context_tag
        ]

        if not candidates:
            return None

        # Sort by success rate (descending), then by observation count (descending)
        candidates.sort(key=lambda v: (v.observed_success_rate, v.times_observed), reverse=True)
        return candidates[0]

    def adopted_variants(self) -> list[CulturalVariant]:
        """Get all adopted variants.

        Returns:
            List of all CulturalVariants with adopted=True
        """
        return [v for v in self._variants.values() if v.adopted]

    def all_variants(self) -> list[CulturalVariant]:
        """Get all variants in the repertoire.

        Returns:
            List of all CulturalVariants (adopted and non-adopted)
        """
        return list(self._variants.values())

    def variant_count(self) -> int:
        """Total number of variants in repertoire.

        Returns:
            Total variant count
        """
        return len(self._variants)

    def adopted_count(self) -> int:
        """Number of adopted variants.

        Returns:
            Count of variants with adopted=True
        """
        return sum(1 for v in self._variants.values() if v.adopted)

    def to_dict(self) -> dict:
        """Serialize repertoire for trajectory recording.

        Produces a compact representation with key metrics for each variant.

        Returns:
            Dictionary mapping variant_id to variant metadata
        """
        result = {}
        for variant in self._variants.values():
            result[variant.variant_id] = {
                "context": variant.context_tag,
                "action": variant.action_type,
                "observed_success_rate": round(variant.observed_success_rate, 3),
                "times_observed": variant.times_observed,
                "adopted": variant.adopted,
                "times_used": variant.times_used,
                "learned_from": variant.learned_from[:5],  # Cap at 5 entries
            }
        return result
