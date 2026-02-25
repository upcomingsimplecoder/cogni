"""Transmission biases for cultural learning (Boyd & Richerson).

Implements four transmission biases that shape how agents learn from observations:
- Prestige bias: Learn from successful, high-status individuals
- Conformity bias: Adopt common behaviors (frequency-dependent)
- Content bias: Learn behaviors with intrinsically good outcomes
- Anti-conformity bias: Adopt rare/distinctive behaviors

These biases are weighted by agent personality to produce culturally-transmitted
action preferences.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.evolution.observation import ObservationMemory


# Conformist transmission exponent (Boyd & Richerson)
CONFORMIST_EXPONENT = 1.5


class LearningStyle(Enum):
    """Dominant transmission bias for an agent."""

    PRESTIGE = "prestige"
    CONFORMIST = "conformist"
    CONTENT = "content"
    ANTI_CONFORMIST = "anti_conformist"
    BALANCED = "balanced"


@dataclass
class TransmissionWeights:
    """Weighting of the four transmission biases for an agent.

    Each weight is 0.0-1.0, and all four should sum to 1.0.
    The dominant style determines which bias has strongest influence.
    """

    prestige: float = 0.25
    conformity: float = 0.25
    content: float = 0.25
    anti_conformity: float = 0.25

    @property
    def dominant_style(self) -> LearningStyle:
        """Determine which transmission bias dominates.

        Returns:
            BALANCED if top two weights are within 0.1, otherwise the max weight's style.
        """
        weights_map = {
            "prestige": self.prestige,
            "conformist": self.conformity,
            "content": self.content,
            "anti_conformist": self.anti_conformity,
        }

        sorted_weights = sorted(weights_map.items(), key=lambda x: x[1], reverse=True)
        top_weight = sorted_weights[0][1]
        second_weight = sorted_weights[1][1]

        # If top two are close, it's balanced
        if abs(top_weight - second_weight) <= 0.1:
            return LearningStyle.BALANCED

        # Otherwise return the dominant style
        dominant_name = sorted_weights[0][0]
        return LearningStyle(dominant_name)

    @staticmethod
    def from_personality(traits: dict[str, float]) -> TransmissionWeights:
        """Derive transmission weights from personality traits.

        Maps personality traits to learning biases:
        - prestige: sociable, cooperative agents learn from high-status others
        - conformity: less curious, more sociable agents follow the crowd
        - content: curious, risk-tolerant agents evaluate intrinsic quality
        - anti_conformity: curious, bold, less sociable agents seek novelty

        Args:
            traits: Dictionary of personality trait values (0.0-1.0)

        Returns:
            TransmissionWeights normalized to sum to 1.0
        """
        # Extract traits with defaults
        sociability = traits.get("sociability", 0.5)
        cooperation_tendency = traits.get("cooperation_tendency", 0.5)
        curiosity = traits.get("curiosity", 0.5)
        risk_tolerance = traits.get("risk_tolerance", 0.5)

        # Compute raw weights
        prestige = sociability * 0.6 + cooperation_tendency * 0.4
        conformity = (1 - curiosity) * 0.5 + sociability * 0.5
        content = curiosity * 0.5 + risk_tolerance * 0.3 + 0.2
        anti_conformity = curiosity * 0.4 + risk_tolerance * 0.4 + (1 - sociability) * 0.2

        # Normalize to sum to 1.0
        total = prestige + conformity + content + anti_conformity
        if total > 0:
            prestige /= total
            conformity /= total
            content /= total
            anti_conformity /= total

        # Round to 3 decimal places
        prestige = round(prestige, 3)
        conformity = round(conformity, 3)
        content = round(content, 3)
        anti_conformity = round(anti_conformity, 3)

        return TransmissionWeights(
            prestige=prestige,
            conformity=conformity,
            content=content,
            anti_conformity=anti_conformity,
        )


def prestige_score(observation_memory: ObservationMemory, actor_id: str) -> float:
    """Compute prestige score for an actor based on observed success.

    Prestige is a recency-weighted success rate, combining outcome success
    with fitness gains. Recent observations matter more.

    Args:
        observation_memory: The observer's observation memory
        actor_id: The actor whose prestige to compute

    Returns:
        Prestige score in [0, 1], or 0.0 if no observations
    """
    observations = observation_memory.observations_of(actor_id)
    if not observations:
        return 0.0

    # Find max tick for recency weighting
    max_tick = max(obs.tick for obs in observations)

    weighted_sum = 0.0
    weight_total = 0.0

    for obs in observations:
        # Recency weight: decay with age
        age = max_tick - obs.tick
        recency_weight = 0.95 ** (age / 10)

        # Base score: success or failure
        score = 1.0 if obs.outcome_success else 0.0

        # Bonus for positive fitness delta (capped at 0.5)
        if obs.outcome_fitness_delta > 0:
            fitness_bonus = min(0.5, obs.outcome_fitness_delta / 20)
            score += fitness_bonus

        weighted_sum += score * recency_weight
        weight_total += recency_weight

    if weight_total == 0:
        return 0.0

    prestige = weighted_sum / weight_total
    return max(0.0, min(1.0, prestige))


def prestige_bias(observation_memory: ObservationMemory, context_tag: str) -> dict[str, float]:
    """Compute prestige-based weights for each observed variant.

    Variants are weighted by the average prestige of actors who performed them.
    High-prestige individuals' actions get higher weight.

    Args:
        observation_memory: The observer's observation memory
        context_tag: Context to filter observations

    Returns:
        Dictionary mapping variant_id to prestige weight
    """
    observations = observation_memory.observations_in_context(context_tag)
    if not observations:
        return {}

    # Group by variant_id
    variant_actors: dict[str, set[str]] = {}
    for obs in observations:
        variant_id = f"{obs.context_tag}:{obs.action_type}"
        if variant_id not in variant_actors:
            variant_actors[variant_id] = set()
        variant_actors[variant_id].add(obs.actor_id)

    # Compute average prestige per variant
    variant_prestige: dict[str, float] = {}
    for variant_id, actor_ids in variant_actors.items():
        prestige_scores = [prestige_score(observation_memory, actor_id) for actor_id in actor_ids]
        avg_prestige = sum(prestige_scores) / len(prestige_scores)
        variant_prestige[variant_id] = avg_prestige

    return variant_prestige


def conformity_bias(
    observation_memory: ObservationMemory, context_tag: str, window: int = 50
) -> dict[str, float]:
    """Compute conformist frequency-dependent weights (Boyd & Richerson).

    Common behaviors get disproportionately higher weight via super-linear
    frequency transformation. This creates cultural convergence.

    Args:
        observation_memory: The observer's observation memory
        context_tag: Context to filter observations
        window: Number of recent observations to consider

    Returns:
        Dictionary mapping variant_id to conformity weight (normalized)
    """
    observations = observation_memory.observations_in_context(context_tag, n=window)
    if not observations:
        return {}

    # Count frequency of each variant
    variant_counts: dict[str, int] = {}
    for obs in observations:
        variant_id = f"{obs.context_tag}:{obs.action_type}"
        variant_counts[variant_id] = variant_counts.get(variant_id, 0) + 1

    total_count = len(observations)

    # Apply super-linear exponent for conformist bias
    variant_weights: dict[str, float] = {}
    for variant_id, count in variant_counts.items():
        frequency = count / total_count
        weight = frequency**CONFORMIST_EXPONENT
        variant_weights[variant_id] = weight

    # Normalize to sum to 1
    total_weight = sum(variant_weights.values())
    if total_weight > 0:
        variant_weights = {vid: w / total_weight for vid, w in variant_weights.items()}

    return variant_weights


def content_bias(observation_memory: ObservationMemory, context_tag: str) -> dict[str, float]:
    """Compute content-based weights by intrinsic success rate.

    Variants are evaluated by their actual outcomes, independent of who
    performed them or how common they are.

    Args:
        observation_memory: The observer's observation memory
        context_tag: Context to filter observations

    Returns:
        Dictionary mapping variant_id to content quality weight
    """
    observations = observation_memory.observations_in_context(context_tag)
    if not observations:
        return {}

    # Track success and fitness per variant
    variant_stats: dict[str, dict[str, float]] = {}
    for obs in observations:
        variant_id = f"{obs.context_tag}:{obs.action_type}"
        if variant_id not in variant_stats:
            variant_stats[variant_id] = {
                "total": 0,
                "successes": 0,
                "fitness_sum": 0.0,
            }

        stats = variant_stats[variant_id]
        stats["total"] += 1
        if obs.outcome_success:
            stats["successes"] += 1
        stats["fitness_sum"] += obs.outcome_fitness_delta

    # Compute weight per variant
    variant_weights: dict[str, float] = {}
    for variant_id, stats in variant_stats.items():
        success_rate = stats["successes"] / stats["total"]
        avg_fitness_delta = stats["fitness_sum"] / stats["total"]

        # Weight is success rate plus fitness bonus (capped)
        fitness_bonus = max(0, avg_fitness_delta / 20)
        weight = min(1.0, success_rate + fitness_bonus)
        variant_weights[variant_id] = weight

    return variant_weights


def anti_conformity_bias(
    observation_memory: ObservationMemory, context_tag: str, window: int = 50
) -> dict[str, float]:
    """Compute anti-conformist weights (inverse of conformity).

    Rare behaviors get higher weight. This promotes cultural diversity
    and exploration of novel strategies.

    Args:
        observation_memory: The observer's observation memory
        context_tag: Context to filter observations
        window: Number of recent observations to consider

    Returns:
        Dictionary mapping variant_id to anti-conformity weight (normalized)
    """
    conformity_weights = conformity_bias(observation_memory, context_tag, window)
    if not conformity_weights:
        return {}

    max_conf_weight = max(conformity_weights.values())

    # Invert: rare variants get high weight
    variant_weights: dict[str, float] = {}
    for variant_id, conf_weight in conformity_weights.items():
        weight = max(0.01, max_conf_weight - conf_weight + 0.01)
        variant_weights[variant_id] = weight

    # Normalize to sum to 1
    total_weight = sum(variant_weights.values())
    if total_weight > 0:
        variant_weights = {vid: w / total_weight for vid, w in variant_weights.items()}

    return variant_weights


def compute_combined_bias(
    observation_memory: ObservationMemory,
    context_tag: str,
    weights: TransmissionWeights,
) -> dict[str, float]:
    """Combine all four transmission biases using agent's weights.

    This produces the final culturally-influenced action preferences
    for the given context.

    Args:
        observation_memory: The observer's observation memory
        context_tag: Context to filter observations
        weights: Agent's transmission bias weights

    Returns:
        Dictionary mapping variant_id to combined score (normalized to max=1.0)
    """
    # Compute all four biases
    prestige_weights = prestige_bias(observation_memory, context_tag)
    conformity_weights = conformity_bias(observation_memory, context_tag)
    content_weights = content_bias(observation_memory, context_tag)
    anti_conformity_weights = anti_conformity_bias(observation_memory, context_tag)

    # Collect all variant_ids
    all_variant_ids: set[str] = set()
    all_variant_ids.update(prestige_weights.keys())
    all_variant_ids.update(conformity_weights.keys())
    all_variant_ids.update(content_weights.keys())
    all_variant_ids.update(anti_conformity_weights.keys())

    if not all_variant_ids:
        return {}

    # Combine using weighted sum
    combined: dict[str, float] = {}
    for variant_id in all_variant_ids:
        score = (
            weights.prestige * prestige_weights.get(variant_id, 0)
            + weights.conformity * conformity_weights.get(variant_id, 0)
            + weights.content * content_weights.get(variant_id, 0)
            + weights.anti_conformity * anti_conformity_weights.get(variant_id, 0)
        )
        combined[variant_id] = score

    # Normalize so max = 1.0
    max_score = max(combined.values())
    if max_score > 0:
        combined = {vid: score / max_score for vid, score in combined.items()}

    return combined
