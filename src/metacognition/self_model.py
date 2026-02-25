"""Cognitive self-model: agent's learned model of its own capabilities.

Agents build an understanding of which domains they excel in by tracking
performance over time using exponential moving average updates.
"""

from __future__ import annotations

from src.metacognition.types import CapabilityRating


class CognitiveSelfModel:
    """Agent's learned model of its own capabilities across domains.

    Uses exponential moving average (EMA) to slowly adapt ratings based on
    observed outcomes. Requires MIN_SAMPLES before forming strong opinions.

    Known domains: gathering, social, threat_assessment, exploration, combat, planning
    """

    EMA_ALPHA = 0.05  # Slow learning rate
    MIN_SAMPLES = 10  # Don't form strong opinions until enough data

    def __init__(self):
        """Initialize an empty self-model with no capability ratings."""
        self._ratings: dict[str, CapabilityRating] = {}

    def update_from_outcome(self, domain: str, succeeded: bool, prediction_error: float) -> None:
        """Update capability rating for a domain based on an outcome.

        Uses EMA to blend new evidence with existing rating. Confidence grows
        with sample count.

        Args:
            domain: The capability domain (e.g., "gathering", "social")
            succeeded: Whether the action in this domain succeeded
            prediction_error: Magnitude of prediction error (currently unused,
                             reserved for future bias estimation)
        """
        # Get or create rating for this domain
        if domain not in self._ratings:
            self._ratings[domain] = CapabilityRating(
                domain=domain,
                rating=0.5,  # Neutral prior
                confidence=0.0,
                sample_count=0,
                bias_estimate=0.0,
            )

        rating = self._ratings[domain]
        rating.sample_count += 1

        # Only update rating after minimum samples collected
        if rating.sample_count >= self.MIN_SAMPLES:
            new_value = 1.0 if succeeded else 0.0
            rating.rating = self.EMA_ALPHA * new_value + (1 - self.EMA_ALPHA) * rating.rating

        # Update confidence: grows with sample count
        rating.confidence = min(1.0, rating.sample_count / (self.MIN_SAMPLES * 5))

    def update_bias_estimate(self, domain: str, bias: float) -> None:
        """Update the bias estimate for a domain using EMA.

        Args:
            domain: The capability domain
            bias: New bias observation (positive = overestimate ability)
        """
        if domain not in self._ratings:
            self._ratings[domain] = CapabilityRating(
                domain=domain,
                rating=0.5,
                confidence=0.0,
                sample_count=0,
                bias_estimate=0.0,
            )

        rating = self._ratings[domain]
        rating.bias_estimate = self.EMA_ALPHA * bias + (1 - self.EMA_ALPHA) * rating.bias_estimate

    def get_rating(self, domain: str) -> CapabilityRating:
        """Get capability rating for a domain.

        Args:
            domain: The capability domain

        Returns:
            CapabilityRating for the domain, or neutral prior if not yet learned
        """
        if domain in self._ratings:
            return self._ratings[domain]

        # Return neutral prior for unknown domains
        return CapabilityRating(
            domain=domain,
            rating=0.5,
            confidence=0.0,
            sample_count=0,
            bias_estimate=0.0,
        )

    def strongest_domain(self) -> str:
        """Identify the domain with highest capability rating.

        Returns:
            Domain name with highest rating, or "unknown" if no ratings exist
        """
        if not self._ratings:
            return "unknown"

        return max(self._ratings.items(), key=lambda x: x[1].rating)[0]

    def weakest_domain(self) -> str:
        """Identify the domain with lowest capability rating.

        Returns:
            Domain name with lowest rating, or "unknown" if no ratings exist
        """
        if not self._ratings:
            return "unknown"

        return min(self._ratings.items(), key=lambda x: x[1].rating)[0]

    def capability_summary(self) -> dict[str, float]:
        """Get a summary of all capability ratings.

        Returns:
            Dict mapping domain name to rating value (0.0-1.0)
        """
        return {domain: rating.rating for domain, rating in self._ratings.items()}

    def to_dict(self) -> dict:
        """Serialize the full self-model to a dictionary.

        Returns:
            Dict with all domains and their full CapabilityRating data
        """
        return {
            domain: {
                "domain": rating.domain,
                "rating": rating.rating,
                "confidence": rating.confidence,
                "sample_count": rating.sample_count,
                "bias_estimate": rating.bias_estimate,
            }
            for domain, rating in self._ratings.items()
        }
