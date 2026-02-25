"""Cultural transmission and norm emergence.

Implements social learning and cultural evolution:
- CulturalTransmission: Agents learn from observing others
- CulturalNorm: Shared behavioral patterns
- NormDetector: Detects emergent norms from action patterns
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.identity import PersonalityTraits


@dataclass
class CulturalNorm:
    """A shared behavioral pattern in the population.

    Represents an emergent norm detected from repeated behaviors.
    """

    norm_type: str  # "cooperation", "territoriality", "resource_sharing", etc.
    behavior_pattern: str  # Description of the pattern
    strength: float  # 0.0-1.0, how prevalent is this norm
    adopters: set[str] = field(default_factory=set)  # Agent IDs following norm
    formation_tick: int = 0

    def add_adopter(self, agent_id: str) -> None:
        """Record an agent following this norm."""
        self.adopters.add(agent_id)

    def remove_adopter(self, agent_id: str) -> None:
        """Record an agent abandoning this norm."""
        self.adopters.discard(agent_id)

    def update_strength(self, population_size: int) -> None:
        """Update norm strength based on adoption rate."""
        if population_size > 0:
            self.strength = len(self.adopters) / population_size
        else:
            self.strength = 0.0


class CulturalTransmission:
    """Handles social learning and norm propagation.

    Agents can learn behaviors from successful neighbors,
    leading to cultural evolution alongside genetic evolution.
    """

    def __init__(self, learning_rate: float = 0.01):
        """Initialize cultural transmission system.

        Args:
            learning_rate: Rate at which traits shift through observation
        """
        self.learning_rate = learning_rate

    def observe_and_learn(
        self,
        observer_traits: PersonalityTraits,
        model_traits: PersonalityTraits,
        model_success: bool,
        observer_openness: float = 0.5,
    ) -> PersonalityTraits:
        """Agent learns from observing another agent's behavior.

        If the observed agent is successful, the observer's traits
        shift slightly toward the model's traits.

        Args:
            observer_traits: Traits of the learning agent
            model_traits: Traits of the observed agent
            model_success: Whether the observed agent was successful
            observer_openness: How receptive the observer is (0.0-1.0)

        Returns:
            Updated traits for the observer
        """
        if not model_success:
            # Don't learn from failures
            return observer_traits

        # Copy traits
        new_traits = observer_traits.copy()

        # Shift each trait slightly toward the model's value
        effective_rate = self.learning_rate * observer_openness

        for trait_name, observer_value in observer_traits.as_dict().items():
            model_value = getattr(model_traits, trait_name)

            # Shift toward model
            delta = (model_value - observer_value) * effective_rate

            # Clamp to [0.0, 1.0]
            new_traits.shift_trait(trait_name, delta)

        return new_traits

    def propagate_norm(
        self,
        norm: CulturalNorm,
        agent_id: str,
        agent_traits: PersonalityTraits,
        social_influence: float = 0.5,
    ) -> bool:
        """Attempt to propagate a norm to an agent.

        Args:
            norm: Cultural norm to propagate
            agent_id: Target agent ID
            agent_traits: Target agent's traits
            social_influence: Strength of social influence (0.0-1.0)

        Returns:
            True if agent adopted the norm
        """
        # Already following this norm
        if agent_id in norm.adopters:
            return True

        # Adoption probability based on norm strength and agent sociability
        adoption_chance = norm.strength * agent_traits.sociability * social_influence

        import random

        if random.random() < adoption_chance:
            norm.add_adopter(agent_id)
            return True

        return False


class NormDetector:
    """Detects emergent cultural norms from behavioral patterns.

    Tracks repeated behaviors across the population and identifies
    when they become widespread enough to be considered norms.
    """

    def __init__(self, detection_threshold: int = 5, strength_threshold: float = 0.3):
        """Initialize norm detector.

        Args:
            detection_threshold: Minimum occurrences to detect a pattern
            strength_threshold: Minimum adoption rate to be considered a norm
        """
        self.detection_threshold = detection_threshold
        self.strength_threshold = strength_threshold
        self._action_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._detected_norms: list[CulturalNorm] = []

    def record_action(
        self,
        agent_id: str,
        action_type: str,
        context: str,
        tick: int,
    ) -> None:
        """Record an agent's action in a given context.

        Args:
            agent_id: Agent performing action
            action_type: Type of action taken
            context: Context or situation (e.g., "near_food", "near_agent")
            tick: Current tick
        """
        # Track pattern: context -> action
        pattern_key = f"{context}:{action_type}"
        self._action_counts[agent_id][pattern_key] += 1

    def detect(
        self,
        population_size: int,
        tick: int,
    ) -> list[CulturalNorm]:
        """Detect emergent norms from recorded actions.

        Args:
            population_size: Current population size
            tick: Current tick

        Returns:
            List of detected cultural norms
        """
        # Count pattern prevalence across population
        pattern_prevalence: dict[str, set[str]] = defaultdict(set)

        for agent_id, patterns in self._action_counts.items():
            for pattern_key, count in patterns.items():
                if count >= self.detection_threshold:
                    pattern_prevalence[pattern_key].add(agent_id)

        # Identify norms (patterns followed by sufficient proportion)
        for pattern_key, adopters in pattern_prevalence.items():
            strength = len(adopters) / population_size if population_size > 0 else 0.0

            if strength < self.strength_threshold:
                continue

            # Check if we already detected this norm
            existing = False
            for norm in self._detected_norms:
                if norm.behavior_pattern == pattern_key:
                    # Update existing norm
                    norm.adopters = adopters
                    norm.update_strength(population_size)
                    existing = True
                    break

            if not existing:
                # Create new norm
                context, action = pattern_key.split(":", 1)
                norm_type = self._classify_norm_type(context, action)

                norm = CulturalNorm(
                    norm_type=norm_type,
                    behavior_pattern=pattern_key,
                    strength=strength,
                    adopters=adopters,
                    formation_tick=tick,
                )
                self._detected_norms.append(norm)

        return self._detected_norms

    def _classify_norm_type(self, context: str, action: str) -> str:
        """Classify what type of norm a pattern represents.

        Args:
            context: Situation context
            action: Action taken

        Returns:
            Norm type classification
        """
        if "share" in action.lower() or "give" in action.lower():
            return "cooperation"
        elif "attack" in action.lower() or "threat" in action.lower():
            return "aggression"
        elif "territory" in context.lower() or "defend" in action.lower():
            return "territoriality"
        elif "gather" in action.lower() or "food" in context.lower():
            return "resource_gathering"
        else:
            return "other"

    def get_norms(self) -> list[CulturalNorm]:
        """Get all detected norms.

        Returns:
            List of CulturalNorm objects
        """
        return self._detected_norms.copy()

    def clear_weak_norms(self, min_strength: float = 0.1) -> None:
        """Remove norms that have weakened below threshold.

        Args:
            min_strength: Minimum strength to keep norm
        """
        self._detected_norms = [
            norm for norm in self._detected_norms if norm.strength >= min_strength
        ]
