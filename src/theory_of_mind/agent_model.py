"""Agent mental models: tracking and predicting other agents' traits, needs, and behavior."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class AgentModel:
    """Mental model of another agent.

    Built from observed behavior over time. Used to predict actions and
    inform strategic decisions.
    """

    agent_id: str
    estimated_traits: dict[str, float] = field(default_factory=dict)
    trait_confidence: dict[str, float] = field(default_factory=dict)
    action_history: deque = field(default_factory=lambda: deque(maxlen=50))
    action_distribution: dict[str, int] = field(default_factory=dict)
    estimated_needs: dict[str, float] = field(default_factory=dict)
    estimated_disposition: float = 0.0  # -1=hostile, 0=neutral, 1=friendly
    predicted_next_action: str | None = None
    prediction_accuracy: float = 0.5  # running accuracy of our predictions
    ticks_observed: int = 0
    last_observed_tick: int = 0
    last_observed_position: tuple[int, int] | None = None
    times_helped_me: int = 0
    times_attacked_me: int = 0
    times_i_helped: int = 0
    times_i_attacked: int = 0

    def __post_init__(self):
        """Initialize default trait estimates."""
        if not self.estimated_traits:
            self.estimated_traits = {
                "cooperation_tendency": 0.5,
                "curiosity": 0.5,
                "risk_tolerance": 0.5,
                "resource_sharing": 0.5,
                "aggression": 0.5,
                "sociability": 0.5,
            }
        if not self.trait_confidence:
            self.trait_confidence = {trait: 0.0 for trait in self.estimated_traits}

    def update_trait(self, trait: str, delta: float, confidence_gain: float = 0.05):
        """Update trait estimate and confidence."""
        if trait in self.estimated_traits:
            old_val = self.estimated_traits[trait]
            new_val = max(0.0, min(1.0, old_val + delta))
            self.estimated_traits[trait] = new_val

            # Confidence increases with observations, capped at 1.0
            current_conf = self.trait_confidence.get(trait, 0.0)
            self.trait_confidence[trait] = min(1.0, current_conf + confidence_gain)

    def record_action(self, action: str):
        """Record observed action to history and distribution."""
        self.action_history.append(action)
        self.action_distribution[action] = self.action_distribution.get(action, 0) + 1

    def get_action_probability(self, action: str) -> float:
        """Get empirical probability of this agent taking given action."""
        total = sum(self.action_distribution.values())
        if total == 0:
            return 0.0
        return self.action_distribution.get(action, 0) / total

    def get_most_common_action(self) -> str | None:
        """Get agent's most frequent action."""
        if not self.action_distribution:
            return None
        return max(self.action_distribution.items(), key=lambda x: x[1])[0]


@dataclass
class MindState:
    """Collection of all mental models maintained by one agent.

    This is the agent's "theory of mind" state — their understanding
    of other agents in the world.
    """

    owner_id: str
    models: dict[str, AgentModel] = field(default_factory=dict)

    def get_or_create(self, agent_id: str) -> AgentModel:
        """Get existing model or create new one for an agent."""
        if agent_id not in self.models:
            self.models[agent_id] = AgentModel(agent_id=agent_id)
        return self.models[agent_id]

    def get(self, agent_id: str) -> AgentModel | None:
        """Get model if it exists."""
        return self.models.get(agent_id)

    def all_models(self) -> list[AgentModel]:
        """Get all agent models."""
        return list(self.models.values())

    def known_agents(self) -> list[str]:
        """Get all known agent IDs."""
        return list(self.models.keys())
