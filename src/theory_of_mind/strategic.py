"""Strategic reasoner: uses mental models to make social decisions."""

from __future__ import annotations

from src.theory_of_mind.agent_model import MindState
from src.theory_of_mind.predictor import IntentionPredictor


class StrategicReasoner:
    """Makes strategic decisions based on theory of mind."""

    @staticmethod
    def evaluate_approach(
        mind_state: MindState,
        agent_id: str,
        own_needs: dict[str, float],
    ) -> tuple[str, float]:
        """Evaluate whether to approach an agent.

        Returns:
            (recommendation, confidence)
            recommendation: "approach", "avoid", "neutral"
        """
        model = mind_state.get(agent_id)
        if model is None:
            return ("neutral", 0.3)

        disposition = IntentionPredictor.predict_disposition_toward_me(model)
        will_attack = IntentionPredictor.will_likely_attack(model)

        # Avoid hostile agents
        if disposition == "hostile" or will_attack:
            confidence = 0.7 if model.times_attacked_me > 0 else 0.5
            return ("avoid", confidence)

        # Approach friendly agents, especially if we need help
        if disposition == "friendly":
            own_health = own_needs.get("health", 100.0)
            own_hunger = own_needs.get("hunger", 100.0)

            # Higher confidence if we're in need and they've helped before
            if (own_health < 50 or own_hunger < 30) and model.times_helped_me > 0:
                return ("approach", 0.8)

            return ("approach", 0.6)

        # Neutral agents — approach if they seem cooperative
        cooperation = model.estimated_traits.get("cooperation_tendency", 0.5)
        if cooperation > 0.6:
            return ("approach", 0.5)

        return ("neutral", 0.4)

    @staticmethod
    def evaluate_sharing(
        mind_state: MindState,
        agent_id: str,
        own_inventory: dict[str, int],
        own_needs: dict[str, float],
    ) -> tuple[bool, str, float]:
        """Evaluate whether to share resources with an agent.

        Returns:
            (should_share, item_to_share, confidence)
        """
        model = mind_state.get(agent_id)
        if model is None or not own_inventory:
            return (False, "", 0.0)

        # Don't share with hostile agents
        disposition = IntentionPredictor.predict_disposition_toward_me(model)
        if disposition == "hostile":
            return (False, "", 0.9)

        # More likely to share with friendly agents who have helped us
        if disposition == "friendly" and model.times_helped_me > 0:
            # Share if we can afford it (needs are above 40)
            own_hunger = own_needs.get("hunger", 100.0)
            own_health = own_needs.get("health", 100.0)

            if own_hunger > 40 and own_health > 40:
                # Choose item to share
                item = next(iter(own_inventory))
                return (True, item, 0.7)

        # Neutral agents — share if they seem likely to reciprocate
        if disposition == "neutral":
            cooperation = model.estimated_traits.get("cooperation_tendency", 0.5)
            sharing = model.estimated_traits.get("resource_sharing", 0.5)

            if (
                cooperation > 0.6
                and sharing > 0.5
                and all(need > 60 for need in own_needs.values())
            ):
                # Only share if we have surplus (needs > 60)
                item = next(iter(own_inventory))
                return (True, item, 0.5)

        return (False, "", 0.3)

    @staticmethod
    def rank_agents_by_threat(mind_state: MindState) -> list[tuple[str, float]]:
        """Rank all known agents by threat level.

        Returns:
            List of (agent_id, threat_score) sorted by threat (highest first)
        """
        threat_scores = []

        for agent_id, model in mind_state.models.items():
            aggression = model.estimated_traits.get("aggression", 0.5)
            disposition = model.estimated_disposition

            # Threat = aggression - disposition (hostile aggressive agents are highest threat)
            threat = aggression - disposition

            # Weight by attack history
            if model.times_attacked_me > 0:
                threat += 0.3 * model.times_attacked_me

            # Reduce threat if they've helped us
            if model.times_helped_me > 0:
                threat -= 0.2 * model.times_helped_me

            threat_scores.append((agent_id, threat))

        # Sort by threat descending
        threat_scores.sort(key=lambda x: x[1], reverse=True)
        return threat_scores

    @staticmethod
    def suggest_social_action(
        mind_state: MindState,
        agent_id: str,
        own_inventory: dict[str, int],
        own_needs: dict[str, float],
        own_position: tuple[int, int],
    ) -> tuple[str, dict]:
        """Suggest best social action toward an agent.

        Returns:
            (action_type, action_params)
            action_type: "approach", "avoid", "share", "threaten", "attack", "none"
        """
        model = mind_state.get(agent_id)
        if model is None:
            return ("none", {})

        # Calculate distance if we know their position
        distance = None
        if model.last_observed_position:
            dx = abs(model.last_observed_position[0] - own_position[0])
            dy = abs(model.last_observed_position[1] - own_position[1])
            distance = dx + dy

        disposition = IntentionPredictor.predict_disposition_toward_me(model)
        will_attack = IntentionPredictor.will_likely_attack(model)

        # Immediate threats — attack or flee
        if (disposition == "hostile" or will_attack) and distance is not None and distance <= 2:
            # Attack if we're healthy enough, otherwise flee
            own_health = own_needs.get("health", 100.0)
            if own_health > 60:
                return ("attack", {"target_agent_id": agent_id})
            else:
                return ("avoid", {"target_agent_id": agent_id})

        # Distant threats — avoid
        if disposition == "hostile" and distance is not None and distance > 2:
            return ("avoid", {"target_agent_id": agent_id})

        # Friendly agents — evaluate sharing
        if disposition == "friendly":
            should_share, item, conf = StrategicReasoner.evaluate_sharing(
                mind_state, agent_id, own_inventory, own_needs
            )
            if should_share and distance is not None and distance <= 1:
                return ("share", {"target_agent_id": agent_id, "item": item, "confidence": conf})

            # Not adjacent — approach
            if distance is not None and distance > 1:
                return ("approach", {"target_agent_id": agent_id})

        # Neutral agents — approach if cooperative
        if disposition == "neutral":
            cooperation = model.estimated_traits.get("cooperation_tendency", 0.5)
            if cooperation > 0.6 and distance is not None and distance > 1:
                return ("approach", {"target_agent_id": agent_id})

        return ("none", {})
