"""Intention predictor: predicts what other agents will do next."""

from __future__ import annotations

from src.theory_of_mind.agent_model import AgentModel


class IntentionPredictor:
    """Predicts actions and dispositions of other agents."""

    @staticmethod
    def predict_next_action(model: AgentModel) -> tuple[str | None, float]:
        """Predict agent's next action based on their history.

        Returns:
            (predicted_action, confidence)
        """
        if not model.action_distribution:
            return (None, 0.0)

        # Simple frequency-based prediction
        most_common = model.get_most_common_action()
        if most_common is None:
            return (None, 0.0)

        probability = model.get_action_probability(most_common)

        # Adjust confidence based on observation count and trait confidence
        observation_factor = min(1.0, model.ticks_observed / 20.0)
        avg_trait_confidence = sum(model.trait_confidence.values()) / len(model.trait_confidence)

        confidence = probability * observation_factor * (0.5 + 0.5 * avg_trait_confidence)

        model.predicted_next_action = most_common
        return (most_common, confidence)

    @staticmethod
    def predict_disposition_toward_me(model: AgentModel) -> str:
        """Predict how agent will behave toward observer.

        Returns:
            "hostile", "neutral", or "friendly"
        """
        disp = model.estimated_disposition

        # Weight by interaction history
        if model.times_attacked_me > model.times_helped_me:
            disp -= 0.2
        elif model.times_helped_me > model.times_attacked_me:
            disp += 0.2

        # Factor in aggression trait
        aggression = model.estimated_traits.get("aggression", 0.5)
        cooperation = model.estimated_traits.get("cooperation_tendency", 0.5)

        adjusted_disp = disp + (cooperation - aggression) * 0.3

        if adjusted_disp < -0.3:
            return "hostile"
        elif adjusted_disp > 0.3:
            return "friendly"
        else:
            return "neutral"

    @staticmethod
    def evaluate_prediction(model: AgentModel, actual_action: str) -> float:
        """Evaluate accuracy of last prediction.

        Updates model's prediction_accuracy running average.

        Returns:
            1.0 if correct, 0.0 if wrong
        """
        if model.predicted_next_action is None:
            return 0.5  # No prediction made

        correct = 1.0 if model.predicted_next_action == actual_action else 0.0

        # Update running average (exponential moving average)
        alpha = 0.1  # learning rate
        model.prediction_accuracy = alpha * correct + (1 - alpha) * model.prediction_accuracy

        return correct

    @staticmethod
    def will_likely_attack(model: AgentModel) -> bool:
        """Predict if agent is likely to attack soon.

        Based on aggression trait and recent attack actions.
        """
        aggression = model.estimated_traits.get("aggression", 0.5)

        # Check recent actions for attack pattern
        recent_attacks = sum(1 for action in list(model.action_history)[-10:] if action == "ATTACK")

        attack_rate = (
            recent_attacks / min(10, len(model.action_history)) if model.action_history else 0.0
        )

        return aggression > 0.6 and attack_rate > 0.2

    @staticmethod
    def will_likely_share(model: AgentModel) -> bool:
        """Predict if agent is likely to share resources.

        Based on sharing trait and past giving behavior.
        """
        sharing = model.estimated_traits.get("resource_sharing", 0.5)
        cooperation = model.estimated_traits.get("cooperation_tendency", 0.5)

        # Check recent giving actions
        recent_gives = sum(1 for action in list(model.action_history)[-10:] if action == "GIVE")

        give_rate = (
            recent_gives / min(10, len(model.action_history)) if model.action_history else 0.0
        )

        return (sharing > 0.5 or cooperation > 0.6) and give_rate > 0.1
