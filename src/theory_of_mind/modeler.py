"""Mind modeler: updates mental models based on observations."""

from __future__ import annotations

from src.theory_of_mind.agent_model import MindState


class MindModeler:
    """Updates agent models from observations.

    Maps observed actions to trait inferences using ACTION_TRAIT_MAP.
    """

    # Map actions to trait adjustments (trait_name: delta)
    ACTION_TRAIT_MAP: dict[str, dict[str, float]] = {
        "GIVE": {
            "resource_sharing": 0.02,
            "cooperation_tendency": 0.015,
        },
        "ATTACK": {
            "aggression": 0.03,
            "cooperation_tendency": -0.02,
        },
        "MOVE": {
            "curiosity": 0.005,
        },
        "GATHER": {
            "risk_tolerance": -0.005,  # cautious behavior
        },
        "REST": {
            "risk_tolerance": -0.01,
        },
        "WAIT": {
            "risk_tolerance": -0.005,
        },
    }

    @staticmethod
    def update_from_observation(
        mind_state: MindState,
        agent_id: str,
        action: str,
        position: tuple[int, int],
        tick: int,
        apparent_health: str = "healthy",
    ) -> None:
        """Update mental model based on observed action.

        Args:
            mind_state: The observing agent's mind state
            agent_id: ID of observed agent
            action: Action type (e.g., "MOVE", "ATTACK")
            position: Observed position
            tick: Current simulation tick
            apparent_health: Observed health status
        """
        model = mind_state.get_or_create(agent_id)

        # Update observation metadata
        model.ticks_observed += 1
        model.last_observed_tick = tick
        model.last_observed_position = position

        # Record action
        model.record_action(action)

        # Update traits based on action
        if action in MindModeler.ACTION_TRAIT_MAP:
            trait_deltas = MindModeler.ACTION_TRAIT_MAP[action]
            for trait, delta in trait_deltas.items():
                model.update_trait(trait, delta)

        # Infer needs from observed health
        if apparent_health == "critical":
            model.estimated_needs["health"] = 20.0
            model.estimated_needs["hunger"] = 30.0
        elif apparent_health == "injured":
            model.estimated_needs["health"] = 50.0

    @staticmethod
    def update_from_interaction(
        mind_state: MindState,
        agent_id: str,
        interaction_type: str,
        tick: int,
    ) -> None:
        """Update mental model based on direct interaction.

        Args:
            mind_state: The observing agent's mind state
            agent_id: ID of interacting agent
            interaction_type: Type of interaction ("helped_me", "attacked_me", etc.)
            tick: Current simulation tick
        """
        model = mind_state.get_or_create(agent_id)
        model.last_observed_tick = tick

        match interaction_type:
            case "helped_me":
                model.times_helped_me += 1
                model.estimated_disposition += 0.1
                model.update_trait("cooperation_tendency", 0.05)
                model.update_trait("resource_sharing", 0.04)

            case "attacked_me":
                model.times_attacked_me += 1
                model.estimated_disposition -= 0.2  # Hostility drops disposition faster
                model.update_trait("aggression", 0.08)
                model.update_trait("cooperation_tendency", -0.05)

            case "i_helped":
                model.times_i_helped += 1
                # Observe their reaction to help (assume neutral if no counter-action)
                model.estimated_disposition += 0.05

            case "i_attacked":
                model.times_i_attacked += 1
                model.estimated_disposition -= 0.15

            case "communicated":
                model.update_trait("sociability", 0.02)
                model.estimated_disposition += 0.02

            case "ignored_me":
                model.update_trait("sociability", -0.01)

        # Clamp disposition to [-1, 1]
        model.estimated_disposition = max(-1.0, min(1.0, model.estimated_disposition))

    @staticmethod
    def decay_confidence(
        mind_state: MindState, agent_id: str, ticks_since_observation: int
    ) -> None:
        """Reduce confidence in a model when we haven't observed the agent recently.

        Args:
            mind_state: The observing agent's mind state
            agent_id: ID of agent whose model should decay
            ticks_since_observation: Number of ticks since last observation
        """
        model = mind_state.get(agent_id)
        if model is None:
            return

        # Decay confidence by 1% per 10 ticks of no observation
        decay_rate = 0.01 * (ticks_since_observation / 10.0)
        for trait in model.trait_confidence:
            model.trait_confidence[trait] = max(0.0, model.trait_confidence[trait] - decay_rate)
