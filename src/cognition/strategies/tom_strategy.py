"""Theory of Mind strategy: wraps an inner strategy with mental-model-based social reasoning.

Follows the same delegation pattern as CulturallyModulatedStrategy:
- Always updates mental models from observations (passive, no cost)
- Overrides the inner strategy ONLY for social decisions when ToM has useful models
- Delegates all survival, gathering, communication, and exploration to the inner strategy
"""

from __future__ import annotations

from typing import Any

from src.awareness.types import Expression, Intention, Reflection, Sensation
from src.simulation.actions import Action, ActionType
from src.theory_of_mind.agent_model import MindState
from src.theory_of_mind.modeler import MindModeler
from src.theory_of_mind.strategic import StrategicReasoner


class TheoryOfMindStrategy:
    """Wraps any DecisionStrategy to add mental-model-based social reasoning.

    Before the inner strategy forms an intention, this wrapper:
    1. Updates mental models from current observations (always)
    2. Checks for high-priority social situations (threats, strategic opportunities)
    3. If a ToM-informed social decision is warranted, overrides the inner strategy
    4. Otherwise, delegates entirely to the inner strategy

    This ensures the inner strategy's survival logic (gathering, eating, sharing,
    communication, exploration) remains intact while ToM adds a social reasoning layer.
    """

    def __init__(self, agent_id: str, inner_strategy: Any | None = None):
        """Initialize with agent ID and optional inner strategy.

        Args:
            agent_id: This agent's ID (for MindState ownership)
            inner_strategy: The base strategy to wrap (e.g., PersonalityStrategy).
                If None, a default PersonalityStrategy is created.
        """
        self.mind_state = MindState(owner_id=agent_id)
        if inner_strategy is None:
            from src.cognition.strategies.personality import PersonalityStrategy

            inner_strategy = PersonalityStrategy()
        self.inner_strategy = inner_strategy
        self.last_tick = 0

    def form_intention(self, sensation: Sensation, reflection: Reflection) -> Intention:
        """Form intention: ToM social override or inner strategy delegation.

        Always updates mental models first (passive observation).
        Only overrides when ToM identifies a high-priority social situation.
        """
        # Always update mental models from observations
        self._update_models(sensation, reflection)

        # ToM override: strategic social decisions when agents are visible
        # and mental models suggest a meaningful response
        if sensation.visible_agents:
            social_intention = self._form_social_intention(sensation, reflection)
            if social_intention:
                return social_intention

        # Delegate everything else to inner strategy
        return self.inner_strategy.form_intention(sensation, reflection)

    def express(
        self, sensation: Sensation, reflection: Reflection, intention: Intention
    ) -> Expression:
        """Convert intention to action.

        ToM-specific intentions (approach/avoid/share/attack based on models)
        are handled here. Everything else delegates to the inner strategy.
        """
        match intention.primary_goal:
            case "approach_agent":
                if intention.target_position:
                    return self.inner_strategy._express_move_toward(sensation, intention)
                return Expression(action=Action(type=ActionType.WAIT))

            case "avoid_agent":
                if intention.target_position:
                    from src.simulation.actions import Direction

                    dx = sensation.own_position[0] - intention.target_position[0]
                    dy = sensation.own_position[1] - intention.target_position[1]

                    if abs(dx) >= abs(dy):
                        direction = Direction.EAST if dx > 0 else Direction.WEST
                    else:
                        direction = Direction.SOUTH if dy > 0 else Direction.NORTH

                    return Expression(action=Action(type=ActionType.MOVE, direction=direction))
                return Expression(action=Action(type=ActionType.WAIT))

            case "share_with_agent":
                if intention.target_agent_id and sensation.own_inventory:
                    item = next(iter(sensation.own_inventory))
                    return Expression(
                        action=Action(
                            type=ActionType.GIVE,
                            target=item,
                            target_agent_id=intention.target_agent_id,
                            quantity=1,
                        ),
                        internal_monologue=(
                            f"Strategically sharing {item} with {intention.target_agent_id}"
                        ),
                    )
                return Expression(action=Action(type=ActionType.WAIT))

            case "attack_agent":
                if intention.target_agent_id:
                    return Expression(
                        action=Action(
                            type=ActionType.ATTACK,
                            target_agent_id=intention.target_agent_id,
                        ),
                        internal_monologue=(
                            f"Preemptive attack on threat {intention.target_agent_id}"
                        ),
                    )
                return Expression(action=Action(type=ActionType.WAIT))

            case _:
                # Delegate to inner strategy for all other intentions
                return self.inner_strategy.express(sensation, reflection, intention)

    def _update_models(self, sensation: Sensation, reflection: Reflection):
        """Update mental models from current observations."""
        tick = sensation.tick

        # Update models from visible agents
        for agent_sum in sensation.visible_agents:
            agent_id = str(agent_sum.agent_id)

            model = self.mind_state.get_or_create(agent_id)
            model.last_observed_tick = tick
            model.last_observed_position = agent_sum.position

            # Infer needs from apparent health
            if agent_sum.apparent_health == "critical":
                model.estimated_needs["health"] = 20.0
            elif agent_sum.apparent_health == "injured":
                model.estimated_needs["health"] = 50.0
            else:
                model.estimated_needs["health"] = 80.0

            # Update from observed action
            if agent_sum.last_action:
                MindModeler.update_from_observation(
                    self.mind_state,
                    agent_id,
                    agent_sum.last_action,
                    agent_sum.position,
                    tick,
                    agent_sum.apparent_health,
                )

        # Update from interaction outcomes
        for outcome in reflection.recent_interaction_outcomes:
            agent_id = str(outcome.other_agent_id)
            interaction_type = outcome.interaction_type

            if outcome.was_positive:
                if "help" in interaction_type.lower():
                    MindModeler.update_from_interaction(
                        self.mind_state, agent_id, "helped_me", tick
                    )
            else:
                if "attack" in interaction_type.lower():
                    MindModeler.update_from_interaction(
                        self.mind_state, agent_id, "attacked_me", tick
                    )

        # Decay confidence for agents we haven't seen recently
        for agent_id, model in self.mind_state.models.items():
            ticks_since = tick - model.last_observed_tick
            if ticks_since > 10:
                MindModeler.decay_confidence(self.mind_state, agent_id, ticks_since)

        self.last_tick = tick

    def _form_social_intention(
        self, sensation: Sensation, reflection: Reflection
    ) -> Intention | None:
        """Form intention based on social reasoning with mental models.

        Only fires when ToM models suggest a high-priority social action:
        - Avoid/attack genuine threats (based on observed aggression)
        - Strategically share with cooperative agents
        - Approach potential allies

        Returns None to let the inner strategy handle non-social situations.
        """
        # Rank agents by threat
        threats = StrategicReasoner.rank_agents_by_threat(self.mind_state)

        # Handle high threats — only override if threat is credible
        if threats and threats[0][1] > 0.5:
            threat_id = threats[0][0]
            threat_model = self.mind_state.get(threat_id)

            if threat_model and threat_model.last_observed_position:
                dx = abs(threat_model.last_observed_position[0] - sensation.own_position[0])
                dy = abs(threat_model.last_observed_position[1] - sensation.own_position[1])
                distance = dx + dy

                # Only react if threat is actually visible (nearby)
                # Check the threat agent is in our current visible_agents
                threat_visible = any(str(a.agent_id) == threat_id for a in sensation.visible_agents)
                if not threat_visible:  # Don't override for stale threat models
                    pass
                elif distance <= 1 and sensation.own_needs.get("health", 100) > 60:
                    return Intention(
                        primary_goal="attack_agent",
                        target_agent_id=threat_model.agent_id,
                        target_position=threat_model.last_observed_position,
                        planned_actions=["attack"],
                        confidence=0.7,
                    )
                elif distance <= 5:
                    return Intention(
                        primary_goal="avoid_agent",
                        target_agent_id=threat_model.agent_id,
                        target_position=threat_model.last_observed_position,
                        planned_actions=["move_away"],
                        confidence=0.6,
                    )

        # Evaluate visible agents for strategic social opportunities
        for agent_sum in sensation.visible_agents:
            agent_id = str(agent_sum.agent_id)

            action_type, params = StrategicReasoner.suggest_social_action(
                self.mind_state,
                agent_id,
                sensation.own_inventory,
                sensation.own_needs,
                sensation.own_position,
            )

            if action_type == "share":
                return Intention(
                    primary_goal="share_with_agent",
                    target_agent_id=agent_id,
                    target_position=agent_sum.position,
                    planned_actions=["give_item"],
                    confidence=params.get("confidence", 0.6),
                )

            if action_type == "approach":
                return Intention(
                    primary_goal="approach_agent",
                    target_agent_id=agent_id,
                    target_position=agent_sum.position,
                    planned_actions=["move_toward_agent"],
                    confidence=0.5,
                )

            if action_type == "avoid":
                return Intention(
                    primary_goal="avoid_agent",
                    target_agent_id=agent_id,
                    target_position=agent_sum.position,
                    planned_actions=["move_away"],
                    confidence=0.6,
                )

            if action_type == "attack":
                return Intention(
                    primary_goal="attack_agent",
                    target_agent_id=agent_id,
                    target_position=agent_sum.position,
                    planned_actions=["attack"],
                    confidence=0.7,
                )

        return None
