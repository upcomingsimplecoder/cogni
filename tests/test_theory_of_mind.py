"""Tests for Theory of Mind module."""

from __future__ import annotations

import pytest

from src.awareness.types import AgentSummary, Reflection, Sensation
from src.cognition.strategies.tom_strategy import TheoryOfMindStrategy
from src.theory_of_mind.agent_model import AgentModel, MindState
from src.theory_of_mind.modeler import MindModeler
from src.theory_of_mind.predictor import IntentionPredictor
from src.theory_of_mind.strategic import StrategicReasoner


class TestAgentModel:
    """Test AgentModel and MindState."""

    def test_agent_model_initialization(self):
        """Test that AgentModel initializes with default traits."""
        model = AgentModel(agent_id="agent_1")

        assert model.agent_id == "agent_1"
        assert "cooperation_tendency" in model.estimated_traits
        assert model.estimated_traits["cooperation_tendency"] == 0.5
        assert model.ticks_observed == 0
        assert model.estimated_disposition == 0.0

    def test_update_trait(self):
        """Test trait updating with clamping."""
        model = AgentModel(agent_id="agent_1")

        # Normal update
        model.update_trait("aggression", 0.1)
        assert model.estimated_traits["aggression"] == 0.6
        assert model.trait_confidence["aggression"] == 0.05

        # Clamp at 1.0
        model.update_trait("aggression", 0.8)
        assert model.estimated_traits["aggression"] == 1.0

        # Clamp at 0.0
        model.update_trait("aggression", -2.0)
        assert model.estimated_traits["aggression"] == 0.0

    def test_record_action(self):
        """Test action recording and distribution."""
        model = AgentModel(agent_id="agent_1")

        model.record_action("MOVE")
        model.record_action("MOVE")
        model.record_action("ATTACK")

        assert len(model.action_history) == 3
        assert model.action_distribution["MOVE"] == 2
        assert model.action_distribution["ATTACK"] == 1
        assert model.get_action_probability("MOVE") == 2 / 3
        assert model.get_most_common_action() == "MOVE"

    def test_mind_state_get_or_create(self):
        """Test MindState agent model management."""
        mind_state = MindState(owner_id="me")

        # Create new model
        model1 = mind_state.get_or_create("agent_1")
        assert model1.agent_id == "agent_1"

        # Get existing model
        model2 = mind_state.get_or_create("agent_1")
        assert model2 is model1

        # Check known agents
        assert "agent_1" in mind_state.known_agents()
        assert len(mind_state.all_models()) == 1


class TestMindModeler:
    """Test MindModeler."""

    def test_update_from_observation(self):
        """Test updating model from observed action."""
        mind_state = MindState(owner_id="me")

        MindModeler.update_from_observation(
            mind_state,
            agent_id="agent_1",
            action="GIVE",
            position=(5, 5),
            tick=10,
            apparent_health="healthy",
        )

        model = mind_state.get("agent_1")
        assert model is not None
        assert model.ticks_observed == 1
        assert model.last_observed_tick == 10
        assert model.last_observed_position == (5, 5)
        assert "GIVE" in model.action_distribution

        # GIVE action increases resource_sharing and cooperation_tendency
        assert model.estimated_traits["resource_sharing"] > 0.5
        assert model.estimated_traits["cooperation_tendency"] > 0.5

    def test_update_from_attack_observation(self):
        """Test that ATTACK increases aggression trait."""
        mind_state = MindState(owner_id="me")

        MindModeler.update_from_observation(
            mind_state,
            agent_id="agent_2",
            action="ATTACK",
            position=(3, 3),
            tick=5,
        )

        model = mind_state.get("agent_2")
        assert model.estimated_traits["aggression"] > 0.5
        assert model.estimated_traits["cooperation_tendency"] < 0.5

    def test_update_from_interaction(self):
        """Test updating model from direct interaction."""
        mind_state = MindState(owner_id="me")

        # Helped me
        MindModeler.update_from_interaction(
            mind_state,
            agent_id="agent_1",
            interaction_type="helped_me",
            tick=10,
        )

        model = mind_state.get("agent_1")
        assert model.times_helped_me == 1
        assert model.estimated_disposition > 0.0
        assert model.estimated_traits["cooperation_tendency"] > 0.5

        # Attacked me
        MindModeler.update_from_interaction(
            mind_state,
            agent_id="agent_2",
            interaction_type="attacked_me",
            tick=11,
        )

        model2 = mind_state.get("agent_2")
        assert model2.times_attacked_me == 1
        assert model2.estimated_disposition < 0.0
        assert model2.estimated_traits["aggression"] > 0.5

    def test_decay_confidence(self):
        """Test confidence decay over time."""
        mind_state = MindState(owner_id="me")

        MindModeler.update_from_observation(mind_state, "agent_1", "MOVE", (5, 5), tick=10)

        model = mind_state.get("agent_1")
        initial_confidence = model.trait_confidence["curiosity"]
        assert initial_confidence > 0.0

        # Decay confidence after 100 ticks
        MindModeler.decay_confidence(mind_state, "agent_1", ticks_since_observation=100)

        new_confidence = model.trait_confidence["curiosity"]
        assert new_confidence < initial_confidence


class TestIntentionPredictor:
    """Test IntentionPredictor."""

    def test_predict_next_action(self):
        """Test action prediction based on history."""
        model = AgentModel(agent_id="agent_1")
        model.ticks_observed = 20

        # Record action pattern
        for _ in range(8):
            model.record_action("MOVE")
        for _ in range(2):
            model.record_action("GATHER")

        # Build up some trait confidence through observations
        for trait in model.trait_confidence:
            model.trait_confidence[trait] = 0.5

        predicted, confidence = IntentionPredictor.predict_next_action(model)

        assert predicted == "MOVE"
        assert confidence > 0.5  # Should be fairly confident

    def test_predict_disposition_hostile(self):
        """Test hostile disposition prediction."""
        model = AgentModel(agent_id="agent_1")
        model.estimated_disposition = -0.5
        model.times_attacked_me = 2
        model.estimated_traits["aggression"] = 0.8

        disposition = IntentionPredictor.predict_disposition_toward_me(model)
        assert disposition == "hostile"

    def test_predict_disposition_friendly(self):
        """Test friendly disposition prediction."""
        model = AgentModel(agent_id="agent_1")
        model.estimated_disposition = 0.5
        model.times_helped_me = 3
        model.estimated_traits["cooperation_tendency"] = 0.8

        disposition = IntentionPredictor.predict_disposition_toward_me(model)
        assert disposition == "friendly"

    def test_evaluate_prediction(self):
        """Test prediction evaluation and accuracy tracking."""
        model = AgentModel(agent_id="agent_1")
        model.predicted_next_action = "MOVE"
        model.prediction_accuracy = 0.5

        # Correct prediction
        accuracy = IntentionPredictor.evaluate_prediction(model, "MOVE")
        assert accuracy == 1.0
        assert model.prediction_accuracy > 0.5

        # Incorrect prediction
        model.predicted_next_action = "ATTACK"
        accuracy = IntentionPredictor.evaluate_prediction(model, "MOVE")
        assert accuracy == 0.0

    def test_will_likely_attack(self):
        """Test attack prediction."""
        model = AgentModel(agent_id="agent_1")
        model.estimated_traits["aggression"] = 0.8

        # Record some attack actions
        for _ in range(3):
            model.record_action("ATTACK")
        for _ in range(7):
            model.record_action("MOVE")

        assert IntentionPredictor.will_likely_attack(model)

    def test_will_likely_share(self):
        """Test sharing prediction."""
        model = AgentModel(agent_id="agent_1")
        model.estimated_traits["resource_sharing"] = 0.7
        model.estimated_traits["cooperation_tendency"] = 0.6

        # Record some giving actions
        for _ in range(2):
            model.record_action("GIVE")
        for _ in range(8):
            model.record_action("MOVE")

        assert IntentionPredictor.will_likely_share(model)


class TestStrategicReasoner:
    """Test StrategicReasoner."""

    def test_evaluate_approach_avoid_hostile(self):
        """Test that we avoid hostile agents."""
        mind_state = MindState(owner_id="me")
        model = mind_state.get_or_create("agent_1")
        model.estimated_disposition = -0.6
        model.times_attacked_me = 2

        recommendation, confidence = StrategicReasoner.evaluate_approach(
            mind_state, "agent_1", {"health": 100.0, "hunger": 100.0}
        )

        assert recommendation == "avoid"
        assert confidence > 0.5

    def test_evaluate_approach_friendly(self):
        """Test that we approach friendly agents."""
        mind_state = MindState(owner_id="me")
        model = mind_state.get_or_create("agent_1")
        model.estimated_disposition = 0.6
        model.times_helped_me = 1

        recommendation, confidence = StrategicReasoner.evaluate_approach(
            mind_state, "agent_1", {"health": 100.0, "hunger": 100.0}
        )

        assert recommendation == "approach"

    def test_evaluate_sharing_with_friendly(self):
        """Test sharing evaluation with friendly agent."""
        mind_state = MindState(owner_id="me")
        model = mind_state.get_or_create("agent_1")
        model.estimated_disposition = 0.7
        model.times_helped_me = 2

        should_share, item, conf = StrategicReasoner.evaluate_sharing(
            mind_state,
            "agent_1",
            own_inventory={"berries": 3},
            own_needs={"hunger": 60.0, "health": 80.0},
        )

        assert should_share
        assert item == "berries"
        assert conf > 0.5

    def test_evaluate_sharing_dont_share_with_hostile(self):
        """Test that we don't share with hostile agents."""
        mind_state = MindState(owner_id="me")
        model = mind_state.get_or_create("agent_1")
        model.estimated_disposition = -0.6
        model.times_attacked_me = 1

        should_share, _, _ = StrategicReasoner.evaluate_sharing(
            mind_state,
            "agent_1",
            own_inventory={"berries": 3},
            own_needs={"hunger": 80.0, "health": 80.0},
        )

        assert not should_share

    def test_rank_agents_by_threat(self):
        """Test threat ranking."""
        mind_state = MindState(owner_id="me")

        # High threat agent
        model1 = mind_state.get_or_create("agent_1")
        model1.estimated_traits["aggression"] = 0.9
        model1.estimated_disposition = -0.5
        model1.times_attacked_me = 2

        # Low threat agent
        model2 = mind_state.get_or_create("agent_2")
        model2.estimated_traits["aggression"] = 0.2
        model2.estimated_disposition = 0.6
        model2.times_helped_me = 1

        threats = StrategicReasoner.rank_agents_by_threat(mind_state)

        assert len(threats) == 2
        assert threats[0][0] == "agent_1"  # Highest threat first
        assert threats[0][1] > threats[1][1]

    def test_suggest_social_action_attack_threat(self):
        """Test that we attack nearby high-threat agents."""
        mind_state = MindState(owner_id="me")
        model = mind_state.get_or_create("agent_1")
        model.estimated_disposition = -0.8
        model.times_attacked_me = 2
        model.last_observed_position = (5, 5)

        action_type, params = StrategicReasoner.suggest_social_action(
            mind_state,
            "agent_1",
            own_inventory={},
            own_needs={"health": 80.0},
            own_position=(5, 5),  # Same tile
        )

        assert action_type == "attack"
        assert params["target_agent_id"] == "agent_1"

    def test_suggest_social_action_share(self):
        """Test sharing suggestion with friendly adjacent agent."""
        mind_state = MindState(owner_id="me")
        model = mind_state.get_or_create("agent_1")
        model.estimated_disposition = 0.7
        model.times_helped_me = 1
        model.last_observed_position = (5, 6)  # Adjacent

        action_type, params = StrategicReasoner.suggest_social_action(
            mind_state,
            "agent_1",
            own_inventory={"berries": 5},
            own_needs={"hunger": 70.0, "health": 80.0},
            own_position=(5, 5),
        )

        assert action_type == "share"
        assert params["item"] == "berries"


class TestTheoryOfMindStrategy:
    """Test TheoryOfMindStrategy."""

    def test_initialization(self):
        """Test strategy initializes with MindState."""
        strategy = TheoryOfMindStrategy(agent_id="me")

        assert strategy.mind_state.owner_id == "me"
        assert len(strategy.mind_state.models) == 0

    def test_form_intention_critical_survival(self):
        """Test that critical needs override social reasoning."""
        strategy = TheoryOfMindStrategy(agent_id="me")

        sensation = Sensation(
            tick=10,
            own_needs={"energy": 5.0, "hunger": 50.0},
            own_position=(5, 5),
            own_inventory={},
            visible_tiles=[],
            visible_agents=[],
            own_traits={"aggression": 0.5, "risk_tolerance": 0.5},
        )

        reflection = Reflection(
            last_action_succeeded=True,
            need_trends={},
        )

        intention = strategy.form_intention(sensation, reflection)

        assert intention.primary_goal == "rest"
        assert intention.confidence > 0.8

    def test_form_intention_avoid_threat(self):
        """Test avoiding hostile agents."""
        strategy = TheoryOfMindStrategy(agent_id="me")

        # Pre-populate mind state with hostile agent
        model = strategy.mind_state.get_or_create("agent_1")
        model.estimated_disposition = -0.8
        model.times_attacked_me = 2
        model.last_observed_position = (5, 7)
        model.estimated_traits["aggression"] = 0.9

        sensation = Sensation(
            tick=10,
            own_needs={"energy": 60.0, "hunger": 60.0, "health": 50.0},
            own_position=(5, 5),
            own_inventory={},
            visible_tiles=[],
            visible_agents=[
                AgentSummary(
                    agent_id="agent_1",
                    position=(5, 7),
                    apparent_health="healthy",
                    is_carrying_items=False,
                )
            ],
            own_traits={"aggression": 0.3},
        )

        reflection = Reflection(
            last_action_succeeded=True,
            need_trends={},
        )

        intention = strategy.form_intention(sensation, reflection)

        # Should avoid the threat
        assert intention.primary_goal == "avoid_agent"
        assert intention.target_agent_id == "agent_1"

    def test_form_intention_share_with_friendly(self):
        """Test sharing with friendly adjacent agents."""
        strategy = TheoryOfMindStrategy(agent_id="me")

        # Pre-populate with friendly agent
        model = strategy.mind_state.get_or_create("agent_1")
        model.estimated_disposition = 0.7
        model.times_helped_me = 2
        model.last_observed_position = (5, 6)

        sensation = Sensation(
            tick=10,
            own_needs={"energy": 80.0, "hunger": 70.0, "health": 90.0},
            own_position=(5, 5),
            own_inventory={"berries": 5},
            visible_tiles=[],
            visible_agents=[
                AgentSummary(
                    agent_id="agent_1",
                    position=(5, 6),
                    apparent_health="healthy",
                    is_carrying_items=False,
                )
            ],
            own_traits={"resource_sharing": 0.7},
        )

        reflection = Reflection(
            last_action_succeeded=True,
            need_trends={},
        )

        intention = strategy.form_intention(sensation, reflection)

        # Should share with friendly agent
        assert intention.primary_goal == "share_with_agent"
        assert intention.target_agent_id == "agent_1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
