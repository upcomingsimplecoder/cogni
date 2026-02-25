"""Tests for LLM response parser."""

from __future__ import annotations

from src.cognition.prompts.parser import (
    VALID_GOALS,
    LLMDecision,
    LLMResponseParser,
)


class TestValidGoals:
    """Test valid goals list."""

    def test_valid_goals_is_comprehensive(self):
        """Valid goals should cover all major agent behaviors."""
        expected_goals = {
            "rest",
            "satisfy_hunger",
            "satisfy_thirst",
            "explore",
            "socialize",
            "attack",
            "share_resources",
            "gather_opportunistic",
            "flee",
            "wait",
        }
        assert set(VALID_GOALS) == expected_goals


class TestLLMDecision:
    """Test LLMDecision dataclass."""

    def test_minimal_decision(self):
        """Decision with only required fields."""
        decision = LLMDecision(goal="rest", reason="tired")
        assert decision.goal == "rest"
        assert decision.reason == "tired"
        assert decision.target_agent is None
        assert decision.confidence == 1.0

    def test_full_decision(self):
        """Decision with all fields."""
        decision = LLMDecision(
            goal="attack",
            reason="defending territory",
            target_agent="agent_123",
            confidence=0.9,
            raw_text='{"goal": "attack"}',
        )
        assert decision.goal == "attack"
        assert decision.target_agent == "agent_123"
        assert decision.confidence == 0.9
        assert "attack" in decision.raw_text


class TestParserCleanJSON:
    """Test parser on clean JSON responses."""

    def setup_method(self):
        self.parser = LLMResponseParser()

    def test_minimal_clean_json(self):
        """Parse minimal valid JSON."""
        response = '{"goal": "rest", "reason": "low energy"}'
        decision = self.parser.parse(response)

        assert decision.goal == "rest"
        assert decision.reason == "low energy"
        assert decision.confidence >= 0.9

    def test_json_with_target_agent(self):
        """Parse JSON with target_agent field."""
        response = '{"goal": "attack", "reason": "threat detected", "target_agent": "agent_456"}'
        decision = self.parser.parse(response)

        assert decision.goal == "attack"
        assert decision.target_agent == "agent_456"

    def test_json_with_null_target(self):
        """Parse JSON with explicit null target."""
        response = '{"goal": "explore", "reason": "need resources", "target_agent": null}'
        decision = self.parser.parse(response)

        assert decision.goal == "explore"
        assert decision.target_agent is None

    def test_json_with_extra_fields(self):
        """Parser should ignore extra fields."""
        response = '{"goal": "rest", "reason": "tired", "extra": "ignored", "confidence": 0.8}'
        decision = self.parser.parse(response)

        assert decision.goal == "rest"
        # Parser uses its own confidence, not from JSON


class TestParserMarkdownFences:
    """Test parser on JSON wrapped in markdown code fences."""

    def setup_method(self):
        self.parser = LLMResponseParser()

    def test_json_in_code_fence(self):
        """Parse JSON wrapped in ```json fence."""
        response = '```json\n{"goal": "explore", "reason": "map area"}\n```'
        decision = self.parser.parse(response)

        assert decision.goal == "explore"
        assert decision.reason == "map area"

    def test_json_in_plain_fence(self):
        """Parse JSON wrapped in ``` fence without language."""
        response = '```\n{"goal": "gather_opportunistic", "reason": "berries nearby"}\n```'
        decision = self.parser.parse(response)

        assert decision.goal == "gather_opportunistic"

    def test_json_with_text_before_fence(self):
        """Parser should extract JSON even with prose before."""
        response = 'Here is my decision:\n```json\n{"goal": "rest", "reason": "tired"}\n```'
        decision = self.parser.parse(response)

        assert decision.goal == "rest"


class TestParserSynonymMapping:
    """Test parser's synonym mapping for common goal variations."""

    def setup_method(self):
        self.parser = LLMResponseParser()

    def test_eat_maps_to_satisfy_hunger(self):
        """'eat' should map to 'satisfy_hunger'."""
        response = '{"goal": "eat", "reason": "hungry"}'
        decision = self.parser.parse(response)

        assert decision.goal == "satisfy_hunger"

    def test_drink_maps_to_satisfy_thirst(self):
        """'drink' should map to 'satisfy_thirst'."""
        response = '{"goal": "drink", "reason": "thirsty"}'
        decision = self.parser.parse(response)

        assert decision.goal == "satisfy_thirst"

    def test_sleep_maps_to_rest(self):
        """'sleep' should map to 'rest'."""
        response = '{"goal": "sleep", "reason": "exhausted"}'
        decision = self.parser.parse(response)

        assert decision.goal == "rest"

    def test_gather_maps_to_gather_opportunistic(self):
        """'gather' should map to 'gather_opportunistic'."""
        response = '{"goal": "gather", "reason": "resources nearby"}'
        decision = self.parser.parse(response)

        assert decision.goal == "gather_opportunistic"

    def test_fight_maps_to_attack(self):
        """'fight' should map to 'attack'."""
        response = '{"goal": "fight", "reason": "hostile agent"}'
        decision = self.parser.parse(response)

        assert decision.goal == "attack"


class TestParserMalformedJSON:
    """Test parser on malformed or partial JSON."""

    def setup_method(self):
        self.parser = LLMResponseParser()

    def test_missing_closing_brace(self):
        """Parser should extract goal even if JSON incomplete."""
        response = '{"goal": "explore", "reason": "need to find water"'
        decision = self.parser.parse(response)

        assert decision.goal == "explore"

    def test_missing_quotes_on_field(self):
        """Parser should handle missing quotes with regex fallback."""
        response = '{goal: "rest", reason: "tired"}'
        decision = self.parser.parse(response)

        assert decision.goal == "rest"

    def test_goal_only_no_reason(self):
        """Parser should handle missing reason field."""
        response = '{"goal": "wait"}'
        decision = self.parser.parse(response)

        assert decision.goal == "wait"
        assert "no reason" in decision.reason.lower() or decision.reason == "No reason provided"


class TestParserVerboseResponses:
    """Test parser on verbose/chatty LLM responses."""

    def setup_method(self):
        self.parser = LLMResponseParser()

    def test_goal_in_prose(self):
        """Parser should extract goal keyword from prose."""
        response = "I think I should rest because my energy is low."
        decision = self.parser.parse(response)

        assert decision.goal == "rest"
        assert decision.confidence < 0.9  # Lower confidence for keyword extraction

    def test_multiple_goal_keywords_prioritizes_specificity(self):
        """More specific goals should be prioritized."""
        # "satisfy_hunger" is more specific than "eat" synonym
        response = "I need to satisfy my hunger by eating berries."
        decision = self.parser.parse(response)

        assert decision.goal == "satisfy_hunger"

    def test_no_recognizable_goal_defaults_to_wait(self):
        """Unparseable responses should default to 'wait'."""
        response = "This is completely unrelated text with no goals."
        decision = self.parser.parse(response)

        assert decision.goal == "wait"
        assert decision.confidence < 0.5


class TestParserEdgeCases:
    """Test parser on edge cases and adversarial inputs."""

    def setup_method(self):
        self.parser = LLMResponseParser()

    def test_empty_string(self):
        """Empty response should default to wait."""
        decision = self.parser.parse("")

        assert decision.goal == "wait"
        assert decision.confidence < 0.5

    def test_only_whitespace(self):
        """Whitespace-only response should default to wait."""
        decision = self.parser.parse("   \n\n\t  ")

        assert decision.goal == "wait"

    def test_json_with_wrong_case(self):
        """Parser should handle case variations."""
        response = '{"goal": "EXPLORE", "reason": "Need resources"}'
        decision = self.parser.parse(response)

        assert decision.goal == "explore"

    def test_invalid_goal_in_json(self):
        """Invalid goal should be caught by validation."""
        response = '{"goal": "invalid_goal", "reason": "testing"}'
        decision = self.parser.parse(response)

        # Should default to wait due to invalid goal
        assert decision.goal == "wait"

    def test_very_long_response(self):
        """Parser should handle very long responses."""
        response = "A" * 5000 + '{"goal": "explore", "reason": "long text"}'
        decision = self.parser.parse(response)

        assert decision.goal == "explore"


class TestParserValidation:
    """Test decision validation logic."""

    def setup_method(self):
        self.parser = LLMResponseParser()

    def test_valid_goal_passes_validation(self):
        """Valid goals should pass without modification."""
        decision = LLMDecision(goal="rest", reason="tired")
        validated = self.parser.validate(decision)

        assert validated.goal == "rest"
        assert validated.confidence == 1.0

    def test_invalid_goal_normalized_to_wait(self):
        """Invalid goals should be normalized to 'wait'."""
        decision = LLMDecision(goal="invalid", reason="test")
        validated = self.parser.validate(decision)

        assert validated.goal == "wait"
        assert validated.confidence < 1.0  # Reduced confidence

    def test_empty_reason_filled_with_default(self):
        """Empty reason should be filled with default."""
        decision = LLMDecision(goal="rest", reason="")
        validated = self.parser.validate(decision)

        assert validated.reason == "No reason provided"
        assert validated.confidence < 1.0

    def test_synonym_goal_normalized(self):
        """Synonym goals should be normalized to valid goals."""
        decision = LLMDecision(goal="eat", reason="hungry")
        validated = self.parser.validate(decision)

        assert validated.goal == "satisfy_hunger"


class TestParserConsistency:
    """Test parser consistency and determinism."""

    def setup_method(self):
        self.parser = LLMResponseParser()

    def test_same_input_produces_same_output(self):
        """Parser should be deterministic."""
        response = '{"goal": "explore", "reason": "need water"}'

        decision1 = self.parser.parse(response)
        decision2 = self.parser.parse(response)

        assert decision1.goal == decision2.goal
        assert decision1.reason == decision2.reason
        assert decision1.confidence == decision2.confidence

    def test_different_inputs_produce_different_outputs(self):
        """Different inputs should produce different outputs."""
        decision1 = self.parser.parse('{"goal": "rest", "reason": "tired"}')
        decision2 = self.parser.parse('{"goal": "attack", "reason": "threat"}')

        assert decision1.goal != decision2.goal
