"""Tests for prompt templates and system prompt building."""

from __future__ import annotations

from src.cognition.prompts.templates import (
    ARCHETYPE_PERSONAS,
    TRAIT_DESCRIPTIONS,
    _describe_trait,
    build_system_prompt,
)


class TestArchetypePersonas:
    """Test archetype persona definitions."""

    def test_all_archetypes_defined(self):
        """All expected archetypes should have persona text."""
        expected = {"gatherer", "explorer", "guardian", "predator", "diplomat", "survivalist"}
        assert set(ARCHETYPE_PERSONAS.keys()) == expected

    def test_persona_text_is_descriptive(self):
        """Each persona should be a non-empty string."""
        for archetype, text in ARCHETYPE_PERSONAS.items():
            assert isinstance(text, str)
            assert len(text) > 50, f"{archetype} persona too short"
            assert "you are" in text.lower(), f"{archetype} not written in second person"


class TestTraitDescriptions:
    """Test trait description mappings."""

    def test_all_traits_defined(self):
        """All expected traits should have descriptions."""
        expected = {
            "cooperation_tendency",
            "curiosity",
            "risk_tolerance",
            "resource_sharing",
            "aggression",
            "sociability",
        }
        assert set(TRAIT_DESCRIPTIONS.keys()) == expected

    def test_trait_has_three_levels(self):
        """Each trait should have low/medium/high descriptions."""
        for trait, levels in TRAIT_DESCRIPTIONS.items():
            assert set(levels.keys()) == {"low", "medium", "high"}, f"{trait} missing levels"
            for level, desc in levels.items():
                assert isinstance(desc, str)
                assert len(desc) > 10, f"{trait}.{level} too short"


class TestDescribeTrait:
    """Test trait value to description conversion."""

    def test_low_value_maps_to_low_description(self):
        """Values < 0.35 should map to 'low' description."""
        result = _describe_trait("curiosity", 0.2)
        expected = TRAIT_DESCRIPTIONS["curiosity"]["low"]
        assert result == expected

    def test_medium_value_maps_to_medium_description(self):
        """Values 0.35-0.65 should map to 'medium' description."""
        result = _describe_trait("aggression", 0.5)
        expected = TRAIT_DESCRIPTIONS["aggression"]["medium"]
        assert result == expected

    def test_high_value_maps_to_high_description(self):
        """Values > 0.65 should map to 'high' description."""
        result = _describe_trait("sociability", 0.8)
        expected = TRAIT_DESCRIPTIONS["sociability"]["high"]
        assert result == expected

    def test_boundary_values(self):
        """Test exact boundary values."""
        # 0.35 should be medium (>= 0.35)
        result = _describe_trait("curiosity", 0.35)
        assert result == TRAIT_DESCRIPTIONS["curiosity"]["medium"]

        # 0.65 should be high (>= 0.65)
        result = _describe_trait("curiosity", 0.65)
        assert result == TRAIT_DESCRIPTIONS["curiosity"]["high"]

    def test_unknown_trait_fallback(self):
        """Unknown traits should return a generic description."""
        result = _describe_trait("unknown_trait", 0.7)
        assert "unknown_trait" in result
        assert "0.7" in result


class TestBuildSystemPrompt:
    """Test system prompt generation."""

    def test_minimal_prompt_without_customization(self):
        """Prompt without archetype/traits should still be valid."""
        prompt = build_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 100
        assert "survival simulation" in prompt.lower()
        assert "JSON" in prompt

    def test_prompt_with_archetype(self):
        """Archetype should be included in prompt."""
        prompt = build_system_prompt(archetype="gatherer")
        assert "gatherer" in prompt.lower() or ARCHETYPE_PERSONAS["gatherer"] in prompt

    def test_prompt_with_traits(self):
        """Traits should be described in prompt."""
        traits = {"curiosity": 0.9, "aggression": 0.1}
        prompt = build_system_prompt(traits=traits)

        # Should contain trait descriptions
        assert "curious" in prompt.lower() or "driven to discover" in prompt.lower()
        assert "avoid conflict" in prompt.lower() or "aggression" in prompt.lower()

    def test_prompt_with_valid_goals(self):
        """Valid goals list should be included."""
        goals = ["rest", "explore", "attack"]
        prompt = build_system_prompt(valid_goals=goals)

        assert "rest" in prompt
        assert "explore" in prompt
        assert "attack" in prompt

    def test_prompt_with_all_options(self):
        """Prompt with all options should be comprehensive."""
        traits = {"curiosity": 0.7, "sociability": 0.8}
        goals = ["rest", "socialize", "explore"]

        prompt = build_system_prompt(
            archetype="diplomat",
            traits=traits,
            valid_goals=goals,
        )

        assert "diplomat" in prompt.lower() or ARCHETYPE_PERSONAS["diplomat"] in prompt
        assert "rest" in prompt
        assert "socialize" in prompt
        assert len(prompt) > 300

    def test_prompt_structure_sections(self):
        """Prompt should have clear sections."""
        prompt = build_system_prompt(
            archetype="explorer",
            traits={"curiosity": 0.9},
            valid_goals=["explore"],
        )

        # Should have section headers (markdown-style)
        assert "##" in prompt or "Your" in prompt
        assert "JSON" in prompt  # Format section
        assert "goal" in prompt.lower()  # Response format

    def test_invalid_archetype_ignored(self):
        """Invalid archetype should be silently ignored."""
        prompt = build_system_prompt(archetype="nonexistent")
        # Should still generate valid prompt
        assert len(prompt) > 100
        assert "survival simulation" in prompt.lower()

    def test_empty_traits_handled(self):
        """Empty traits dict should not break prompt."""
        prompt = build_system_prompt(traits={})
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_empty_goals_handled(self):
        """Empty goals list should not break prompt."""
        prompt = build_system_prompt(valid_goals=[])
        assert isinstance(prompt, str)
        assert len(prompt) > 100


class TestPromptConsistency:
    """Test that prompts are deterministic and consistent."""

    def test_same_inputs_produce_same_output(self):
        """Same inputs should produce identical prompts."""
        traits = {"curiosity": 0.5, "aggression": 0.3}
        goals = ["rest", "explore"]

        prompt1 = build_system_prompt(archetype="gatherer", traits=traits, valid_goals=goals)
        prompt2 = build_system_prompt(archetype="gatherer", traits=traits, valid_goals=goals)

        assert prompt1 == prompt2

    def test_different_archetypes_produce_different_prompts(self):
        """Different archetypes should produce different prompts."""
        prompt1 = build_system_prompt(archetype="gatherer")
        prompt2 = build_system_prompt(archetype="predator")

        assert prompt1 != prompt2

    def test_different_trait_values_produce_different_prompts(self):
        """Different trait values should produce different prompts."""
        prompt1 = build_system_prompt(traits={"curiosity": 0.2})
        prompt2 = build_system_prompt(traits={"curiosity": 0.8})

        assert prompt1 != prompt2
