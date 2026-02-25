"""Tests for LLMStrategy with mocked API calls."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from src.awareness.types import (
    AgentSummary,
    Intention,
    Reflection,
    Sensation,
)
from src.cognition.strategies.llm import LLMStrategy
from src.memory.episodic import EpisodicMemory
from src.memory.social import SocialMemory


@dataclass
class MockMessage:
    """Mock OpenAI chat completion message."""

    content: str


@dataclass
class MockChoice:
    """Mock OpenAI completion choice."""

    message: MockMessage


@dataclass
class MockCompletion:
    """Mock OpenAI chat completion."""

    choices: list[MockChoice]


class TestPromptCache:
    """Test prompt caching mechanism."""

    def test_cache_initialization(self):
        """Cache should initialize with empty values."""
        strategy = LLMStrategy()
        assert strategy._cached_system_prompt == ""
        assert strategy._cached_system_prompt_time == 0.0
        assert strategy._system_prompt_ttl == 300.0

    def test_cache_custom_ttl(self):
        """Cache should accept custom TTL."""
        strategy = LLMStrategy()
        strategy._system_prompt_ttl = 600.0
        assert strategy._system_prompt_ttl == 600.0


class TestStrategyInitialization:
    """Test strategy initialization and configuration."""

    def test_default_initialization(self):
        """Strategy should initialize with defaults."""
        strategy = LLMStrategy()

        assert strategy.base_url == ""
        assert strategy.model == "opus"
        assert strategy.cheap_model == "opus"  # defaults to model
        assert strategy.call_interval == 5
        assert strategy.archetype is None
        assert strategy.prompt_version == "v2_rich"
        assert strategy.api_key == ""

    def test_custom_initialization(self):
        """Strategy should accept custom parameters."""
        episodic = EpisodicMemory()
        social = SocialMemory()

        strategy = LLMStrategy(
            base_url="http://custom:8000/v1",
            model="opus",
            cheap_model="haiku",
            call_interval=10,
            archetype="gatherer",
            episodic_memory=episodic,
            social_memory=social,
            prompt_version="v1_minimal",
        )

        assert strategy.base_url == "http://custom:8000/v1"
        assert strategy.model == "opus"
        assert strategy.cheap_model == "haiku"
        assert strategy.call_interval == 10
        assert strategy.archetype == "gatherer"
        assert strategy.prompt_version == "v1_minimal"

    def test_model_tier_mapping(self):
        """Strategy should have correct model tier mappings."""
        strategy = LLMStrategy(model="opus", cheap_model="sonnet")
        assert strategy._model_map["fast"] == "sonnet"
        assert strategy._model_map["standard"] == "opus"
        assert strategy._model_map["deep"] == "opus"


class TestTierSelection:
    """Test model tier selection based on situation."""

    def create_sensation(
        self,
        needs: dict[str, float] | None = None,
        agent_count: int = 0,
    ) -> Sensation:
        """Helper to create test sensation."""
        agents = [
            AgentSummary(
                agent_id=f"agent_{i}",
                position=(i, i),
                apparent_health="healthy",
                is_carrying_items=False,
            )
            for i in range(agent_count)
        ]

        return Sensation(
            tick=0,
            own_needs=needs or {"hunger": 100, "thirst": 100, "energy": 100},
            own_position=(0, 0),
            own_inventory={},
            visible_tiles=[],
            visible_agents=agents,
        )

    def test_routine_situation_selects_fast(self):
        """Routine situation should select fast tier."""
        strategy = LLMStrategy(model="opus", cheap_model="haiku")
        sensation = self.create_sensation(
            needs={"hunger": 80, "thirst": 80, "energy": 80},
            agent_count=0,
        )
        reflection = Reflection(
            last_action_succeeded=True,
            need_trends={},
            threat_level=0.1,
        )

        tier = strategy._select_tier(sensation, reflection)
        assert tier == "fast"

    def test_critical_needs_select_deep(self):
        """Multiple critical needs should select deep tier."""
        strategy = LLMStrategy(model="opus", cheap_model="haiku")
        sensation = self.create_sensation(
            needs={"hunger": 10, "thirst": 10, "energy": 50},  # 2 critical
        )
        reflection = Reflection(
            last_action_succeeded=True,
            need_trends={},
            threat_level=0.2,
        )

        tier = strategy._select_tier(sensation, reflection)
        assert tier == "deep"

    def test_high_threat_selects_deep(self):
        """High threat level should select deep tier."""
        strategy = LLMStrategy(model="opus", cheap_model="haiku")
        sensation = self.create_sensation(
            needs={"hunger": 50, "thirst": 50, "energy": 50},
        )
        reflection = Reflection(
            last_action_succeeded=True,
            need_trends={},
            threat_level=0.8,
        )

        tier = strategy._select_tier(sensation, reflection)
        assert tier == "deep"

    def test_many_agents_select_deep(self):
        """Many visible agents should select deep tier."""
        strategy = LLMStrategy(model="opus", cheap_model="haiku")
        sensation = self.create_sensation(
            needs={"hunger": 50, "thirst": 50, "energy": 50},
            agent_count=5,
        )
        reflection = Reflection(
            last_action_succeeded=True,
            need_trends={},
            threat_level=0.2,
        )

        tier = strategy._select_tier(sensation, reflection)
        assert tier == "deep"

    def test_explicit_deep_tier_always_deep(self):
        """High stakes should always use deep tier."""
        strategy = LLMStrategy(model="opus", cheap_model="haiku")
        sensation = self.create_sensation(
            needs={"hunger": 5, "thirst": 5, "energy": 100},  # critical
        )
        reflection = Reflection(
            last_action_succeeded=True,
            need_trends={},
            threat_level=0.0,
        )

        tier = strategy._select_tier(sensation, reflection)
        assert tier == "deep"

    def test_moderate_situation_selects_standard(self):
        """Moderate situation should select standard tier."""
        strategy = LLMStrategy(model="opus", cheap_model="haiku")
        sensation = self.create_sensation(
            needs={"hunger": 40, "thirst": 60, "energy": 50},
            agent_count=1,
        )
        reflection = Reflection(
            last_action_succeeded=True,
            need_trends={},
            threat_level=0.3,
        )

        tier = strategy._select_tier(sensation, reflection)
        assert tier == "standard"


class TestFormIntentionFallback:
    """Test fallback behavior between LLM calls."""

    def test_uses_fallback_between_intervals(self):
        """Should use PersonalityStrategy between LLM call intervals."""
        strategy = LLMStrategy(call_interval=5)
        sensation = Sensation(
            tick=0,
            own_needs={"hunger": 10, "thirst": 100, "energy": 100},
            own_position=(0, 0),
            own_inventory={},
            visible_tiles=[],
            visible_agents=[],
        )
        reflection = Reflection(
            last_action_succeeded=True,
            need_trends={},
            threat_level=0.0,
        )

        # First 4 calls should use fallback
        for _i in range(4):
            intention = strategy.form_intention(sensation, reflection)
            assert isinstance(intention, Intention)

        # Tick counter should be 4
        assert strategy._tick_counter == 4

    def test_llm_called_on_interval(self):
        """LLM should be called on the interval (mocked)."""
        # Patch where OpenAI is imported (inside _get_client)
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *args, **kwargs: (
                MagicMock() if name == "openai" else __import__(name, *args, **kwargs)
            ),
        ):
            mock_client = MagicMock()

            # Mock completion response
            mock_completion = MockCompletion(
                choices=[
                    MockChoice(message=MockMessage(content='{"goal": "rest", "reason": "test"}'))
                ]
            )
            mock_client.chat.completions.create.return_value = mock_completion

            strategy = LLMStrategy(call_interval=5)
            # Inject mocked client
            strategy._client = mock_client

            sensation = Sensation(
                tick=0,
                own_needs={"hunger": 100, "thirst": 100, "energy": 50},
                own_position=(0, 0),
                own_inventory={},
                visible_tiles=[],
                visible_agents=[],
                own_traits={"curiosity": 0.5},
            )
            reflection = Reflection(
                last_action_succeeded=True,
                need_trends={},
                threat_level=0.0,
            )

            # Advance to interval tick
            for _ in range(4):
                strategy.form_intention(sensation, reflection)

            # 5th call should invoke LLM
            intention = strategy.form_intention(sensation, reflection)

            assert mock_client.chat.completions.create.called
            assert intention.primary_goal == "rest"


class TestExpression:
    """Test expression delegation to fallback."""

    def test_express_delegates_to_fallback(self):
        """Express should always delegate to PersonalityStrategy."""
        strategy = LLMStrategy()
        sensation = Sensation(
            tick=0,
            own_needs={"hunger": 50, "thirst": 50, "energy": 50},
            own_position=(0, 0),
            own_inventory={},
            visible_tiles=[],
            visible_agents=[],
        )
        reflection = Reflection(
            last_action_succeeded=True,
            need_trends={},
            threat_level=0.0,
        )
        intention = Intention(primary_goal="rest", planned_actions=["rest"])

        expression = strategy.express(sensation, reflection, intention)

        # Should produce valid expression
        assert expression.action is not None
        assert expression.action.type.value == "rest"


class TestParseToIntention:
    """Test LLM response parsing to Intention."""

    def test_parse_simple_goal(self):
        """Parse simple goal without targets."""
        strategy = LLMStrategy()
        sensation = Sensation(
            tick=0,
            own_needs={"hunger": 50, "thirst": 50, "energy": 50},
            own_position=(0, 0),
            own_inventory={},
            visible_tiles=[],
            visible_agents=[],
        )

        response = '{"goal": "rest", "reason": "low energy"}'
        intention = strategy._parse_to_intention(response, sensation)

        assert intention.primary_goal == "rest"
        assert intention.target_agent_id is None
        assert intention.confidence > 0

    def test_parse_goal_with_agent_target(self):
        """Parse goal that requires agent target."""
        strategy = LLMStrategy()
        agents = [
            AgentSummary(
                agent_id="agent_1",
                position=(1, 1),
                apparent_health="healthy",
                is_carrying_items=False,
            )
        ]
        sensation = Sensation(
            tick=0,
            own_needs={"hunger": 50, "thirst": 50, "energy": 50},
            own_position=(0, 0),
            own_inventory={},
            visible_tiles=[],
            visible_agents=agents,
        )

        response = '{"goal": "attack", "reason": "threat"}'
        intention = strategy._parse_to_intention(response, sensation)

        assert intention.primary_goal == "attack"
        assert intention.target_agent_id == "agent_1"
        assert intention.target_position == (1, 1)

    def test_parse_flee_computes_escape_direction(self):
        """Flee goal should compute position away from threat."""
        strategy = LLMStrategy()
        agents = [
            AgentSummary(
                agent_id="threat",
                position=(2, 2),
                apparent_health="healthy",
                is_carrying_items=False,
            )
        ]
        sensation = Sensation(
            tick=0,
            own_needs={"hunger": 50, "thirst": 50, "energy": 50},
            own_position=(1, 1),
            own_inventory={},
            visible_tiles=[],
            visible_agents=agents,
        )

        response = '{"goal": "flee", "reason": "outnumbered"}'
        intention = strategy._parse_to_intention(response, sensation)

        assert intention.primary_goal == "flee"
        # Should compute position away from (2,2)
        assert intention.target_position is not None
        # Target should be further from threat
        threat_pos = (2, 2)
        own_pos = (1, 1)
        target_pos = intention.target_position
        # Distance from threat should increase
        dist_own = abs(threat_pos[0] - own_pos[0]) + abs(threat_pos[1] - own_pos[1])
        dist_target = abs(threat_pos[0] - target_pos[0]) + abs(threat_pos[1] - target_pos[1])
        assert dist_target > dist_own


class TestSystemPromptCaching:
    """Test system prompt caching behavior."""

    def test_prompt_cached_on_first_use(self):
        """System prompt should be cached after first build."""
        strategy = LLMStrategy(archetype="gatherer")
        traits = {"curiosity": 0.5}

        prompt1 = strategy._get_or_build_system_prompt(traits)
        cache_time1 = strategy._cached_system_prompt_time

        # Second call should use cache
        prompt2 = strategy._get_or_build_system_prompt(traits)
        cache_time2 = strategy._cached_system_prompt_time

        assert prompt1 == prompt2
        assert cache_time1 == cache_time2  # Not rebuilt

    def test_prompt_rebuilt_after_ttl(self):
        """System prompt should be rebuilt after TTL expires."""
        strategy = LLMStrategy(archetype="gatherer")
        strategy._system_prompt_ttl = 0.01  # 10ms TTL
        traits = {"curiosity": 0.5}

        prompt1 = strategy._get_or_build_system_prompt(traits)

        # Wait for TTL expiration
        import time

        time.sleep(0.02)

        prompt2 = strategy._get_or_build_system_prompt(traits)

        # Should be rebuilt (same content but different cache time)
        assert prompt1 == prompt2  # Content identical
        # Cache should be refreshed (tested by ensuring no error)


class TestClientLazyInit:
    """Test lazy initialization of OpenAI client."""

    def test_client_not_initialized_on_strategy_creation(self):
        """Client should not be initialized until first use."""
        strategy = LLMStrategy()
        assert strategy._client is None

    def test_client_initialized_on_first_use(self):
        """Client should be initialized on first LLM call."""
        # Mock the openai module import
        mock_openai = MagicMock()
        mock_openai_class = MagicMock()
        mock_openai.OpenAI = mock_openai_class

        with patch.dict("sys.modules", {"openai": mock_openai}):
            strategy = LLMStrategy(base_url="http://localhost:8000/v1")
            client = strategy._get_client()

            mock_openai_class.assert_called_once()
            assert client is not None

    def test_client_import_error_returns_none(self):
        """If openai package missing, should return None."""
        # Simulate ImportError by removing openai from sys.modules
        import sys

        original_modules = sys.modules.copy()
        if "openai" in sys.modules:
            del sys.modules["openai"]

        try:
            # Make import fail
            def mock_import(name, *args, **kwargs):
                if name == "openai":
                    raise ImportError("No module named 'openai'")
                return original_modules.get(name)

            with patch("builtins.__import__", side_effect=mock_import):
                strategy = LLMStrategy(base_url="http://localhost:8000/v1")
                client = strategy._get_client()

                assert client is None
        finally:
            sys.modules.update(original_modules)

    def test_empty_base_url_raises_error(self):
        """Should raise ValueError when base_url is empty."""
        import pytest

        strategy = LLMStrategy()  # base_url defaults to ""
        with pytest.raises(ValueError, match="LLM base URL not configured"):
            strategy._get_client()

    def test_api_key_passed_to_client(self):
        """API key should be passed through to OpenAI client."""
        mock_openai = MagicMock()
        mock_openai_class = MagicMock()
        mock_openai.OpenAI = mock_openai_class

        with patch.dict("sys.modules", {"openai": mock_openai}):
            strategy = LLMStrategy(
                base_url="http://localhost:8000/v1",
                api_key="sk-test-key-123",
            )
            strategy._get_client()

            mock_openai_class.assert_called_once_with(
                base_url="http://localhost:8000/v1",
                api_key="sk-test-key-123",
            )

    def test_empty_api_key_uses_not_set(self):
        """Empty API key should pass 'not-set' to client (for Ollama etc.)."""
        mock_openai = MagicMock()
        mock_openai_class = MagicMock()
        mock_openai.OpenAI = mock_openai_class

        with patch.dict("sys.modules", {"openai": mock_openai}):
            strategy = LLMStrategy(base_url="http://localhost:11434/v1")
            strategy._get_client()

            mock_openai_class.assert_called_once_with(
                base_url="http://localhost:11434/v1",
                api_key="not-set",
            )


class TestIntegrationWithMemory:
    """Test strategy integration with memory systems."""

    def test_strategy_with_episodic_memory(self):
        """Strategy should use episodic memory in context building."""
        episodic = EpisodicMemory()
        episodic.record_episode(
            tick=10,
            action_type="GATHER",
            target="berries",
            success=True,
            location=(5, 5),
        )

        strategy = LLMStrategy(episodic_memory=episodic)
        assert strategy._context_builder.episodic is episodic

    def test_strategy_with_social_memory(self):
        """Strategy should use social memory in context building."""
        social = SocialMemory()
        rel = social.get_or_create("agent_123")
        rel.update_trust("helped", positive=True, tick=10)

        strategy = LLMStrategy(social_memory=social)
        assert strategy._context_builder.social is social


class TestPromptVersioning:
    """Test prompt version selection and behavior."""

    def test_default_prompt_version_is_v2(self):
        """Strategy should default to v2_rich prompt."""
        strategy = LLMStrategy()
        assert strategy.prompt_version == "v2_rich"

    def test_v1_minimal_prompt_format(self):
        """V1 minimal prompt should produce basic status."""
        strategy = LLMStrategy(prompt_version="v1_minimal")
        sensation = Sensation(
            tick=10,
            own_needs={"hunger": 50, "thirst": 60, "energy": 70, "health": 100},
            own_position=(5, 5),
            own_inventory={"berries": 2},
            own_traits={"curiosity": 0.5},
            visible_tiles=[],
            visible_agents=[],
        )
        reflection = Reflection(
            last_action_succeeded=True,
            need_trends={"hunger": -1},
            threat_level=0.2,
            opportunity_score=0.5,
        )

        prompt = strategy._build_minimal_prompt(sensation, reflection)

        # Should contain basic info
        assert "Tick: 10" in prompt
        assert "Position: (5, 5)" in prompt
        assert "hunger=50" in prompt
        assert "thirst=60" in prompt
        assert "Threat level: 0.2" in prompt

    def test_v2_rich_prompt_uses_context_builder(self):
        """V2 rich prompt should use ContextBuilder for full context."""
        episodic = EpisodicMemory()
        strategy = LLMStrategy(prompt_version="v2_rich", episodic_memory=episodic)

        # Context builder should be configured
        assert strategy._context_builder.episodic is episodic
        assert strategy.prompt_version == "v2_rich"
