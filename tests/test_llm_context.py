"""Tests for context building from Sensation and Reflection."""

from __future__ import annotations

from dataclasses import dataclass

from src.awareness.types import (
    AgentSummary,
    InteractionOutcome,
    Reflection,
    Sensation,
    TileSummary,
)
from src.cognition.prompts.context import ContextBuilder
from src.memory.episodic import EpisodicMemory
from src.memory.social import SocialMemory


@dataclass
class MockMessage:
    """Mock message for testing."""

    sender_id: str
    content: str
    message_type: object = None


class TestContextBuilderInitialization:
    """Test context builder initialization."""

    def test_default_initialization(self):
        """Builder should initialize without memory."""
        builder = ContextBuilder()
        assert builder.episodic is None
        assert builder.social is None

    def test_initialization_with_memory(self):
        """Builder should accept memory systems."""
        episodic = EpisodicMemory()
        social = SocialMemory()
        builder = ContextBuilder(episodic_memory=episodic, social_memory=social)

        assert builder.episodic is episodic
        assert builder.social is social


class TestStatusSection:
    """Test agent status section building."""

    def test_basic_status(self):
        """Status section should include position, time, and needs."""
        builder = ContextBuilder()
        sensation = Sensation(
            tick=42,
            own_needs={"hunger": 60, "thirst": 70, "energy": 80, "health": 100},
            own_position=(5, 10),
            own_inventory={},
            visible_tiles=[],
            visible_agents=[],
            time_of_day="day",
        )

        section = builder._build_status_section(sensation)

        assert "42" in section  # Tick
        assert "(5, 10)" in section  # Position
        assert "day" in section  # Time
        assert "60" in section  # Hunger value
        assert "70" in section  # Thirst value

    def test_status_with_inventory(self):
        """Status should list inventory items."""
        builder = ContextBuilder()
        sensation = Sensation(
            tick=1,
            own_needs={"hunger": 50, "thirst": 50, "energy": 50},
            own_position=(0, 0),
            own_inventory={"berries": 3, "water": 2},
            visible_tiles=[],
            visible_agents=[],
        )

        section = builder._build_status_section(sensation)

        assert "berries" in section
        assert "3" in section
        assert "water" in section

    def test_status_with_empty_inventory(self):
        """Empty inventory should be noted."""
        builder = ContextBuilder()
        sensation = Sensation(
            tick=1,
            own_needs={"hunger": 50, "thirst": 50, "energy": 50},
            own_position=(0, 0),
            own_inventory={},
            visible_tiles=[],
            visible_agents=[],
        )

        section = builder._build_status_section(sensation)

        assert "empty" in section.lower()


class TestEnvironmentSection:
    """Test environment/tiles section building."""

    def test_no_visibility(self):
        """Should handle no visible tiles."""
        builder = ContextBuilder()
        sensation = Sensation(
            tick=1,
            own_needs={},
            own_position=(0, 0),
            own_inventory={},
            visible_tiles=[],
            visible_agents=[],
        )

        section = builder._build_environment_section(sensation)

        assert "no visibility" in section.lower()

    def test_tile_counts(self):
        """Should count and summarize tile types."""
        builder = ContextBuilder()
        tiles = [
            TileSummary(x=0, y=0, tile_type="grass", resources=[]),
            TileSummary(x=1, y=0, tile_type="grass", resources=[]),
            TileSummary(x=0, y=1, tile_type="water", resources=[("water_source", 10)]),
        ]
        sensation = Sensation(
            tick=1,
            own_needs={},
            own_position=(0, 0),
            own_inventory={},
            visible_tiles=tiles,
            visible_agents=[],
        )

        section = builder._build_environment_section(sensation)

        assert "grass" in section
        assert "water" in section
        assert "3" in section  # Total tiles visible

    def test_resource_summary(self):
        """Should summarize available resources."""
        builder = ContextBuilder()
        tiles = [
            TileSummary(x=0, y=0, tile_type="grass", resources=[("berry_bush", 5)]),
            TileSummary(x=1, y=0, tile_type="grass", resources=[("berry_bush", 3)]),
        ]
        sensation = Sensation(
            tick=1,
            own_needs={},
            own_position=(0, 0),
            own_inventory={},
            visible_tiles=tiles,
            visible_agents=[],
        )

        section = builder._build_environment_section(sensation)

        assert "berry_bush" in section
        assert "8" in section  # Total berries

    def test_nearest_resources(self):
        """Should identify nearest resources by type."""
        builder = ContextBuilder()
        tiles = [
            TileSummary(x=0, y=0, tile_type="grass", resources=[("berry_bush", 5)]),
            TileSummary(x=5, y=5, tile_type="grass", resources=[("berry_bush", 3)]),
            TileSummary(x=1, y=1, tile_type="water", resources=[("water_source", 10)]),
        ]
        sensation = Sensation(
            tick=1,
            own_needs={},
            own_position=(0, 0),
            own_inventory={},
            visible_tiles=tiles,
            visible_agents=[],
        )

        section = builder._build_environment_section(sensation)

        # Should identify nearest berry_bush at (0,0) dist 0
        # and water_source at (1,1) dist 2
        assert "berry_bush at (0, 0)" in section
        assert "water_source at (1, 1)" in section


class TestAgentsSection:
    """Test visible agents section building."""

    def test_no_agents(self):
        """Should handle no visible agents."""
        builder = ContextBuilder()
        sensation = Sensation(
            tick=1,
            own_needs={},
            own_position=(0, 0),
            own_inventory={},
            visible_tiles=[],
            visible_agents=[],
        )

        section = builder._build_agents_section(sensation)

        assert "none visible" in section.lower()

    def test_agent_summary(self):
        """Should list visible agents with details."""
        builder = ContextBuilder()
        agents = [
            AgentSummary(
                agent_id="agent_abc123",
                position=(2, 3),
                apparent_health="healthy",
                is_carrying_items=True,
            ),
            AgentSummary(
                agent_id="agent_def456",
                position=(1, 1),
                apparent_health="injured",
                is_carrying_items=False,
            ),
        ]
        sensation = Sensation(
            tick=1,
            own_needs={},
            own_position=(0, 0),
            own_inventory={},
            visible_tiles=[],
            visible_agents=agents,
        )

        section = builder._build_agents_section(sensation)

        assert "agent_ab" in section  # Truncated ID (8 chars)
        assert "healthy" in section
        assert "carrying items" in section
        assert "injured" in section
        assert "empty-handed" in section


class TestReflectionSection:
    """Test reflection section building."""

    def test_basic_reflection(self):
        """Should include action result and trends."""
        builder = ContextBuilder()
        reflection = Reflection(
            last_action_succeeded=True,
            need_trends={"hunger": "declining", "thirst": "stable"},
            threat_level=0.3,
            opportunity_score=0.7,
        )

        section = builder._build_reflection_section(reflection)

        assert "succeeded" in section
        assert "hunger=declining" in section
        assert "thirst=stable" in section
        assert "0.3" in section  # Threat level
        assert "0.7" in section  # Opportunity

    def test_failed_action(self):
        """Should note failed actions."""
        builder = ContextBuilder()
        reflection = Reflection(
            last_action_succeeded=False,
            need_trends={},
        )

        section = builder._build_reflection_section(reflection)

        assert "failed" in section

    def test_recent_interactions(self):
        """Should summarize recent interactions."""
        builder = ContextBuilder()
        reflection = Reflection(
            last_action_succeeded=True,
            need_trends={},
            recent_interaction_outcomes=[
                InteractionOutcome(
                    other_agent_id="agent1",
                    tick=10,
                    was_positive=True,
                    interaction_type="trade",
                ),
                InteractionOutcome(
                    other_agent_id="agent2",
                    tick=11,
                    was_positive=False,
                    interaction_type="attacked",
                ),
            ],
        )

        section = builder._build_reflection_section(reflection)

        assert "positive trade" in section
        assert "negative attacked" in section


class TestEpisodicSection:
    """Test episodic memory section building."""

    def test_no_episodic_memory(self):
        """Should return empty string if no episodic memory."""
        builder = ContextBuilder()
        section = builder._build_episodic_section()
        assert section == ""

    def test_empty_episodic_memory(self):
        """Should return empty string if no episodes."""
        episodic = EpisodicMemory()
        builder = ContextBuilder(episodic_memory=episodic)
        section = builder._build_episodic_section()
        assert section == ""

    def test_action_success_rates(self):
        """Should report success rates for actions."""
        episodic = EpisodicMemory()
        for i in range(5):
            episodic.record_episode(
                tick=i,
                action_type="GATHER",
                target="berries",
                success=i < 3,  # 3/5 success
                location=(i, i),
            )

        builder = ContextBuilder(episodic_memory=episodic)
        section = builder._build_episodic_section()

        assert "GATHER" in section
        assert "60%" in section or "3/5" in section or "0.6" in section

    def test_notable_episodes(self):
        """Should highlight failed or social episodes."""
        episodic = EpisodicMemory()
        episodic.record_episode(
            tick=10,
            action_type="ATTACK",
            target=None,
            success=False,
            location=(5, 5),
            involved_agent="enemy_agent",
        )

        builder = ContextBuilder(episodic_memory=episodic)
        section = builder._build_episodic_section()

        assert "ATTACK" in section
        assert "failed" in section
        assert "(5, 5)" in section


class TestSocialSection:
    """Test social memory section building."""

    def test_no_social_memory(self):
        """Should return empty string if no social memory."""
        builder = ContextBuilder()
        sensation = Sensation(
            tick=1,
            own_needs={},
            own_position=(0, 0),
            own_inventory={},
            visible_tiles=[],
            visible_agents=[],
        )
        section = builder._build_social_section(sensation)
        assert section == ""

    def test_empty_social_memory(self):
        """Should return empty string if no relationships."""
        social = SocialMemory()
        builder = ContextBuilder(social_memory=social)
        sensation = Sensation(
            tick=1,
            own_needs={},
            own_position=(0, 0),
            own_inventory={},
            visible_tiles=[],
            visible_agents=[],
        )
        section = builder._build_social_section(sensation)
        assert section == ""

    def test_allies_and_enemies(self):
        """Should list allies and enemies."""
        social = SocialMemory()
        ally = social.get_or_create("ally_agent")
        ally.trust = 0.9

        enemy = social.get_or_create("enemy_agent")
        enemy.trust = 0.1

        builder = ContextBuilder(social_memory=social)
        sensation = Sensation(
            tick=1,
            own_needs={},
            own_position=(0, 0),
            own_inventory={},
            visible_tiles=[],
            visible_agents=[],
        )

        section = builder._build_social_section(sensation)

        assert "Allies" in section
        assert "ally_age" in section  # Truncated (8 chars)
        assert "Enemies" in section
        assert "enemy_ag" in section  # Truncated (8 chars)

    def test_visible_agent_relationships(self):
        """Should detail relationships with visible agents."""
        social = SocialMemory()
        rel = social.get_or_create("visible_agent")
        rel.trust = 0.5

        builder = ContextBuilder(social_memory=social)
        agents = [
            AgentSummary(
                agent_id="visible_agent",
                position=(1, 1),
                apparent_health="healthy",
                is_carrying_items=False,
            )
        ]
        sensation = Sensation(
            tick=1,
            own_needs={},
            own_position=(0, 0),
            own_inventory={},
            visible_tiles=[],
            visible_agents=agents,
        )

        section = builder._build_social_section(sensation)

        assert "visible_" in section
        assert "neutral" in section or "0.5" in section


class TestMessagesSection:
    """Test messages section building."""

    def test_no_messages(self):
        """Should return empty string if no messages."""
        builder = ContextBuilder()
        sensation = Sensation(
            tick=1,
            own_needs={},
            own_position=(0, 0),
            own_inventory={},
            visible_tiles=[],
            visible_agents=[],
            incoming_messages=[],
        )

        section = builder._build_messages_section(sensation)
        assert section == ""

    def test_message_summary(self):
        """Should list recent messages."""
        builder = ContextBuilder()
        messages = [
            MockMessage(sender_id="agent1", content="Hello there!"),
            MockMessage(sender_id="agent2", content="Watch out for danger"),
        ]
        sensation = Sensation(
            tick=1,
            own_needs={},
            own_position=(0, 0),
            own_inventory={},
            visible_tiles=[],
            visible_agents=[],
            incoming_messages=messages,
        )

        section = builder._build_messages_section(sensation)

        assert "agent1" in section
        assert "Hello there" in section
        assert "agent2" in section
        assert "danger" in section


class TestFullContextBuild:
    """Test building complete context from all sections."""

    def test_minimal_context(self):
        """Should build context with minimal data."""
        builder = ContextBuilder()
        sensation = Sensation(
            tick=1,
            own_needs={"hunger": 50, "thirst": 50, "energy": 50},
            own_position=(0, 0),
            own_inventory={},
            visible_tiles=[],
            visible_agents=[],
        )
        reflection = Reflection(
            last_action_succeeded=True,
            need_trends={},
        )

        context = builder.build_full_context(sensation, reflection)

        # Should have basic sections
        assert "Your Status" in context or "Status" in context
        assert "50" in context  # Needs
        assert "succeeded" in context

    def test_full_context_with_memory(self):
        """Should build context including memory sections."""
        episodic = EpisodicMemory()
        episodic.record_episode(
            tick=5,
            action_type="GATHER",
            target="berries",
            success=True,
            location=(1, 1),
        )

        social = SocialMemory()
        social.get_or_create("friend").trust = 0.8

        builder = ContextBuilder(episodic_memory=episodic, social_memory=social)

        sensation = Sensation(
            tick=10,
            own_needs={"hunger": 50, "thirst": 50, "energy": 50},
            own_position=(2, 2),
            own_inventory={"berries": 2},
            visible_tiles=[TileSummary(x=2, y=2, tile_type="grass", resources=[])],
            visible_agents=[
                AgentSummary(
                    agent_id="friend",
                    position=(3, 3),
                    apparent_health="healthy",
                    is_carrying_items=False,
                )
            ],
        )
        reflection = Reflection(
            last_action_succeeded=True,
            need_trends={"hunger": "stable"},
        )

        context = builder.build_full_context(sensation, reflection, include_memory=True)

        # Should have all major sections
        assert "Status" in context
        assert "Environment" in context or "Visible" in context
        assert "Agents" in context
        assert "Evaluation" in context or "Recent" in context
        assert "GATHER" in context  # Episodic
        assert "friend" in context  # Social

    def test_context_without_memory(self):
        """Should build context without memory sections if disabled."""
        episodic = EpisodicMemory()
        episodic.record_episode(
            tick=5,
            action_type="GATHER",
            target="berries",
            success=True,
            location=(1, 1),
        )

        builder = ContextBuilder(episodic_memory=episodic)

        sensation = Sensation(
            tick=10,
            own_needs={"hunger": 50, "thirst": 50, "energy": 50},
            own_position=(2, 2),
            own_inventory={},
            visible_tiles=[],
            visible_agents=[],
        )
        reflection = Reflection(
            last_action_succeeded=True,
            need_trends={},
        )

        context = builder.build_full_context(sensation, reflection, include_memory=False)

        # Should NOT have memory sections
        assert "GATHER" not in context
        assert "Recent Memory" not in context

    def test_context_sections_separated(self):
        """Context sections should be separated for readability."""
        builder = ContextBuilder()
        sensation = Sensation(
            tick=1,
            own_needs={"hunger": 50, "thirst": 50, "energy": 50},
            own_position=(0, 0),
            own_inventory={},
            visible_tiles=[],
            visible_agents=[],
        )
        reflection = Reflection(
            last_action_succeeded=True,
            need_trends={},
        )

        context = builder.build_full_context(sensation, reflection)

        # Should have section separators (double newlines)
        assert "\n\n" in context
