"""Shared test doubles for AUTOCOG test suites.

Provides mock dataclasses used across multiple test files (Phase 2, Phase 3, etc.)
to avoid duplication. These are dataclass-based test doubles, not unittest.mock.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.awareness.types import Expression, Intention
from src.simulation.actions import Action, ActionType

# ============================================================================
# Mock Types for Sensation / Agent / Engine
# ============================================================================


@dataclass
class MockTileSummary:
    """Mock tile for sensation creation."""

    x: int = 0
    y: int = 0
    tile_type: str = "grass"
    resources: list = field(default_factory=list)
    occupants: list = field(default_factory=list)


@dataclass
class MockAgentSummary:
    """Mock agent for sensation visible_agents."""

    agent_id: str = "other"
    position: tuple = (1, 1)
    apparent_health: str = "healthy"
    is_carrying_items: bool = False


@dataclass
class MockSensation:
    """Mock sensation for testing."""

    tick: int = 0
    own_needs: dict = field(
        default_factory=lambda: {
            "hunger": 80,
            "thirst": 80,
            "energy": 80,
            "health": 80,
        }
    )
    own_position: tuple = (5, 5)
    own_inventory: dict = field(default_factory=dict)
    visible_tiles: list = field(default_factory=list)
    visible_agents: list = field(default_factory=list)
    incoming_messages: list = field(default_factory=list)
    time_of_day: str = "day"
    own_traits: dict = field(default_factory=dict)


@dataclass
class MockReflection:
    """Mock reflection for testing."""

    last_action_succeeded: bool = True
    need_trends: dict = field(
        default_factory=lambda: {
            "hunger": "stable",
            "thirst": "stable",
            "energy": "stable",
            "health": "stable",
        }
    )
    recent_interaction_outcomes: list = field(default_factory=list)
    threat_level: float = 0.0
    opportunity_score: float = 0.0


@dataclass
class MockAgent:
    """Mock agent for engine/cultural/metacognition tests."""

    agent_id: str
    x: int
    y: int
    alive: bool = True
    profile: MockProfile = None

    def __post_init__(self):
        if self.profile is None:
            self.profile = MockProfile()


@dataclass
class MockProfile:
    """Mock profile for agent personality."""

    traits: dict = field(
        default_factory=lambda: {
            "sociability": 0.5,
            "cooperation_tendency": 0.5,
            "curiosity": 0.5,
            "risk_tolerance": 0.5,
        }
    )

    def as_dict(self):
        return self.traits


@dataclass
class MockAwarenessLoop:
    """Mock awareness loop."""

    last_sensation: MockSensation = None
    last_reflection: MockReflection = None
    last_intention: Intention = None
    strategy: object = None
    _deliberation: object = None


@dataclass
class MockActionResult:
    """Mock action result."""

    success: bool = True
    needs_delta: dict = field(default_factory=lambda: {"hunger": 5.0, "thirst": 5.0})


@dataclass
class MockAgentTickRecord:
    """Mock tick record for cultural/metacognition tests."""

    agent_id: str
    position: tuple
    action: Action = None
    result: MockActionResult = None
    needs_before: dict = field(
        default_factory=lambda: {
            "hunger": 80.0,
            "thirst": 80.0,
            "energy": 80.0,
            "health": 80.0,
        }
    )
    needs_after: dict = field(
        default_factory=lambda: {
            "hunger": 75.0,
            "thirst": 75.0,
            "energy": 75.0,
            "health": 80.0,
        }
    )
    internal_monologue: str = ""


@dataclass
class MockRegistry:
    """Mock registry for engine tests."""

    agents: list = field(default_factory=list)
    awareness_loops: dict = field(default_factory=dict)

    def living_agents(self):
        return [a for a in self.agents if a.alive]

    def get(self, agent_id):
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        return None

    def get_awareness_loop(self, agent_id):
        return self.awareness_loops.get(agent_id)


@dataclass
class MockEngine:
    """Mock simulation engine."""

    registry: MockRegistry = field(default_factory=MockRegistry)


class MockStrategy:
    """Mock inner strategy for wrapper tests."""

    def form_intention(self, sensation, reflection):
        return Intention(primary_goal="mock_goal", confidence=0.5)

    def express(self, sensation, reflection, intention):
        return Expression(action=Action(type=ActionType.WAIT))
