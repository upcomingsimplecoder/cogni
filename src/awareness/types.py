"""Data types for the Sensation-Reflection-Intention-Expression awareness loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TileSummary:
    """Compressed tile info for agent perception."""

    x: int
    y: int
    tile_type: str
    resources: list[tuple[str, int]]  # (kind, quantity)
    occupants: list[object] = field(default_factory=list)  # AgentIDs


@dataclass
class AgentSummary:
    """What one agent can observe about another."""

    agent_id: object  # AgentID
    position: tuple[int, int]
    apparent_health: str  # "healthy", "injured", "critical"
    is_carrying_items: bool
    last_action: str = ""  # What they did last tick
    last_action_target: object = None  # Who they targeted


@dataclass
class Sensation:
    """What the agent perceives this tick.

    Assembled by the SensationModule from world state.
    """

    tick: int
    own_needs: dict[str, float]
    own_position: tuple[int, int]
    own_inventory: dict[str, int]
    visible_tiles: list[TileSummary]
    visible_agents: list[AgentSummary]
    incoming_messages: list[Any] = field(default_factory=list)  # Message objects
    time_of_day: str = "day"
    own_traits: dict[str, float] = field(default_factory=dict)


@dataclass
class InteractionOutcome:
    """Result of a past interaction with another agent."""

    other_agent_id: object  # AgentID
    tick: int
    was_positive: bool
    interaction_type: str  # "trade", "shared_info", "attacked", "was_helped"


@dataclass
class Reflection:
    """Agent's evaluation of recent experience."""

    last_action_succeeded: bool
    need_trends: dict[str, str]  # "hunger": "declining" / "stable" / "improving"
    recent_interaction_outcomes: list[InteractionOutcome] = field(default_factory=list)
    threat_level: float = 0.0  # 0-1
    opportunity_score: float = 0.0  # 0-1


@dataclass
class Intention:
    """Agent's current goals and plan. Short-term: 1-5 tick horizon."""

    primary_goal: str  # "satisfy_hunger", "explore", "trade", "flee"
    target_position: tuple[int, int] | None = None
    target_agent_id: object | None = None  # AgentID
    planned_actions: list[str] = field(default_factory=list)
    confidence: float = 0.5  # 0-1


@dataclass
class Expression:
    """Agent's chosen output: action + optional message.

    This is what the awareness loop produces each tick.
    """

    action: Any  # Action object from simulation.actions
    message: Any | None = None  # Message object from communication.protocol
    internal_monologue: str = ""  # For visualization / debugging
