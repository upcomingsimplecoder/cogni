"""Trajectory data schemas.

Defines the structure of recorded agent trajectories, including:
- AgentSnapshot: Complete agent state at a single tick
- EmergenceSnapshot: Emergence events at a single tick
- RunMetadata: Metadata for an entire simulation run
- TrajectoryDataset: Complete dataset from one run
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class AgentSnapshot:
    """Complete agent state at a single tick."""

    tick: int
    agent_id: str
    agent_name: str
    archetype: str
    position: tuple[int, int]
    alive: bool

    # Needs (raw values)
    hunger: float
    thirst: float
    energy: float
    health: float

    # Personality (current, may have evolved)
    traits: dict[str, float]

    # SRIE pipeline outputs
    sensation_summary: dict  # Compressed: visible_agent_count, resource_count, message_count
    reflection: dict  # threat_level, opportunity_score, need_trends, last_action_succeeded
    intention: dict  # primary_goal, confidence, target_position, target_agent_id

    # Action taken
    action_type: str
    action_target: str | None
    action_target_agent: str | None
    action_succeeded: bool

    # Needs delta from action
    needs_delta: dict[str, float]

    # Inventory snapshot
    inventory: dict[str, int]

    # Messages sent/received this tick
    messages_sent: list[dict] = field(default_factory=list)  # [{type, receiver, content}]
    messages_received: list[dict] = field(default_factory=list)  # [{type, sender, content}]

    # Internal monologue
    internal_monologue: str = ""

    # Trait evolution this tick (if any)
    trait_changes: list[dict] = field(
        default_factory=list
    )  # [{trait, old_value, new_value, trigger_event}]

    # Cultural transmission (Phase 2)
    cultural_repertoire: dict = field(
        default_factory=dict
    )  # {variant_id: {context, action, success_rate, adopted, ...}}
    cultural_learning_style: str = (
        ""  # "prestige" | "conformist" | "content" | "anti_conformist" | "balanced"
    )
    transmission_events_this_tick: list[dict] = field(
        default_factory=list
    )  # [{variant_id, bias_type, adopted, probability}]
    cultural_group_id: int = -1  # cultural group index (-1 = none)

    # Theory of Mind (ToM)
    tom_model_count: int = 0  # number of agents modeled
    tom_models: dict = field(default_factory=dict)
    # Format: {agent_id: {estimated_disposition, prediction_accuracy, ticks_observed,
    #                     times_helped_me, times_attacked_me, ...}}

    # Coalition membership
    coalition_id: str | None = None  # coalition ID if in one
    coalition_role: str = ""  # "leader" | "member" | ""
    coalition_goal: str = ""  # coalition's shared goal

    # Social relationships (Addition 1)
    social_relationships: dict = field(default_factory=dict)
    # Format: {agent_id: {trust, interaction_count, net_resources_given, was_attacked_by,
    #                     was_helped_by, last_interaction_tick}}

    # Metacognition enrichment (Addition 4 & 5)
    metacog_calibration_curve: list[dict] = field(
        default_factory=list
    )  # [{bin_center, accuracy, count}]
    metacog_deliberation_invoked: bool = False  # whether System 2 was invoked this tick

    # Planning state (Addition 6)
    plan_state: dict = field(default_factory=dict)  # {goal, steps, current_step, status, progress}

    # Language symbols (Addition 7)
    language_symbols: list[dict] = field(
        default_factory=list
    )  # [{form, meaning, success_rate, times_used, strength}]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class EmergenceSnapshot:
    """Emergence events at a single tick."""

    tick: int
    pattern_type: str
    agents_involved: list[str]
    description: str
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class RunMetadata:
    """Metadata for an entire simulation run."""

    run_id: str  # UUID
    timestamp: str  # ISO 8601
    seed: int
    config: dict  # Full SimulationConfig as dict
    num_agents: int
    max_ticks: int
    actual_ticks: int = 0  # How many ticks actually ran
    agents: list[dict] = field(default_factory=list)  # [{id, name, archetype, initial_traits}]
    architecture: str | None = None  # Cognitive architecture name
    final_state: dict = field(
        default_factory=dict
    )  # {agents_alive, agents_dead, emergence_events_count}

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TrajectoryDataset:
    """Complete dataset from one run."""

    metadata: RunMetadata
    agent_snapshots: list[AgentSnapshot] = field(default_factory=list)
    emergence_events: list[EmergenceSnapshot] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "agent_snapshots": [s.to_dict() for s in self.agent_snapshots],
            "emergence_events": [e.to_dict() for e in self.emergence_events],
        }
