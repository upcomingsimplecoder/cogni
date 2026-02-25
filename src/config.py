"""Configuration settings for the cogniarch simulation.

Uses Pydantic Settings for validation and environment variable support.
All settings can be overridden via AUTOCOG_* environment variables.
"""

from pydantic import Field
from pydantic_settings import BaseSettings


class SimulationConfig(BaseSettings):
    """Global configuration for the survival simulation."""

    # World
    world_width: int = 32
    world_height: int = 32
    seed: int = 42

    # Agent needs decay per tick
    hunger_decay: float = 0.35  # per tick
    thirst_decay: float = 0.4  # per tick (thirst drains faster)
    energy_decay: float = 0.3  # per tick

    # Resource regen
    resource_regen_rate: float = 0.1  # per tick, probability of +1

    # Tick
    ticks_per_day: int = 24
    max_ticks: int = 5000

    # Agent vision
    vision_radius: int = 5

    # Multi-agent
    num_agents: int = 5
    agent_archetypes: list[str] = Field(
        default=["gatherer", "explorer", "diplomat", "aggressor", "survivalist"]
    )

    # Cognitive architecture
    default_architecture: str = "reactive"

    # Communication
    communication_range: int = 10
    broadcast_range: int = 6
    message_energy_cost: float = 0.5
    max_messages_per_tick: int = 2

    # Combat
    attack_damage: float = 15.0
    attack_self_damage: float = 2.0
    attack_energy_cost: float = 5.0

    # Trait evolution
    trait_learning_rate: float = 0.01

    # Emergence detection
    cluster_distance_threshold: int = 3
    cluster_sustain_ticks: int = 5
    territory_radius: int = 5
    territory_sustain_ticks: int = 20

    # LLM strategy
    llm_strategy_enabled: bool = False
    llm_call_interval: int = 5  # call LLM every N ticks

    # LLM models (any OpenAI-compatible endpoint)
    llm_base_url: str = ""  # set via --llm-url or AUTOCOG_LLM_BASE_URL
    llm_api_key: str = ""  # set via --llm-key or AUTOCOG_LLM_API_KEY
    llm_model: str = "opus"  # primary model name (provider-dependent)
    llm_cheap_model: str = "sonnet"  # fast model name (provider-dependent)
    llm_prompt_version: str = "v2_rich"  # "v1_minimal" | "v2_rich"

    # Trajectory recording
    trajectory_recording: bool = False
    trajectory_output_dir: str = "data/trajectories"

    # Checkpointing
    checkpoint_interval: int = 0  # 0 = disabled
    checkpoint_dir: str = "data/checkpoints"
    checkpoint_max: int = 10

    # Theory of Mind
    theory_of_mind_enabled: bool = False

    # Evolution
    evolution_enabled: bool = False
    mutation_rate: float = 0.05
    mutation_magnitude: float = 0.1
    max_population: int = 15
    min_reproduction_age: int = 100
    reproduction_fitness_threshold: float = 0.6
    cultural_transmission_enabled: bool = False
    cultural_observation_range: int = 5  # manhattan distance for behavioral observation
    cultural_adoption_threshold: float = 0.3  # min combined bias score to adopt variant
    cultural_unadoption_threshold: float = 0.15  # unadopt if own success rate below this
    cultural_adoption_cooldown: int = 10  # min ticks between adoption decisions per agent
    cultural_override_probability: float = 0.7  # P(cultural variant overrides personality)
    cultural_snapshot_interval: int = 25  # record cultural state every N ticks

    # Metacognition (Phase 3)
    metacognition_enabled: bool = False
    metacognition_switch_threshold: float = 0.3
    metacognition_switch_patience: int = 5
    metacognition_help_confidence_threshold: float = 0.3
    metacognition_help_stakes_threshold: float = 0.6
    metacognition_deliberation_adjustment_rate: float = 0.05
    metacognition_switch_cooldown: int = 10
    metacognition_fok_enabled: bool = True

    # Emergent Language (Phase 4)
    language_enabled: bool = False
    language_innovation_rate: float = 0.05  # P(agent creates new symbol per tick)
    language_bandwidth_limit: int = 3  # max symbols per message (drives compression)
    language_symbol_energy_cost: float = 0.3  # energy per symbol sent
    language_grounding_reinforcement: float = 0.15  # strength gain on success
    language_grounding_weakening: float = 0.08  # strength loss on failure
    language_decay_rate: float = 0.003  # per-tick decay of unused associations
    language_convention_threshold: float = 0.5  # min strength for convention
    language_convention_min_adopters: int = 3  # min agents to be "established"
    language_communication_range: int = 8  # manhattan distance for symbol messages

    # Coalitions
    coalitions_enabled: bool = False

    # Effectiveness scoring (Track 2)
    effectiveness_scoring_enabled: bool = False
    effectiveness_scoring_interval: int = 50  # ticks between scoring runs
    effectiveness_lookback_window: int = 25  # ticks to look back for outcome measurement
    effectiveness_nudge_history_max: int = 500  # max nudge records to retain
    router_quality_weight: float = 0.0  # weight for score-weighted routing (0 = disabled)

    model_config = {"env_prefix": "AUTOCOG_"}
