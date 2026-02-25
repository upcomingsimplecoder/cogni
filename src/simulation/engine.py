"""Simulation engine for the multi-agent survival world.

Manages the tick loop, agent orchestration, conflict resolution,
and integration of awareness loops, communication, and emergence detection.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.agents.evolution import TraitEvolution
from src.agents.registry import AgentRegistry
from src.communication.channel import MessageBus
from src.communication.protocol import MessageType
from src.config import SimulationConfig
from src.emergence.detector import EmergenceDetector
from src.emergence.metrics import MetricsCollector
from src.simulation.actions import Action, ActionResult, ActionType, execute_action
from src.simulation.entities import Agent
from src.simulation.world import World


@dataclass
class AgentTickRecord:
    """Record of one agent's state change in a tick."""

    agent_id: Any  # AgentID
    action: Action | None
    result: ActionResult | None
    needs_before: dict[str, float]
    needs_after: dict[str, float]
    position: tuple[int, int]
    messages_sent: list[Any] = field(default_factory=list)
    messages_received: list[Any] = field(default_factory=list)
    internal_monologue: str = ""


@dataclass
class TickRecord:
    """Record of a single simulation tick — multi-agent version."""

    tick: int
    agent_records: list[AgentTickRecord] = field(default_factory=list)
    emergent_events: list[Any] = field(default_factory=list)

    # Backward compatibility properties
    @property
    def action(self) -> Action | None:
        """Legacy: first agent's action."""
        return self.agent_records[0].action if self.agent_records else None

    @property
    def result(self) -> ActionResult | None:
        """Legacy: first agent's result."""
        return self.agent_records[0].result if self.agent_records else None

    @property
    def agent_needs_before(self) -> dict[str, float]:
        """Legacy: first agent's needs before."""
        return self.agent_records[0].needs_before if self.agent_records else {}

    @property
    def agent_needs_after(self) -> dict[str, float]:
        """Legacy: first agent's needs after."""
        return self.agent_records[0].needs_after if self.agent_records else {}

    @property
    def agent_position(self) -> tuple[int, int]:
        """Legacy: first agent's position."""
        return self.agent_records[0].position if self.agent_records else (0, 0)


@dataclass
class SimulationState:
    """Current state of the simulation."""

    tick: int = 0
    day: int = 0
    time_of_day: str = "dawn"
    history: list[TickRecord] = field(default_factory=list)


class SimulationEngine:
    """Core simulation engine — multi-agent version.

    Supports both single-agent (backward compat) and multi-agent modes.
    """

    def __init__(self, config: SimulationConfig | None = None):
        self.config = config or SimulationConfig()
        self.world = World(self.config.world_width, self.config.world_height, self.config.seed)
        self.registry = AgentRegistry(self)
        self.message_bus = MessageBus()
        self.metrics_collector = MetricsCollector()
        self.emergence_detector = EmergenceDetector(
            cluster_distance=self.config.cluster_distance_threshold,
            cluster_sustain=self.config.cluster_sustain_ticks,
            territory_radius=self.config.territory_radius,
            territory_sustain=self.config.territory_sustain_ticks,
        )
        self.trait_evolution = TraitEvolution(learning_rate=self.config.trait_learning_rate)
        self.state = SimulationState()

        # Action history buffer for ToM observations
        self._last_tick_actions: dict[Any, tuple[str, bool, Any]] = {}

        # Coalitions (lazy init if enabled)
        self.coalition_manager = None
        if self.config.coalitions_enabled:
            from src.social.coalition import CoalitionManager

            self.coalition_manager = CoalitionManager()

        # Evolution (lazy init if enabled)
        self.population_manager = None
        if self.config.evolution_enabled:
            from src.evolution.genetics import GeneticSystem
            from src.evolution.lineage import LineageTracker
            from src.evolution.population import PopulationManager
            from src.evolution.reproduction import ReproductionSystem

            genetics = GeneticSystem(
                mutation_rate=self.config.mutation_rate,
                mutation_magnitude=self.config.mutation_magnitude,
            )
            reproduction = ReproductionSystem(
                min_age=self.config.min_reproduction_age,
                fitness_threshold=self.config.reproduction_fitness_threshold,
                max_population=self.config.max_population,
            )
            lineage_tracker = LineageTracker()
            self.population_manager = PopulationManager(genetics, reproduction, lineage_tracker)

        # Expose lineage tracker directly for serializer access
        self.lineage_tracker = None
        if self.population_manager and hasattr(self.population_manager, "lineage_tracker"):
            self.lineage_tracker = self.population_manager.lineage_tracker

        # Cultural transmission (lazy init if enabled)
        self.cultural_engine = None
        self.cultural_analyzer = None
        if self.config.cultural_transmission_enabled:
            from src.evolution.cultural_metrics import CulturalEvolutionAnalyzer
            from src.evolution.cultural_transmission import CulturalTransmissionEngine

            self.cultural_engine = CulturalTransmissionEngine(
                adoption_threshold=self.config.cultural_adoption_threshold,
                unadoption_threshold=self.config.cultural_unadoption_threshold,
                observation_range=self.config.cultural_observation_range,
                adoption_cooldown=self.config.cultural_adoption_cooldown,
            )
            self.cultural_analyzer = CulturalEvolutionAnalyzer()

        # Metacognition (lazy init if enabled)
        self.metacognition_engine = None
        self.metacognition_analyzer = None
        if self.config.metacognition_enabled:
            from src.metacognition.engine import MetacognitiveEngine

            self.metacognition_engine = MetacognitiveEngine(
                switch_threshold=self.config.metacognition_switch_threshold,
                switch_patience=self.config.metacognition_switch_patience,
                help_confidence_threshold=self.config.metacognition_help_confidence_threshold,
                help_stakes_threshold=self.config.metacognition_help_stakes_threshold,
                deliberation_adjustment_rate=self.config.metacognition_deliberation_adjustment_rate,
                switch_cooldown=self.config.metacognition_switch_cooldown,
                fok_enabled=self.config.metacognition_fok_enabled,
            )

        # Emergent Language (lazy init if enabled)
        self.language_engine = None
        if self.config.language_enabled:
            from src.communication.language_engine import LanguageEngine

            self.language_engine = LanguageEngine(
                innovation_rate=self.config.language_innovation_rate,
                bandwidth_limit=self.config.language_bandwidth_limit,
                symbol_energy_cost=self.config.language_symbol_energy_cost,
                grounding_reinforcement=self.config.language_grounding_reinforcement,
                grounding_weakening=self.config.language_grounding_weakening,
                decay_rate=self.config.language_decay_rate,
                convention_threshold=self.config.language_convention_threshold,
                convention_min_adopters=self.config.language_convention_min_adopters,
                communication_range=self.config.language_communication_range,
            )

        # Linguistic evolution analyzer (tracks language metrics over time)
        self.language_analyzer = None
        if self.config.language_enabled:
            from src.analysis.linguistic import LinguisticEvolutionAnalyzer

            self.language_analyzer = LinguisticEvolutionAnalyzer()

        # Effectiveness scoring (Track 2, lazy init if enabled)
        self.effectiveness_engine = None
        if self.config.effectiveness_scoring_enabled:
            from src.effectiveness.scoring import EffectivenessEngine

            self.effectiveness_engine = EffectivenessEngine(
                scoring_interval=self.config.effectiveness_scoring_interval,
                lookback_window=self.config.effectiveness_lookback_window,
            )

        # Backward compat: create a single agent at center for legacy mode
        self._legacy_agent: Agent | None = None

        # Performance monitoring (lazy init)
        self._perf_monitor = None

    @property
    def agent(self) -> Agent | None:
        """Backward compatibility: return first living agent or legacy agent."""
        agents = self.registry.living_agents()
        if agents:
            return agents[0]
        return self._legacy_agent

    @property
    def agents(self) -> list[Agent]:
        """All living agents."""
        return self.registry.living_agents()

    @property
    def perf_monitor(self):
        """Get or create performance monitor (lazy init)."""
        if self._perf_monitor is None:
            from src.metrics.timing import PerformanceMonitor

            self._perf_monitor = PerformanceMonitor()
        return self._perf_monitor

    def get_agents(self) -> list[Agent]:
        """Get all living agents (for population manager compatibility)."""
        return self.registry.living_agents()

    def setup_legacy_mode(self) -> None:
        """Set up single-agent mode for backward compatibility.

        Creates a single agent at the world center without awareness loop.
        """
        from src.agents.identity import AgentID, AgentProfile, PersonalityTraits

        profile = AgentProfile(
            agent_id=AgentID(),
            name="agent",
            archetype="survivalist",
            traits=PersonalityTraits(),
        )
        self._legacy_agent = self.registry.spawn(
            profile,
            self.config.world_width // 2,
            self.config.world_height // 2,
        )

    def setup_multi_agent(self) -> None:
        """Set up multi-agent mode: spawn agents from configured archetypes.

        Agents are spawned in a cluster (within spawn_radius of a center point)
        to ensure they can see and interact with each other.
        """
        import random as _random

        from src.agents.archetypes import find_valid_spawn, spawn_archetype

        # Build strategy override if LLM mode enabled
        strategy_override = None
        if self.config.llm_strategy_enabled:
            from src.cognition.strategies.llm import LLMStrategy

            strategy_override = LLMStrategy(
                base_url=self.config.llm_base_url + "/v1",
                model=self.config.llm_model,
                call_interval=self.config.llm_call_interval,
                cheap_model=self.config.llm_cheap_model,
                prompt_version=self.config.llm_prompt_version,
                api_key=self.config.llm_api_key,
            )

        archetypes = self.config.agent_archetypes

        # Pick a spawn center, then scatter agents within spawn_radius
        rng = _random.Random(self.config.seed)
        spawn_radius = 10
        cx, cy = find_valid_spawn(self.world, rng)

        for i in range(self.config.num_agents):
            archetype_name = archetypes[i % len(archetypes)]
            agent_name = f"{archetype_name}_{i}"

            # Find a valid position near the spawn center
            from src.simulation.world import TileType

            for _ in range(200):
                sx = cx + rng.randint(-spawn_radius, spawn_radius)
                sy = cy + rng.randint(-spawn_radius, spawn_radius)
                tile = self.world.get_tile(sx, sy)
                if tile and tile.type != TileType.WATER:
                    break
            else:
                sx, sy = find_valid_spawn(self.world, rng)

            spawn_archetype(
                archetype_name=archetype_name,
                agent_name=agent_name,
                engine=self,
                x=sx,
                y=sy,
                strategy_override=strategy_override,
                architecture=self.config.default_architecture,
            )

        # Wire Theory of Mind if enabled
        if self.config.theory_of_mind_enabled:
            from src.cognition.strategies.tom_strategy import TheoryOfMindStrategy

            for agent in self.registry.living_agents():
                if agent.profile:
                    loop = self.registry.get_awareness_loop(agent.agent_id)
                    if loop:
                        # Wrap existing strategy with ToM (preserves personality logic)
                        loop.strategy = TheoryOfMindStrategy(
                            agent_id=str(agent.agent_id),
                            inner_strategy=loop.strategy,
                        )

        # Register agents with cultural transmission system if enabled
        if self.cultural_engine:
            for agent in self.registry.living_agents():
                if agent.profile:
                    self.cultural_engine.register_agent(
                        str(agent.agent_id),
                        agent.profile.traits.as_dict(),
                    )

            # Wrap each agent's strategy with CulturallyModulatedStrategy
            from src.cognition.strategies.cultural_strategy import (
                CulturallyModulatedStrategy,
            )

            for agent in self.registry.living_agents():
                loop = self.registry.get_awareness_loop(agent.agent_id)
                rep = self.cultural_engine.get_repertoire(str(agent.agent_id))
                if loop and rep:
                    loop.strategy = CulturallyModulatedStrategy(
                        inner_strategy=loop.strategy,
                        repertoire=rep,
                        override_probability=self.config.cultural_override_probability,
                    )

        # Register agents with metacognition system and wrap strategies
        if self.metacognition_engine:
            from src.awareness.deliberation import ThresholdDeliberation
            from src.cognition.strategies.metacognitive_strategy import (
                MetacognitiveStrategy,
            )
            from src.cognition.strategies.personality import PersonalityStrategy
            from src.cognition.strategies.planning import PlanningStrategy

            for agent in self.registry.living_agents():
                loop = self.registry.get_awareness_loop(agent.agent_id)
                if not loop:
                    continue

                agent_id_str = str(agent.agent_id)

                # Build strategy instances for potential switching
                strategy_instances = {
                    "personality": PersonalityStrategy(),
                    "planning": PlanningStrategy(),
                }

                # Get current strategy name from loop
                current_name = "personality"
                if hasattr(loop.strategy, "_active_strategy_name"):
                    current_name = loop.strategy._active_strategy_name

                # Get deliberation threshold if available
                initial_threshold = 0.7
                if loop._deliberation and hasattr(loop._deliberation, "threat_threshold"):
                    initial_threshold = loop._deliberation.threat_threshold

                # Clone ThresholdDeliberation per agent for safe mutation
                if loop._deliberation and isinstance(loop._deliberation, ThresholdDeliberation):
                    loop._deliberation = ThresholdDeliberation(
                        threat_threshold=loop._deliberation.threat_threshold,
                    )

                # Register with metacognition engine
                self.metacognition_engine.register_agent(
                    agent_id=agent_id_str,
                    initial_strategy_name=current_name,
                    available_strategies=["personality", "planning"],
                    strategy_instances=strategy_instances,
                    initial_deliberation_threshold=initial_threshold,
                )

                # Wrap strategy with MetacognitiveStrategy (outermost)
                loop.strategy = MetacognitiveStrategy(
                    inner_strategy=loop.strategy,
                    strategy_name=current_name,
                    fok_enabled=self.config.metacognition_fok_enabled,
                )

        # Register agents with language system if enabled
        if self.language_engine:
            for agent in self.registry.living_agents():
                self.language_engine.register_agent(str(agent.agent_id))

        # Record initial agents as lineage roots
        if self.lineage_tracker:
            for agent in self.registry.living_agents():
                if agent.profile:
                    self.lineage_tracker.record_birth(
                        agent_id=str(agent.agent_id),
                        parent_id=None,  # root nodes
                        traits=agent.profile.traits,
                        birth_tick=0,
                    )

    def get_time_of_day(self) -> str:
        """Compute current time of day based on tick count."""
        hour = self.state.tick % self.config.ticks_per_day
        if 5 <= hour < 8:
            return "dawn"
        elif 8 <= hour < 17:
            return "day"
        elif 17 <= hour < 20:
            return "dusk"
        else:
            return "night"

    def step_all(self) -> TickRecord:
        """Execute one tick for ALL agents simultaneously.

        Process:
        1. Each agent's awareness loop produces an Expression (action + message)
        2. Resolve conflicts (two agents gathering same resource)
        3. Execute all actions
        4. Queue and deliver messages
        5. Decay needs for all agents
        6. Check for deaths
        7. Regenerate world resources
        8. Update agent memories
        9. Collect metrics + detect emergence
        10. Build TickRecord
        """
        tick_start = time.perf_counter()

        living = self.registry.living_agents()
        if not living:
            return self._empty_tick_record()

        # --- Phase 1: Decide ---
        t0 = time.perf_counter()
        expressions: list[tuple[Agent, Any]] = []  # (agent, Expression)
        for agent in living:
            loop = self.registry.get_awareness_loop(agent.agent_id)
            if loop:
                expr = loop.tick(self)
                expressions.append((agent, expr))
                # Record classifier predictions for effectiveness scoring
                if self.effectiveness_engine and loop.last_reflection:
                    self.effectiveness_engine.record_classification(
                        tick=self.state.tick,
                        agent_id=str(agent.agent_id),
                        predicted_threat=loop.last_reflection.threat_level,
                        predicted_opportunity=loop.last_reflection.opportunity_score,
                    )
            else:
                # No awareness loop (legacy mode) — use WAIT
                from src.awareness.types import Expression as ExprType

                expressions.append((agent, ExprType(action=Action(type=ActionType.WAIT))))
        t1 = time.perf_counter()

        # --- Phase 2: Record pre-tick state ---
        pre_states: dict[Any, dict[str, float]] = {}
        for agent in living:
            pre_states[agent.agent_id] = {
                "hunger": agent.needs.hunger,
                "thirst": agent.needs.thirst,
                "energy": agent.needs.energy,
                "health": agent.needs.health,
                "_x": agent.x,
                "_y": agent.y,
            }

        # --- Phase 3: Execute actions + queue messages ---
        t2 = time.perf_counter()
        agent_records: list[AgentTickRecord] = []
        # (agent_id, action_type, success, target_id)
        tick_actions: list[tuple[Any, str, bool, Any]] = []
        messages_this_tick: list[Any] = []

        for agent, expr in expressions:
            # Execute the action
            result = execute_action(expr.action, agent, self.world, registry=self.registry)

            # Apply needs delta from action
            if result and result.needs_delta:
                for need, delta in result.needs_delta.items():
                    current = getattr(agent.needs, need)
                    setattr(agent.needs, need, max(0.0, min(100.0, current + delta)))

            # Track action for metrics
            target_id = getattr(expr.action, "target_agent_id", None)
            tick_actions.append(
                (
                    agent.agent_id,
                    expr.action.type.value,
                    result.success if result else False,
                    target_id,
                )
            )

            # Update world occupancy if agent moved successfully
            if expr.action.type == ActionType.MOVE and result and result.success:
                old_pos = (
                    int(pre_states[agent.agent_id]["_x"]),
                    int(pre_states[agent.agent_id]["_y"]),
                )
                new_pos = (agent.x, agent.y)
                self.world.move_agent(agent.agent_id, old_pos, new_pos)

            # Queue message if present
            if expr.message is not None:
                self.message_bus.send(expr.message)
                messages_this_tick.append(expr.message)

            # Build agent tick record (needs_after filled later)
            agent_records.append(
                AgentTickRecord(
                    agent_id=agent.agent_id,
                    action=expr.action,
                    result=result,
                    needs_before=pre_states.get(agent.agent_id, {}),
                    needs_after={},  # Filled after decay
                    position=(agent.x, agent.y),
                    messages_sent=[expr.message] if expr.message else [],
                    internal_monologue=expr.internal_monologue,
                )
            )

            # Record router outcome for effectiveness scoring
            if self.effectiveness_engine:
                loop = self.registry.get_awareness_loop(agent.agent_id)
                arch_name = self.config.default_architecture
                intention_goal = ""
                if loop and loop.last_intention:
                    intention_goal = loop.last_intention.primary_goal
                needs_delta_sum = sum((result.needs_delta or {}).values()) if result else 0.0
                self.effectiveness_engine.record_routing_outcome(
                    tick=self.state.tick,
                    agent_id=str(agent.agent_id),
                    architecture=arch_name,
                    intention_goal=intention_goal,
                    action_succeeded=result.success if result else False,
                    needs_delta_sum=needs_delta_sum,
                )
        t3 = time.perf_counter()

        # Store actions for ToM observations next tick
        self._last_tick_actions = {
            agent_id: (action_type, success, target_id)
            for agent_id, action_type, success, target_id in tick_actions
        }

        # --- Phase 4: Deliver messages ---
        t4 = time.perf_counter()
        delivered = self.message_bus.deliver_all(
            self.registry,
            self.world,
            broadcast_range=self.config.broadcast_range,
            communication_range=self.config.communication_range,
        )
        t5 = time.perf_counter()

        # --- Phase 4.5: Trait evolution from messages ---
        for msg in delivered:
            sender = self.registry.get(getattr(msg, "sender_id", None))
            if sender and sender.profile:
                self.trait_evolution.process_outcome(
                    sender.profile.traits,
                    "sent_message",
                    True,
                    agent_id=str(sender.agent_id),
                    tick=self.state.tick,
                )
            receiver = self.registry.get(getattr(msg, "receiver_id", None))
            if receiver and receiver.profile:
                self.trait_evolution.process_outcome(
                    receiver.profile.traits,
                    "received_message",
                    True,
                    agent_id=str(receiver.agent_id),
                    tick=self.state.tick,
                )

        # --- Phase 5: Decay needs for all living agents ---
        t6 = time.perf_counter()
        for agent in self.registry.living_agents():
            agent.needs.decay(
                self.config.hunger_decay,
                self.config.thirst_decay,
                self.config.energy_decay,
            )
            agent.ticks_alive += 1

        # --- Phase 5.5: Trait evolution for low-health survival ---
        for agent in self.registry.living_agents():
            if agent.needs.health < 30 and agent.profile:
                self.trait_evolution.process_outcome(
                    agent.profile.traits,
                    "survived_low_health",
                    True,
                    agent_id=str(agent.agent_id),
                    tick=self.state.tick,
                )

        # --- Phase 6: Check for deaths ---
        newly_dead: list[Agent] = []
        for agent in list(self.registry.living_agents()):
            if not agent.needs.is_alive():
                cause = (
                    "starvation"
                    if agent.needs.hunger <= 0
                    else "dehydration"
                    if agent.needs.thirst <= 0
                    else "health_depleted"
                )
                self.registry.kill(agent.agent_id, cause)
                newly_dead.append(agent)

        # Record death in lineage tracker
        if self.lineage_tracker:
            for agent in newly_dead:
                fitness = 0.0
                if self.population_manager:
                    fitness = self.population_manager.genetics.fitness(agent)
                self.lineage_tracker.record_death(
                    agent_id=str(agent.agent_id),
                    death_tick=self.state.tick,
                    final_fitness=fitness,
                )

        # --- Phase 7: Fill in post-tick needs ---
        for record in agent_records:
            found_agent = self.registry.get(record.agent_id)
            if found_agent is not None:
                record.needs_after = {
                    "hunger": found_agent.needs.hunger,
                    "thirst": found_agent.needs.thirst,
                    "energy": found_agent.needs.energy,
                    "health": found_agent.needs.health,
                }
            else:
                # Agent died — use last known state
                record.needs_after = {"hunger": 0, "thirst": 0, "energy": 0, "health": 0}
        t7 = time.perf_counter()

        # --- Phase 7.5: Record actual outcomes for classifier accuracy ---
        if self.effectiveness_engine:
            for record in agent_records:
                # Use needs_delta as proxy for resource gain
                inv_gain = (
                    sum(v for v in (record.result.needs_delta or {}).values() if v > 0)
                    if record.result and record.result.needs_delta
                    else 0.0
                )
                self.effectiveness_engine.record_actual_outcome(
                    agent_id=str(record.agent_id),
                    health_before=record.needs_before.get("health", 100),
                    health_after=record.needs_after.get("health", 100),
                    inventory_value_before=0,  # Simplified: we don't have pre-tick inventory
                    inventory_value_after=inv_gain,
                )

        # --- Phase 8: Update agent memories ---
        t8 = time.perf_counter()
        self._update_memories(agent_records, tick_actions, delivered)
        t9 = time.perf_counter()

        # --- Phase 8.5: Process coalitions ---
        if self.coalition_manager:
            self._process_coalitions(agent_records)

        # --- Phase 8.6: Process evolution ---
        if self.population_manager:
            self.population_manager.tick(self, self.state.tick)

        # --- Phase 8.7: Process cultural transmission ---
        if self.cultural_engine:
            self.cultural_engine.tick(self, self.state.tick, agent_records)
            if self.cultural_analyzer:
                self.cultural_analyzer.record_tick(self.cultural_engine, self.state.tick)

            # Record own-use feedback for culturally-influenced actions
            for record in agent_records:
                if not record.action or not record.result:
                    continue
                loop = self.registry.get_awareness_loop(record.agent_id)
                if (
                    loop
                    and loop.last_intention
                    and loop.last_intention.primary_goal.startswith("cultural_")
                ):
                    from src.evolution.observation import ContextTag

                    context = (
                        ContextTag.extract_primary(loop.last_sensation)
                        if loop.last_sensation
                        else ContextTag.ALONE
                    )
                    rep = self.cultural_engine.get_repertoire(str(record.agent_id))
                    if rep:
                        rep.record_own_use(
                            context,
                            record.action.type.value,
                            record.result.success,
                        )

        # --- Phase 8.8: Process metacognition ---
        if self.metacognition_engine:
            self.metacognition_engine.tick(self, self.state.tick, agent_records)
            if self.metacognition_analyzer:
                self.metacognition_analyzer.record_tick(self.metacognition_engine, self.state.tick)

        # --- Phase 8.9: Process emergent language ---
        if self.language_engine:
            self.language_engine.tick(self, self.state.tick, agent_records)
            if self.language_analyzer:
                self.language_analyzer.record_tick(self.language_engine, self.state.tick)

        # --- Phase 9: Regenerate world resources ---
        self.world.tick_resources()

        # --- Phase 10: Detect emergence + collect metrics ---
        t10 = time.perf_counter()
        emergent_events = self.emergence_detector.detect(
            tick=self.state.tick,
            agents=self.registry.living_agents(),
            tick_actions=tick_actions,
            messages=delivered,
        )

        trade_proposals = sum(
            1
            for m in delivered
            if hasattr(m, "message_type") and m.message_type == MessageType.NEGOTIATE
        )

        self.metrics_collector.collect(
            tick=self.state.tick,
            living_agents=self.registry.living_agents(),
            dead_count=self.registry.count_dead,
            tick_actions=[
                (a.value if hasattr(a, "value") else a, s) for _, a, s, _ in tick_actions
            ],
            messages_sent=len(delivered),
            trade_proposals=trade_proposals,
            cluster_count=len([e for e in emergent_events if e.pattern_type == "cluster"]),
        )
        t11 = time.perf_counter()

        # --- Phase 10.5: Effectiveness scoring (periodic) ---
        if self.effectiveness_engine:
            # Forward new nudge records from trait evolution
            from src.agents.evolution import TraitNudgeRecord

            nudge_history: list[TraitNudgeRecord] = self.trait_evolution.history
            for nudge_record in nudge_history:
                # Type guard: ensure record is TraitNudgeRecord
                if nudge_record.viewed_at is None and nudge_record.tick == self.state.tick:
                    self.effectiveness_engine.record_nudge(nudge_record)
            # Run periodic scoring cron
            self.effectiveness_engine.tick(self, self.state.tick)

        # --- Phase 11: Update simulation state ---
        self.state.tick += 1
        self.state.day = self.state.tick // self.config.ticks_per_day
        self.state.time_of_day = self.get_time_of_day()

        tick_record = TickRecord(
            tick=self.state.tick,
            agent_records=agent_records,
            emergent_events=emergent_events,
        )
        self.state.history.append(tick_record)

        # --- Record timing ---
        tick_end = time.perf_counter()
        if self._perf_monitor is not None:
            from src.metrics.timing import TickTiming

            timing = TickTiming(
                decision_ms=(t1 - t0) * 1000,
                action_execution_ms=(t3 - t2) * 1000,
                message_delivery_ms=(t5 - t4) * 1000,
                need_decay_ms=(t7 - t6) * 1000,
                memory_update_ms=(t9 - t8) * 1000,
                emergence_detection_ms=(t11 - t10) * 1000,
                total_ms=(tick_end - tick_start) * 1000,
            )
            self._perf_monitor.record_tick(timing)

        return tick_record

    def step(self, action: Action | None = None) -> TickRecord:
        """BACKWARD COMPATIBLE: single-agent step.

        If called with an action, applies to self.agent (first/legacy agent).
        """
        agent = self.agent
        if agent is None:
            return self._empty_tick_record()

        # Record pre-tick state
        needs_before = {
            "hunger": agent.needs.hunger,
            "thirst": agent.needs.thirst,
            "energy": agent.needs.energy,
            "health": agent.needs.health,
        }

        # Execute action
        result = None
        if action is not None:
            result = execute_action(action, agent, self.world, registry=self.registry)
            if result.needs_delta:
                for need, delta in result.needs_delta.items():
                    current = getattr(agent.needs, need)
                    setattr(agent.needs, need, max(0.0, min(100.0, current + delta)))

        # Decay needs
        agent.needs.decay(
            self.config.hunger_decay,
            self.config.thirst_decay,
            self.config.energy_decay,
        )

        # Regenerate world resources
        self.world.tick_resources()

        # Record post-tick state
        needs_after = {
            "hunger": agent.needs.hunger,
            "thirst": agent.needs.thirst,
            "energy": agent.needs.energy,
            "health": agent.needs.health,
        }

        # Update simulation state
        self.state.tick += 1
        self.state.day = self.state.tick // self.config.ticks_per_day
        self.state.time_of_day = self.get_time_of_day()

        agent_record = AgentTickRecord(
            agent_id=agent.agent_id,
            action=action,
            result=result,
            needs_before=needs_before,
            needs_after=needs_after,
            position=(agent.x, agent.y),
        )

        record = TickRecord(
            tick=self.state.tick,
            agent_records=[agent_record],
        )
        self.state.history.append(record)

        return record

    def _update_memories(
        self,
        agent_records: list[AgentTickRecord],
        tick_actions: list[tuple[Any, str, bool, Any]],
        delivered_messages: list[Any],
    ) -> None:
        """Update episodic and social memory for all agents."""
        for record in agent_records:
            memory = self.registry.get_memory(record.agent_id)
            if not memory:
                continue
            episodic, social = memory

            # Record episode
            if record.action and record.result:
                episodic.record_episode(
                    tick=self.state.tick,
                    action_type=record.action.type.value,
                    target=record.action.target,
                    success=record.result.success,
                    needs_delta=record.result.needs_delta or {},
                    location=record.position,
                    involved_agent=getattr(record.action, "target_agent_id", None),
                )

            # Update social memory + trait evolution for GIVE and ATTACK actions
            agent = self.registry.get(record.agent_id)
            if record.action:
                target_id = getattr(record.action, "target_agent_id", None)
                if target_id and record.result and record.result.success:
                    if record.action.type == ActionType.GIVE:
                        rel = social.get_or_create(target_id)
                        rel.update_trust("gave_resource", True, self.state.tick)
                        rel.update_resource_flow(record.action.quantity)
                        # Evolve giver's traits
                        if agent and agent.profile:
                            self.trait_evolution.process_outcome(
                                agent.profile.traits,
                                "shared_resource",
                                True,
                                agent_id=str(agent.agent_id),
                                tick=self.state.tick,
                            )
                        # Also update receiver's social memory + traits
                        target_mem = self.registry.get_memory(target_id)
                        if target_mem:
                            _, target_social = target_mem
                            target_rel = target_social.get_or_create(record.agent_id)
                            target_rel.update_trust("received_help", True, self.state.tick)
                            target_rel.update_resource_flow(-record.action.quantity)
                        target_agent = self.registry.get(target_id)
                        if target_agent and target_agent.profile:
                            self.trait_evolution.process_outcome(
                                target_agent.profile.traits,
                                "received_help",
                                True,
                                agent_id=str(target_agent.agent_id),
                                tick=self.state.tick,
                            )

                    elif record.action.type == ActionType.ATTACK:
                        rel = social.get_or_create(target_id)
                        rel.update_trust("attacked", False, self.state.tick)
                        # Evolve attacker's traits
                        if agent and agent.profile:
                            self.trait_evolution.process_outcome(
                                agent.profile.traits,
                                "attacked_agent",
                                True,
                                agent_id=str(agent.agent_id),
                                tick=self.state.tick,
                            )
                        # Target's perspective
                        target_mem = self.registry.get_memory(target_id)
                        if target_mem:
                            _, target_social = target_mem
                            target_rel = target_social.get_or_create(record.agent_id)
                            target_rel.update_trust("attacked", False, self.state.tick)
                            target_rel.was_attacked_by = True
                        target_agent = self.registry.get(target_id)
                        if target_agent and target_agent.profile:
                            self.trait_evolution.process_outcome(
                                target_agent.profile.traits,
                                "was_attacked",
                                False,
                                agent_id=str(target_agent.agent_id),
                                tick=self.state.tick,
                            )

                # Trait evolution for non-social actions
                elif record.result and record.result.success and agent and agent.profile:
                    if record.action.type == ActionType.GATHER:
                        self.trait_evolution.process_outcome(
                            agent.profile.traits,
                            "found_resource",
                            True,
                            agent_id=str(agent.agent_id),
                            tick=self.state.tick,
                        )
                    elif record.action.type == ActionType.MOVE:
                        self.trait_evolution.process_outcome(
                            agent.profile.traits,
                            "explored_new_area",
                            True,
                            agent_id=str(agent.agent_id),
                            tick=self.state.tick,
                        )

    def _process_coalitions(self, agent_records: list[AgentTickRecord]) -> None:
        """Process coalition dynamics: propose, accept/reject, maintain, dissolve.

        Each tick:
        1. Evaluate pending proposals — targets accept/reject based on traits + trust
        2. Propose new coalitions — eligible agents invite trusted visible neighbors
        3. Maintain existing — update cohesion, dissolve if too low
        """
        if self.coalition_manager is None:
            return

        from src.social.coordination import CoalitionCoordinator
        from src.social.formation import CoalitionFormation

        formation = CoalitionFormation(sociability_threshold=0.5)
        coordinator = CoalitionCoordinator()

        # --- 1. Evaluate pending proposals ---
        for coalition_id, proposal in list(self.coalition_manager.pending_proposals().items()):
            for target_id in list(proposal["target_ids"] - proposal["accepted"]):
                # Find the agent — target_id is a string, need to match
                target_agent = None
                for a in self.registry.living_agents():
                    if str(a.agent_id) == target_id:
                        target_agent = a
                        break
                if not target_agent or not target_agent.alive:
                    self.coalition_manager.reject(coalition_id, target_id)
                    continue
                memory = self.registry.get_memory(target_agent.agent_id)
                if not memory:
                    self.coalition_manager.reject(coalition_id, target_id)
                    continue
                _, social = memory
                traits = target_agent.profile.traits if target_agent.profile else None
                if traits and formation.should_accept(
                    traits,
                    proposal["proposer_id"],
                    social,
                    proposal["goal"],
                    {
                        "hunger": target_agent.needs.hunger,
                        "thirst": target_agent.needs.thirst,
                        "energy": target_agent.needs.energy,
                        "health": target_agent.needs.health,
                    },
                ):
                    self.coalition_manager.accept(coalition_id, target_id)
                else:
                    self.coalition_manager.reject(coalition_id, target_id)

        # --- 2. Propose new coalitions ---
        for agent in self.registry.living_agents():
            if not agent.profile:
                continue
            # Skip if already in a coalition
            if self.coalition_manager.get_coalition(str(agent.agent_id)):
                continue

            memory = self.registry.get_memory(agent.agent_id)
            if not memory:
                continue
            _, social = memory

            loop = self.registry.get_awareness_loop(agent.agent_id)
            visible = loop.last_sensation.visible_agents if loop and loop.last_sensation else []
            needs = {
                "hunger": agent.needs.hunger,
                "thirst": agent.needs.thirst,
                "energy": agent.needs.energy,
                "health": agent.needs.health,
            }

            if formation.should_propose(agent.profile.traits, visible, social, needs):
                targets = formation.select_coalition_targets(str(agent.agent_id), visible, social)
                if targets:
                    goal = formation.suggest_coalition_goal(agent.profile.traits, needs, visible)
                    self.coalition_manager.propose(str(agent.agent_id), targets, goal)

        # --- 3. Maintain existing coalitions ---
        for coalition in self.coalition_manager.all_coalitions():
            member_positions = {}
            for member_id in coalition.members:
                # member_id is a string, need to find matching agent
                for a in self.registry.living_agents():
                    if str(a.agent_id) == member_id:
                        member_positions[member_id] = (a.x, a.y)
                        break

            coalition.cohesion = coordinator.calculate_cohesion(coalition, member_positions)

            # Dissolve if cohesion too low
            if coalition.cohesion < 0.2:
                self.coalition_manager.dissolve(coalition.id, reason="low_cohesion")

    def _empty_tick_record(self) -> TickRecord:
        """Create an empty tick record."""
        self.state.tick += 1
        record = TickRecord(tick=self.state.tick)
        self.state.history.append(record)
        return record

    def is_over(self) -> bool:
        """Check if the simulation has ended."""
        return self.registry.count_living == 0 or self.state.tick >= self.config.max_ticks
