"""Cultural transmission engine: Boyd & Richerson transmission biases in agent societies.

Orchestrates the full cultural transmission cycle each tick:
1. Build behavioral observations from agent actions and outcomes
2. Distribute observations to nearby agents (within observation range)
3. Evaluate adoption decisions using weighted transmission biases
4. Track transmission events for metrics and analysis

This is the deep tech moat — theoretically grounded in Boyd & Richerson (1985).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


@dataclass
class TransmissionEvent:
    """Record of a cultural transmission event.

    Logged whenever an agent evaluates (and potentially adopts) a variant.
    """

    tick: int
    observer_id: str
    actor_id: str  # best-contributing actor, or "" if multiple
    variant_id: str
    bias_type: str  # dominant bias: "prestige", "conformist", etc.
    adoption_probability: float
    adopted: bool


class CulturalTransmissionEngine:
    """Orchestrates Boyd & Richerson cultural transmission.

    Called once per tick (Phase 8.7 in engine.step_all) to:
    1. Build observations from this tick's agent actions
    2. Distribute observations to nearby observers
    3. For each agent, evaluate adoption decisions via weighted biases
    4. Handle unadoption of poorly-performing variants

    This is the publishable moat — the biases are theoretically grounded
    in Boyd & Richerson (1985) Culture and the Evolutionary Process.
    """

    def __init__(
        self,
        adoption_threshold: float = 0.3,
        unadoption_threshold: float = 0.15,
        observation_range: int = 5,
        adoption_cooldown: int = 10,
    ):
        """Initialize cultural transmission system.

        Args:
            adoption_threshold: Minimum combined bias score to consider adopting
            unadoption_threshold: If own success rate drops below this, unadopt
            observation_range: How far agents can observe (manhattan distance)
            adoption_cooldown: Min ticks between adoption decisions per agent
        """
        self.adoption_threshold = adoption_threshold
        self.unadoption_threshold = unadoption_threshold
        self.observation_range = observation_range
        self.adoption_cooldown = adoption_cooldown

        # Per-agent state (keyed by agent_id string)
        self._observation_memories: dict[str, Any] = {}  # ObservationMemory
        self._repertoires: dict[str, Any] = {}  # BehavioralRepertoire
        self._transmission_weights: dict[str, Any] = {}  # TransmissionWeights
        self._last_adoption_tick: dict[str, int] = {}

        # Global event log for metrics
        self._transmission_events: list[TransmissionEvent] = []

    def register_agent(self, agent_id: str, traits: dict[str, float]) -> None:
        """Register a new agent with the cultural transmission system.

        Creates observation memory, behavioral repertoire, and derives
        transmission weights from personality traits.

        Args:
            agent_id: String agent identifier
            traits: Personality traits dict (cooperation_tendency, curiosity, etc.)
        """
        from src.evolution.observation import ObservationMemory
        from src.evolution.repertoire import BehavioralRepertoire
        from src.evolution.transmission_biases import TransmissionWeights

        self._observation_memories[agent_id] = ObservationMemory()
        self._repertoires[agent_id] = BehavioralRepertoire()
        self._transmission_weights[agent_id] = TransmissionWeights.from_personality(traits)
        self._last_adoption_tick[agent_id] = -999

    def unregister_agent(self, agent_id: str) -> None:
        """Mark agent as inactive. Keep data for post-mortem analysis."""
        # Intentionally keep data — useful for cultural lineage analysis
        pass

    def get_repertoire(self, agent_id: str) -> Any | None:
        """Get an agent's behavioral repertoire."""
        return self._repertoires.get(agent_id)

    def get_observation_memory(self, agent_id: str) -> Any | None:
        """Get an agent's observation memory."""
        return self._observation_memories.get(agent_id)

    def get_transmission_weights(self, agent_id: str) -> Any | None:
        """Get an agent's transmission bias weights."""
        return self._transmission_weights.get(agent_id)

    def tick(
        self,
        engine: Any,
        tick: int,
        agent_records: list[Any],
    ) -> list[TransmissionEvent]:
        """Process cultural transmission for this tick.

        Steps:
        1. Generate observations from agent_records (this tick's actions)
        2. Distribute observations to nearby agents
        3. For each agent, evaluate adoption decisions

        Args:
            engine: SimulationEngine instance
            tick: Current tick number
            agent_records: List of AgentTickRecord from this tick

        Returns:
            List of TransmissionEvents that occurred this tick
        """

        events_this_tick: list[TransmissionEvent] = []

        # Step 1: Build observations from tick records
        observations = self._build_observations(engine, tick, agent_records)

        # Step 2: Distribute to nearby observers
        self._distribute_observations(engine, observations)

        # Step 3: Adoption decisions for each living agent
        for agent in engine.registry.living_agents():
            aid = str(agent.agent_id)
            if aid not in self._repertoires:
                continue

            # Cooldown check — don't evaluate every tick
            if tick - self._last_adoption_tick.get(aid, -999) < self.adoption_cooldown:
                continue

            new_events = self._evaluate_adoption(aid, tick)
            events_this_tick.extend(new_events)

            # Unadoption check for poorly-performing adopted variants
            self._evaluate_unadoption(aid)

        self._transmission_events.extend(events_this_tick)
        return events_this_tick

    def _build_observations(
        self,
        engine: Any,
        tick: int,
        agent_records: list[Any],
    ) -> list[Any]:
        """Convert AgentTickRecords into BehaviorObservations.

        For each agent that took an action this tick, create an observation
        template (observer_id left blank — filled per-observer in distribute).

        Args:
            engine: SimulationEngine instance
            tick: Current tick number
            agent_records: AgentTickRecord list

        Returns:
            List of BehaviorObservation templates
        """
        from src.evolution.observation import BehaviorObservation, ContextTag

        observations = []

        for record in agent_records:
            if record.action is None or record.result is None:
                continue

            agent = engine.registry.get(record.agent_id)
            if not agent:
                continue

            # Extract context tag from the agent's last sensation
            loop = engine.registry.get_awareness_loop(record.agent_id)
            context_tag = ContextTag.ALONE
            if loop and loop.last_sensation:
                context_tag = ContextTag.extract_primary(loop.last_sensation)

            # Compute fitness delta (sum of needs changes from action)
            fitness_delta = sum((record.result.needs_delta or {}).values())

            obs = BehaviorObservation(
                observer_id="",  # template — filled per-observer in distribute
                actor_id=str(record.agent_id),
                action_type=record.action.type.value,
                context_tag=context_tag,
                outcome_success=record.result.success,
                outcome_fitness_delta=fitness_delta,
                tick=tick,
                actor_position=record.position,
            )
            observations.append(obs)

        return observations

    def _distribute_observations(
        self,
        engine: Any,
        observations: list[Any],
    ) -> None:
        """Distribute observations to nearby agents within observation range.

        Each living agent that is within manhattan distance of the actor
        receives the observation in their memory and repertoire.

        Args:
            engine: SimulationEngine instance
            observations: BehaviorObservation templates from _build_observations
        """
        from src.evolution.observation import BehaviorObservation

        living = engine.registry.living_agents()

        for obs in observations:
            for agent in living:
                aid = str(agent.agent_id)

                # Don't observe yourself
                if aid == obs.actor_id:
                    continue

                # Manhattan distance range check
                dist = abs(agent.x - obs.actor_position[0]) + abs(agent.y - obs.actor_position[1])
                if dist > self.observation_range:
                    continue

                # Record observation for this observer
                mem = self._observation_memories.get(aid)
                if mem is None:
                    continue

                # Create observer-specific copy with observer_id filled
                agent_obs = BehaviorObservation(
                    observer_id=aid,
                    actor_id=obs.actor_id,
                    action_type=obs.action_type,
                    context_tag=obs.context_tag,
                    outcome_success=obs.outcome_success,
                    outcome_fitness_delta=obs.outcome_fitness_delta,
                    tick=obs.tick,
                    actor_position=obs.actor_position,
                )
                mem.record(agent_obs)

                # Also update repertoire evidence
                rep = self._repertoires.get(aid)
                if rep is not None:
                    rep.update_from_observation(
                        context_tag=obs.context_tag,
                        action_type=obs.action_type,
                        success=obs.outcome_success,
                        fitness_delta=obs.outcome_fitness_delta,
                        actor_id=obs.actor_id,
                    )

    def _evaluate_adoption(
        self,
        agent_id: str,
        tick: int,
    ) -> list[TransmissionEvent]:
        """Evaluate whether this agent should adopt new cultural variants.

        For each context where the agent has observations but no adopted
        variant, compute combined bias scores and probabilistically adopt
        the best candidate.

        Args:
            agent_id: String agent identifier
            tick: Current tick number

        Returns:
            List of TransmissionEvents (adopted or not)
        """
        from src.evolution.transmission_biases import compute_combined_bias

        events: list[TransmissionEvent] = []

        mem = self._observation_memories.get(agent_id)
        rep = self._repertoires.get(agent_id)
        weights = self._transmission_weights.get(agent_id)

        if not mem or not rep or not weights:
            return events

        # Find all contexts where we have recent observations
        observed_contexts: set[str] = set()
        for obs in mem.recent(100):
            observed_contexts.add(obs.context_tag)

        for context in observed_contexts:
            # Skip if already have an adopted variant for this context
            if rep.lookup(context) is not None:
                continue

            # Compute combined bias scores for variants in this context
            bias_scores = compute_combined_bias(mem, context, weights)

            if not bias_scores:
                continue

            # Pick best variant
            best_vid: str = max(bias_scores, key=lambda k: bias_scores[k])
            best_score = bias_scores[best_vid]

            if best_score < self.adoption_threshold:
                continue

            # Probabilistic adoption: score = probability
            adopted = random.random() < best_score

            # Determine dominant bias for logging
            dominant = weights.dominant_style.value

            event = TransmissionEvent(
                tick=tick,
                observer_id=agent_id,
                actor_id="",  # could be multiple contributors
                variant_id=best_vid,
                bias_type=dominant,
                adoption_probability=best_score,
                adopted=adopted,
            )
            events.append(event)

            if adopted:
                parts = best_vid.split(":", 1)
                if len(parts) == 2:
                    context_tag, action_type = parts
                    rep.adopt(context_tag, action_type, tick)
                self._last_adoption_tick[agent_id] = tick

        return events

    def _evaluate_unadoption(self, agent_id: str) -> None:
        """Remove adopted variants that perform poorly when the agent uses them.

        If an agent has used a variant 5+ times and its own success rate
        is below unadoption_threshold, unadopt it so the agent reverts
        to personality-based behavior in that context.

        Args:
            agent_id: String agent identifier
        """
        rep = self._repertoires.get(agent_id)
        if not rep:
            return

        for variant in rep.adopted_variants():
            if variant.times_used >= 5 and variant.own_success_rate < self.unadoption_threshold:
                rep.unadopt(variant.context_tag, variant.action_type)

    @property
    def transmission_events(self) -> list[TransmissionEvent]:
        """All transmission events across the simulation."""
        return self._transmission_events

    def get_cultural_stats(self) -> dict[str, Any]:
        """Aggregate cultural statistics for metrics and analysis.

        Returns:
            Dict with variant counts, adoption counts, and bias distribution.
        """
        total_adopted = sum(rep.adopted_count() for rep in self._repertoires.values())
        total_variants = sum(rep.variant_count() for rep in self._repertoires.values())

        # Count successful adoption events
        adoption_count = sum(1 for e in self._transmission_events if e.adopted)

        # Bias distribution across adoptions
        bias_counts: dict[str, int] = {}
        for e in self._transmission_events:
            if e.adopted:
                bias_counts[e.bias_type] = bias_counts.get(e.bias_type, 0) + 1

        return {
            "total_variants_known": total_variants,
            "total_adopted": total_adopted,
            "transmission_events": len(self._transmission_events),
            "adoption_events": adoption_count,
            "bias_distribution": bias_counts,
            "agents_tracked": len(self._repertoires),
        }
