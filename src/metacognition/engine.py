"""Metacognitive engine: orchestrates agent self-monitoring, control, and strategy switching.

The MetacognitiveEngine is the central hub for Phase 3, managing the full cycle:
1. Monitor performance (via MetacognitiveMonitor) → PredictionRecord, StrategyOutcomeRecord
2. Update agent state (calibration, strategy performance, self-model)
3. Compute feeling-of-knowing (FOK) for current situation
4. Make control decisions (via MetacognitiveController) → strategy switch, deliberation adjustment
5. Apply changes by rebuilding strategy wrapper chains
6. Generate snapshots for visualization and trajectory analysis

This is the "agents reason about their own reasoning" layer — the novel contribution
that integrates monitoring, control, and self-modeling into multi-agent societies.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

from src.awareness.context import ContextTag
from src.metacognition.controller import MetacognitiveController
from src.metacognition.monitor import MetacognitiveMonitor
from src.metacognition.self_model import CognitiveSelfModel
from src.metacognition.types import (
    AgentMetacogState,
    MetacognitiveSnapshot,
    StrategyPerformance,
)

if TYPE_CHECKING:
    from src.simulation.engine import AgentTickRecord, SimulationEngine


class MetacognitiveEngine:
    """Orchestrates metacognitive monitoring and control for all agents.

    Follows the same pattern as CulturalTransmissionEngine: single dict for
    per-agent state, strategy instances owned by engine, strategy switching
    rebuilds the full wrapper chain (base → cultural → metacog).

    Called once per tick (Phase 8.8 in engine.step_all) after cultural transmission.
    """

    def __init__(
        self,
        switch_threshold: float = 0.3,
        switch_patience: int = 5,
        help_confidence_threshold: float = 0.3,
        help_stakes_threshold: float = 0.6,
        deliberation_adjustment_rate: float = 0.05,
        switch_cooldown: int = 10,
        fok_enabled: bool = True,
    ):
        """Initialize metacognitive system.

        Args:
            switch_threshold: Switch if recent success rate < this
            switch_patience: Wait N ticks of underperformance before switching
            help_confidence_threshold: Seek help if FOK < this
            help_stakes_threshold: Seek help if threat > this
            deliberation_adjustment_rate: How much to adjust threshold per tick
            switch_cooldown: Minimum ticks between strategy switches
            fok_enabled: Whether to blend FOK into intention confidence
        """
        self._switch_threshold = switch_threshold
        self._switch_patience = switch_patience
        self._help_confidence_threshold = help_confidence_threshold
        self._help_stakes_threshold = help_stakes_threshold
        self._deliberation_adjustment_rate = deliberation_adjustment_rate
        self._switch_cooldown = switch_cooldown
        self._fok_enabled = fok_enabled

        # Per-agent state (single dict for all metacognitive data)
        self._agents: dict[str, AgentMetacogState] = {}

        # Engine owns strategy instances per agent
        self._strategy_instances: dict[str, dict[str, Any]] = {}

        # Shared stateless components
        self._monitor = MetacognitiveMonitor()
        self._controller = MetacognitiveController(
            switch_threshold=switch_threshold,
            switch_patience=switch_patience,
            help_confidence_threshold=help_confidence_threshold,
            help_stakes_threshold=help_stakes_threshold,
            deliberation_adjustment_rate=deliberation_adjustment_rate,
            switch_cooldown=switch_cooldown,
        )

        # Snapshot history for analysis
        self._snapshots: list[MetacognitiveSnapshot] = []

    def register_agent(
        self,
        agent_id: str,
        initial_strategy_name: str,
        available_strategies: list[str],
        strategy_instances: dict[str, Any],
        initial_deliberation_threshold: float = 0.7,
    ) -> None:
        """Register a new agent with the metacognitive system.

        Creates AgentMetacogState with empty calibration, strategy performance,
        and self-model. Stores strategy instances for later switching.

        Args:
            agent_id: String agent identifier
            initial_strategy_name: Name of starting strategy (e.g., "personality")
            available_strategies: List of strategy names this agent can switch between
            strategy_instances: Dict mapping strategy name -> strategy instance
            initial_deliberation_threshold: Starting deliberation threshold (0.3-0.9)
        """
        # Preserve existing state on re-registration
        if agent_id in self._agents:
            old_state = self._agents[agent_id]
            state = AgentMetacogState(
                calibration=old_state.calibration,
                strategy_performance=old_state.strategy_performance,
                self_model=old_state.self_model,
                deliberation_threshold=initial_deliberation_threshold,
                current_strategy_name=initial_strategy_name,
                available_strategies=available_strategies,
                switch_history=old_state.switch_history,
                last_switch_tick=old_state.last_switch_tick,
                underperformance_streak=old_state.underperformance_streak,
            )
        else:
            from src.metacognition.types import CalibrationTracker

            state = AgentMetacogState(
                calibration=CalibrationTracker(),
                strategy_performance={},
                self_model=CognitiveSelfModel(),
                deliberation_threshold=initial_deliberation_threshold,
                current_strategy_name=initial_strategy_name,
                available_strategies=available_strategies,
                switch_history=deque(maxlen=200),
                last_switch_tick=-999,
                underperformance_streak=0,
            )

        self._agents[agent_id] = state
        self._strategy_instances[agent_id] = strategy_instances

    def unregister_agent(self, agent_id: str) -> None:
        """Mark agent as inactive. Keep data for post-mortem analysis."""
        # Intentionally keep data — useful for metacognitive trajectory analysis
        pass

    def tick(
        self,
        engine: SimulationEngine,
        tick: int,
        agent_records: list[AgentTickRecord],
    ) -> list[MetacognitiveSnapshot]:
        """Process metacognitive cycle for this tick.

        For each agent:
        1. Skip if not registered or dead
        2. Extract awareness loop data (sensation, intention, etc.)
        3. Monitor: evaluate performance → (prediction, outcome)
        4. Update: calibration, strategy performance, self-model
        5. Compute: feeling-of-knowing (FOK)
        6. Control: decide strategy switch, deliberation adjustment, help-seeking
        7. Apply: rebuild strategy chain if switch, adjust deliberation threshold
        8. Snapshot: record metacognitive state for this tick
        9. Set FOK on strategy wrapper for NEXT tick's SRIE

        Args:
            engine: SimulationEngine instance
            tick: Current tick number
            agent_records: List of AgentTickRecord from this tick

        Returns:
            List of MetacognitiveSnapshot (one per registered living agent)
        """
        snapshots: list[MetacognitiveSnapshot] = []

        for record in agent_records:
            agent_id = str(record.agent_id)

            # Skip if not registered
            if agent_id not in self._agents:
                continue

            # Skip if agent is dead
            agent = engine.registry.get(record.agent_id)
            if not agent or not agent.alive:
                continue

            # Get awareness loop
            loop = engine.registry.get_awareness_loop(record.agent_id)
            if not loop or not loop.last_intention:
                continue

            state = self._agents[agent_id]

            # Extract context tag from sensation
            context_tag = (
                ContextTag.extract_primary(loop.last_sensation) if loop.last_sensation else "alone"
            )

            # Get current strategy name (from wrapper or state)
            strategy_name = state.current_strategy_name
            if hasattr(loop.strategy, "_active_strategy_name"):
                strategy_name = loop.strategy._active_strategy_name

            # Check if deliberation was used
            deliberation_used = False
            if loop._deliberation and hasattr(loop._deliberation, "should_escalate"):
                deliberation_used = loop._deliberation.should_escalate(
                    loop.last_sensation, loop.last_reflection
                )

            # Monitor: evaluate this tick's performance
            prediction, outcome = self._monitor.evaluate_tick(
                agent_id=agent_id,
                tick=tick,
                sensation=loop.last_sensation,
                reflection=loop.last_reflection,
                intention=loop.last_intention,
                action_result=record.result,
                needs_before=record.needs_before,
                needs_after=record.needs_after,
                strategy_name=strategy_name,
                deliberation_used=deliberation_used,
            )

            # Update calibration
            state.calibration.record(prediction)

            # Update strategy performance
            perf_key = (strategy_name, context_tag)
            if perf_key not in state.strategy_performance:
                state.strategy_performance[perf_key] = StrategyPerformance(
                    strategy_name=strategy_name, context_tag=context_tag
                )
            state.strategy_performance[perf_key].record(
                succeeded=outcome.succeeded, fitness_delta=outcome.needs_delta_sum
            )

            # Update self-model
            domain = self._action_type_to_domain(outcome.action_type)
            prediction_error = abs(
                prediction.predicted_confidence - (1.0 if prediction.was_correct else 0.0)
            )
            from src.metacognition.self_model import CognitiveSelfModel

            if isinstance(state.self_model, CognitiveSelfModel):
                state.self_model.update_from_outcome(
                    domain=domain,
                    succeeded=outcome.succeeded,
                    prediction_error=prediction_error,
                )

            # Compute feeling-of-knowing
            fok = self._monitor.compute_feeling_of_knowing(
                calibration=state.calibration,
                strategy_perf=state.strategy_performance,
                context_tag=context_tag,
                self_model=state.self_model,
            )

            # Control: make decisions
            threat = loop.last_reflection.threat_level if loop.last_reflection else 0.0
            nearby_agent_ids = [
                str(a.agent_id)
                for a in (loop.last_sensation.visible_agents if loop.last_sensation else [])
            ]

            decision = self._controller.decide(
                agent_id=agent_id,
                tick=tick,
                state=state,
                context=context_tag,
                fok=fok,
                threat=threat,
                nearby_agent_ids=nearby_agent_ids,
            )

            # Track switch metadata for snapshot
            strategy_switch_this_tick = False
            previous_strategy = ""
            switch_reason = ""

            # Apply strategy switch if decided
            if decision.switch_strategy:
                previous_strategy = state.current_strategy_name
                self._apply_strategy_switch(
                    engine=engine,
                    agent_id=agent_id,
                    real_agent_id=record.agent_id,
                    new_strategy_name=decision.new_strategy_name,
                    tick=tick,
                )
                strategy_switch_this_tick = True
                switch_reason = decision.switch_reason

            # Apply deliberation threshold adjustment if decided
            if decision.new_deliberation_threshold is not None:
                self._apply_deliberation_change(
                    engine=engine,
                    agent_id=agent_id,
                    real_agent_id=record.agent_id,
                    new_threshold=decision.new_deliberation_threshold,
                )

            # Build snapshot
            self_awareness_score = self._compute_self_awareness_score(agent_id)

            from src.metacognition.self_model import CognitiveSelfModel

            self_model_summary: dict[str, float] = {}
            if isinstance(state.self_model, CognitiveSelfModel):
                self_model_summary = state.self_model.capability_summary()

            snapshot = MetacognitiveSnapshot(
                tick=tick,
                agent_id=agent_id,
                active_strategy=state.current_strategy_name,
                calibration_score=state.calibration.calibration_score,
                self_awareness_score=self_awareness_score,
                confidence_bias=state.calibration.confidence_bias,
                strategy_switch_this_tick=strategy_switch_this_tick,
                previous_strategy=previous_strategy,
                switch_reason=switch_reason,
                deliberation_threshold=state.deliberation_threshold,
                self_model_summary=self_model_summary,
            )
            snapshots.append(snapshot)

            # Set FOK on metacognitive strategy wrapper for NEXT tick
            if hasattr(loop.strategy, "set_fok"):
                loop.strategy.set_fok(fok)

        self._snapshots.extend(snapshots)
        return snapshots

    def _apply_strategy_switch(
        self,
        engine: SimulationEngine,
        agent_id: str,
        real_agent_id: Any,
        new_strategy_name: str,
        tick: int,
    ) -> None:
        """Switch agent to a new strategy by rebuilding the full wrapper chain.

        Chain structure: base_strategy → CulturalWrapper (if applicable) → MetacogWrapper

        Args:
            engine: SimulationEngine instance
            agent_id: String agent identifier
            real_agent_id: The original AgentID object for registry lookups
            new_strategy_name: Name of strategy to switch to
            tick: Current tick (for history logging)
        """
        from src.cognition.strategies.cultural_strategy import CulturallyModulatedStrategy
        from src.cognition.strategies.metacognitive_strategy import MetacognitiveStrategy

        state = self._agents[agent_id]
        old_strategy_name = state.current_strategy_name

        # Get awareness loop
        loop = engine.registry.get_awareness_loop(real_agent_id)
        if not loop:
            return

        # Get new base strategy instance
        if new_strategy_name not in self._strategy_instances[agent_id]:
            return
        new_base = self._strategy_instances[agent_id][new_strategy_name]

        # Check if cultural wrapping is needed
        needs_cultural_wrapper = False
        repertoire = None
        cultural_override_prob = 0.7

        if hasattr(engine, "cultural_engine") and engine.cultural_engine:
            repertoire = engine.cultural_engine.get_repertoire(agent_id)
            if repertoire is not None:
                needs_cultural_wrapper = True
            # Get override probability from config if available
            if hasattr(engine, "config"):
                cultural_override_prob = getattr(
                    engine.config, "cultural_override_probability", 0.7
                )

        # Rebuild wrapper chain
        wrapped_strategy = new_base

        # Layer 1: Cultural wrapper (if applicable)
        if needs_cultural_wrapper:
            wrapped_strategy = CulturallyModulatedStrategy(
                inner_strategy=wrapped_strategy,
                repertoire=repertoire,
                override_probability=cultural_override_prob,
            )

        # Layer 2: Metacognitive wrapper (outermost)
        wrapped_strategy = MetacognitiveStrategy(
            inner_strategy=wrapped_strategy,
            strategy_name=new_strategy_name,
            fok_enabled=self._fok_enabled,
        )

        # Assign to loop
        loop.strategy = wrapped_strategy

        # Update state
        state.current_strategy_name = new_strategy_name
        state.switch_history.append((tick, old_strategy_name, new_strategy_name, "underperforming"))
        state.last_switch_tick = tick

    def _apply_deliberation_change(
        self,
        engine: SimulationEngine,
        agent_id: str,
        real_agent_id: Any,
        new_threshold: float,
    ) -> None:
        """Adjust agent's deliberation threshold.

        Mutates the deliberation strategy's threshold if it supports it.

        Args:
            engine: SimulationEngine instance
            agent_id: String agent identifier
            real_agent_id: The original AgentID object for registry lookups
            new_threshold: New threshold value (0.3-0.9)
        """
        state = self._agents[agent_id]

        # Get awareness loop
        loop = engine.registry.get_awareness_loop(real_agent_id)
        if not loop:
            return

        # Update deliberation threshold if supported
        if loop._deliberation and hasattr(loop._deliberation, "threat_threshold"):
            loop._deliberation.threat_threshold = new_threshold

        # Update state
        state.deliberation_threshold = new_threshold

    def _compute_self_awareness_score(self, agent_id: str) -> float:
        """Compute composite self-awareness score for visualization.

        WARNING: This is a heuristic composite with arbitrary weights.
        Used for visualization only, not for control decisions.

        Components:
        - Calibration quality (0.4 weight): how well predictions match reality
        - Model stability (0.3 weight): average confidence across self-model ratings
        - Adaptiveness (0.2 weight): has agent switched strategies?
        - Bias awareness (0.1 weight): how much is agent over/underconfident?

        Args:
            agent_id: String agent identifier

        Returns:
            Float in [0.0, 1.0] representing composite self-awareness
        """
        state = self._agents.get(agent_id)
        if not state:
            return 0.0

        # Signal 1: Calibration quality (0.4 weight)
        calibration_signal = state.calibration.calibration_score

        # Signal 2: Model stability (0.3 weight) — average confidence across domains
        from src.metacognition.self_model import CognitiveSelfModel

        model_confidences: list[float] = []
        if isinstance(state.self_model, CognitiveSelfModel):
            model_confidences = [rating.confidence for rating in state.self_model._ratings.values()]
        model_stability = (
            sum(model_confidences) / len(model_confidences) if model_confidences else 0.0
        )

        # Signal 3: Adaptiveness (0.2 weight) — has agent switched strategies?
        adaptiveness = 1.0 if len(state.switch_history) > 0 else 0.5

        # Signal 4: Bias awareness (0.1 weight) — lower bias = better awareness
        bias_awareness = 1.0 - min(1.0, abs(state.calibration.confidence_bias))

        # Weighted combination
        score = (
            0.4 * calibration_signal
            + 0.3 * model_stability
            + 0.2 * adaptiveness
            + 0.1 * bias_awareness
        )

        return max(0.0, min(1.0, score))

    def _action_type_to_domain(self, action_type: str) -> str:
        """Map action type to self-model capability domain.

        Args:
            action_type: ActionType.value string (e.g., "gather", "move")

        Returns:
            Domain name for self-model (e.g., "gathering", "exploration")
        """
        # Direct mappings
        if action_type == "gather":
            return "gathering"
        if action_type == "move":
            return "exploration"
        if action_type == "attack":
            return "combat"
        if action_type in ("give", "share"):
            return "social"
        if action_type in ("rest", "eat", "drink"):
            return "gathering"

        # Default
        return "planning"

    # =========================================================================
    # Accessors (for testing, metrics, visualization)
    # =========================================================================

    def get_agent_state(self, agent_id: str) -> AgentMetacogState | None:
        """Get an agent's full metacognitive state."""
        return self._agents.get(agent_id)

    def get_calibration(self, agent_id: str) -> Any | None:
        """Get an agent's calibration tracker."""
        state = self._agents.get(agent_id)
        return state.calibration if state else None

    def get_self_model(self, agent_id: str) -> CognitiveSelfModel | None:
        """Get an agent's cognitive self-model."""
        from src.metacognition.self_model import CognitiveSelfModel

        state = self._agents.get(agent_id)
        if state and isinstance(state.self_model, CognitiveSelfModel):
            return state.self_model
        return None

    def get_strategy_performance(self, agent_id: str) -> dict | None:
        """Get an agent's strategy performance tracker."""
        state = self._agents.get(agent_id)
        return state.strategy_performance if state else None

    def get_switch_history(self, agent_id: str) -> deque | None:
        """Get an agent's strategy switch history."""
        state = self._agents.get(agent_id)
        return state.switch_history if state else None

    def get_metacognitive_stats(self) -> dict[str, Any]:
        """Aggregate metacognitive statistics across all agents.

        Returns:
            Dict with total agents, avg calibration, strategy distribution, total switches
        """
        if not self._agents:
            return {
                "total_agents_tracked": 0,
                "avg_calibration_score": 0.0,
                "strategy_distribution": {},
                "total_switches": 0,
            }

        # Average calibration score
        total_calibration = sum(
            state.calibration.calibration_score for state in self._agents.values()
        )
        avg_calibration = total_calibration / len(self._agents)

        # Strategy distribution (current strategies)
        strategy_counts: dict[str, int] = {}
        for state in self._agents.values():
            strategy = state.current_strategy_name
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        # Total switches
        total_switches = sum(len(state.switch_history) for state in self._agents.values())

        return {
            "total_agents_tracked": len(self._agents),
            "avg_calibration_score": avg_calibration,
            "strategy_distribution": strategy_counts,
            "total_switches": total_switches,
        }

    @property
    def snapshots(self) -> list[MetacognitiveSnapshot]:
        """All metacognitive snapshots across the simulation."""
        return self._snapshots
