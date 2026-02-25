"""Metacognitive control logic: strategy switching and threshold adjustment.

The MetacognitiveController makes high-level decisions about HOW an agent
should reason: which strategy to use, when to deliberate, when to seek help.
Stateless — all decisions are pure functions of input state.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.metacognition.types import AgentMetacogState


@dataclass(frozen=True)
class ControlDecision:
    """Output of metacognitive control: what changes to apply.

    All fields default to "no change" values. Engine applies only the
    non-default values.
    """

    switch_strategy: bool = False
    new_strategy_name: str = ""
    switch_reason: str = ""
    new_deliberation_threshold: float | None = None  # None = no change
    seek_help: bool = False
    help_target_id: str | None = None


class MetacognitiveController:
    """Decides when to switch strategies, adjust deliberation, or seek help.

    All decisions are based on performance data in AgentMetacogState.
    Pure logic — no side effects, no state mutation (that's the engine's job).
    """

    def __init__(
        self,
        switch_threshold: float = 0.3,
        switch_patience: int = 5,
        help_confidence_threshold: float = 0.3,
        help_stakes_threshold: float = 0.6,
        deliberation_adjustment_rate: float = 0.05,
        switch_cooldown: int = 10,
    ):
        """Initialize control parameters.

        Args:
            switch_threshold: Switch if recent success rate < this
            switch_patience: Wait N ticks of underperformance before switching
            help_confidence_threshold: Seek help if FOK < this
            help_stakes_threshold: Seek help if threat > this
            deliberation_adjustment_rate: How much to adjust threshold per tick
            switch_cooldown: Minimum ticks between strategy switches
        """
        self.switch_threshold = switch_threshold
        self.switch_patience = switch_patience
        self.help_confidence_threshold = help_confidence_threshold
        self.help_stakes_threshold = help_stakes_threshold
        self.deliberation_adjustment_rate = deliberation_adjustment_rate
        self.switch_cooldown = switch_cooldown

    def decide(
        self,
        agent_id: str,
        tick: int,
        state: AgentMetacogState,
        context: str,
        fok: float,
        threat: float,
        nearby_agent_ids: list[str],
    ) -> ControlDecision:
        """Make all control decisions for one agent this tick.

        Combines three independent sub-decisions:
        1. Strategy switching (based on performance)
        2. Deliberation threshold adjustment (based on calibration)
        3. Help-seeking (based on FOK and threat)

        Args:
            agent_id: Agent identifier (for logging, not used in logic)
            tick: Current simulation tick
            state: Agent's metacognitive state bundle
            context: Current context tag (e.g., "low_food")
            fok: Feeling of knowing (0-1)
            threat: Current threat level (0-1)
            nearby_agent_ids: List of agent IDs within help-seeking range

        Returns:
            ControlDecision with all changes to apply
        """
        # Strategy switching decision
        switch_decision = self._decide_strategy_switch(tick, state, context)

        # Deliberation threshold adjustment
        new_threshold = self._decide_deliberation_adjustment(state)

        # Help-seeking decision
        seek_help_decision = self._decide_help_seeking(fok, threat, nearby_agent_ids)

        # Combine all decisions
        return ControlDecision(
            switch_strategy=switch_decision[0],
            new_strategy_name=switch_decision[1],
            switch_reason=switch_decision[2],
            new_deliberation_threshold=new_threshold,
            seek_help=seek_help_decision[0],
            help_target_id=seek_help_decision[1],
        )

    def _decide_strategy_switch(
        self, tick: int, state: AgentMetacogState, context: str
    ) -> tuple[bool, str, str]:
        """Decide whether to switch strategies.

        Returns:
            (should_switch, new_strategy_name, reason)
        """
        # Get performance for current strategy in current context
        key = (state.current_strategy_name, context)
        perf = state.strategy_performance.get(key)

        # If no performance data or performance is acceptable, reset streak
        if perf is None or perf.recent_success_rate >= self.switch_threshold:
            state.underperformance_streak = 0
            return (False, "", "")

        # Performance is below threshold — increment streak
        state.underperformance_streak += 1

        # Check if we should switch
        if (
            state.underperformance_streak >= self.switch_patience
            and (tick - state.last_switch_tick) >= self.switch_cooldown
        ):
            # Find best alternative strategy
            best_strategy = self._find_best_alternative(state, context)
            return (True, best_strategy, "underperforming")

        return (False, "", "")

    def _find_best_alternative(self, state: AgentMetacogState, context: str) -> str:
        """Find the best alternative strategy to switch to.

        Looks at recent_success_rate for each available strategy (excluding
        current) in the current context. Falls back to "planning" as
        escalation default if no data.

        Args:
            state: Agent's metacognitive state
            context: Current context tag

        Returns:
            Name of best alternative strategy
        """
        current = state.current_strategy_name
        alternatives = [s for s in state.available_strategies if s != current]

        if not alternatives:
            return "planning"  # Fallback if somehow no alternatives

        best_strategy = None
        best_rate = -1.0

        for strategy_name in alternatives:
            key = (strategy_name, context)
            perf = state.strategy_performance.get(key)
            if perf is not None:
                rate = perf.recent_success_rate
                if rate > best_rate:
                    best_rate = rate
                    best_strategy = strategy_name

        # If no strategy has data, escalate to "planning" as default
        if best_strategy is None:
            result: str = "planning" if "planning" in alternatives else alternatives[0]
            return result

        return best_strategy

    def _decide_deliberation_adjustment(self, state: AgentMetacogState) -> float | None:
        """Decide whether to adjust deliberation threshold.

        Based on expected calibration error (ECE):
        - High ECE (>0.15) → lower threshold (deliberate more)
        - Low ECE (<0.05) → raise threshold (trust System 1 more)

        Returns:
            New threshold value, or None if no change
        """
        ece = state.calibration.expected_calibration_error
        current_threshold = state.deliberation_threshold

        new_threshold = current_threshold

        if ece > 0.15:
            # Poor calibration — deliberate more
            new_threshold = current_threshold - self.deliberation_adjustment_rate
        elif ece < 0.05:
            # Good calibration — trust intuition more
            new_threshold = current_threshold + self.deliberation_adjustment_rate

        # Clamp to valid range
        new_threshold = max(0.3, min(0.9, new_threshold))

        # Only return if changed
        if abs(new_threshold - current_threshold) < 1e-6:
            return None

        return new_threshold

    def _decide_help_seeking(
        self, fok: float, threat: float, nearby_agent_ids: list[str]
    ) -> tuple[bool, str | None]:
        """Decide whether to seek help from nearby agents.

        Seeks help when:
        - Low confidence (FOK < threshold)
        - High stakes (threat > threshold)
        - Help is available (nearby agents exist)

        Returns:
            (should_seek_help, target_agent_id)
        """
        if (
            fok < self.help_confidence_threshold
            and threat > self.help_stakes_threshold
            and len(nearby_agent_ids) > 0
        ):
            # Seek help from nearest agent (first in list)
            return (True, nearby_agent_ids[0])

        return (False, None)
