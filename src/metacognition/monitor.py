"""Metacognitive monitoring: evaluating agent performance post-tick.

The MetacognitiveMonitor is STATELESS — it's a pure evaluator that converts
tick outcomes into structured records for learning. All state lives elsewhere
(CalibrationTracker, StrategyPerformance, CognitiveSelfModel).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.awareness.context import ContextTag
from src.metacognition.types import PredictionRecord, StrategyOutcomeRecord

if TYPE_CHECKING:
    from src.awareness.types import Intention, Reflection, Sensation
    from src.metacognition.types import CalibrationTracker
    from src.simulation.actions import ActionResult


class MetacognitiveMonitor:
    """Stateless post-tick evaluator for metacognitive learning.

    Converts raw tick data (sensation, intention, action_result) into
    structured PredictionRecord and StrategyOutcomeRecord for tracking.
    """

    def evaluate_tick(
        self,
        agent_id: str,
        tick: int,
        sensation: Sensation | None,
        reflection: Reflection | None,
        intention: Intention | None,
        action_result: ActionResult | None,
        needs_before: dict[str, float],
        needs_after: dict[str, float],
        strategy_name: str,
        deliberation_used: bool,
    ) -> tuple[PredictionRecord, StrategyOutcomeRecord]:
        """Evaluate a single tick and produce monitoring records.

        Args:
            agent_id: The agent being evaluated
            tick: Current simulation tick
            sensation: Agent's perception (None if not available)
            reflection: Agent's reflection (None if not available)
            intention: Agent's formed intention (None if not available)
            action_result: Result of action execution (None if not executed)
            needs_before: Agent's needs before action
            needs_after: Agent's needs after action
            strategy_name: Which strategy was used
            deliberation_used: Whether deliberation was triggered

        Returns:
            Tuple of (PredictionRecord, StrategyOutcomeRecord)
        """
        # Extract context from sensation
        context_tag = ContextTag.extract_primary(sensation) if sensation else "alone"

        # Create prediction record for action confidence
        predicted_confidence = intention.confidence if intention else 0.5
        action_succeeded = action_result.success if action_result else False

        prediction = PredictionRecord(
            tick=tick,
            domain="action_success",
            predicted_confidence=predicted_confidence,
            actual_outcome="succeed" if action_succeeded else "fail",
            was_correct=action_succeeded,
            context_tag=context_tag,
            strategy_name=strategy_name,
        )

        # Calculate needs delta sum from action result
        needs_delta_sum = 0.0
        if action_result and action_result.needs_delta:
            needs_delta_sum = sum(action_result.needs_delta.values())

        # Extract action type
        action_type = "wait"
        if action_result and action_result.action:
            action_type = action_result.action.type.value

        # Create strategy outcome record
        outcome = StrategyOutcomeRecord(
            tick=tick,
            strategy_name=strategy_name,
            context_tag=context_tag,
            action_type=action_type,
            succeeded=action_succeeded,
            needs_delta_sum=needs_delta_sum,
            confidence=predicted_confidence,
            deliberation_used=deliberation_used,
        )

        return prediction, outcome

    def compute_feeling_of_knowing(
        self,
        calibration: CalibrationTracker,
        strategy_perf: dict,
        context_tag: str,
        self_model: object | None,
    ) -> float:
        """Compute the agent's feeling-of-knowing score for current situation.

        Combines three signals:
        - Calibration quality: how well the agent's past predictions matched reality
        - Strategy fitness: how well the current strategy performs in this context
        - Self-model capability: the agent's learned rating for this domain

        Args:
            calibration: The agent's calibration tracker
            strategy_perf: Dict mapping (strategy_name, context_tag) -> StrategyPerformance
            context_tag: Current context
            self_model: The agent's cognitive self-model (CognitiveSelfModel or None)

        Returns:
            Float in [0.0, 1.0] where higher = stronger feeling of knowing
        """
        # Signal 1: Calibration quality (0.4 weight)
        # Higher calibration score = more FOK
        calibration_signal = calibration.calibration_score

        # Signal 2: Strategy fitness in this context (0.3 weight)
        # Get current strategy name from first key in strategy_perf
        current_strategy = None
        for (strat, _ctx), _perf in strategy_perf.items():
            current_strategy = strat
            break

        strategy_signal = 0.5  # default prior
        if current_strategy:
            key = (current_strategy, context_tag)
            if key in strategy_perf:
                strategy_signal = strategy_perf[key].recent_success_rate

        # Signal 3: Self-model capability (0.3 weight)
        self_model_signal = 0.5  # default prior
        if self_model:
            from src.metacognition.self_model import CognitiveSelfModel

            if isinstance(self_model, CognitiveSelfModel):
                domain = self._context_to_domain(context_tag)
                rating = self_model.get_rating(domain)
                self_model_signal = rating.rating

        # Weighted combination
        fok = 0.4 * calibration_signal + 0.3 * strategy_signal + 0.3 * self_model_signal

        # Clamp to [0, 1]
        return max(0.0, min(1.0, fok))

    def _context_to_domain(self, context_tag: str) -> str:
        """Map context tag to capability domain for self-model lookup.

        Args:
            context_tag: Context tag from ContextTag

        Returns:
            Domain name for self-model
        """
        # Urgent needs / resource proximity -> gathering
        if context_tag in ["low_hunger", "low_thirst", "near_food", "near_water"]:
            return "gathering"

        # Social contexts -> social
        if context_tag in ["near_agent", "near_hostile", "near_ally", "crowded"]:
            return "social"

        # Threat contexts -> threat_assessment
        if context_tag in ["near_hostile"]:
            return "threat_assessment"

        # State contexts
        if context_tag == "alone":
            return "exploration"

        # Default
        return "planning"
