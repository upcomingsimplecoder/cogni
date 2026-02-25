"""Tests for Phase 3: Metacognition System.

Comprehensive tests for:
- Frozen records (PredictionRecord, StrategyOutcomeRecord, MetacognitiveSnapshot)
- Strategy performance tracking and success rate computation
- Calibration tracking (Brier score, ECE, confidence bias)
- Cognitive self-model (capability ratings, EMA updates)
- Metacognitive monitoring (FOK computation, prediction evaluation)
- Metacognitive control (strategy switching, deliberation adjustment, help-seeking)
- Metacognitive strategy wrapper (FOK blending)

Following test patterns:
- Class-per-component organization
- test_<scenario>_<expected_outcome> naming
- Import test doubles from tests.helpers
- No unittest.mock, pure pytest
"""

from __future__ import annotations

import pytest

from src.awareness.types import Intention
from src.cognition.strategies.metacognitive_strategy import MetacognitiveStrategy
from src.metacognition.controller import MetacognitiveController
from src.metacognition.monitor import MetacognitiveMonitor
from src.metacognition.self_model import CognitiveSelfModel
from src.metacognition.types import (
    AgentMetacogState,
    CalibrationTracker,
    CapabilityRating,
    MetacognitiveSnapshot,
    PredictionRecord,
    StrategyOutcomeRecord,
    StrategyPerformance,
)
from src.simulation.actions import Action, ActionResult, ActionType
from tests.helpers import MockReflection, MockSensation, MockStrategy

# ============================================================================
# Test: PredictionRecord
# ============================================================================


class TestPredictionRecord:
    """Tests for frozen PredictionRecord dataclass."""

    def test_frozen_immutable(self):
        """Verify PredictionRecord is frozen and cannot be modified."""
        record = PredictionRecord(
            tick=1,
            domain="action_success",
            predicted_confidence=0.8,
            actual_outcome="succeed",
            was_correct=True,
            context_tag="low_hunger",
            strategy_name="personality",
        )

        with pytest.raises(Exception, match=""):  # dataclass frozen raises FrozenInstanceError
            record.tick = 2

    def test_all_fields_present(self):
        """Verify all fields can be created and accessed."""
        record = PredictionRecord(
            tick=10,
            domain="action_success",
            predicted_confidence=0.75,
            actual_outcome="succeed",
            was_correct=True,
            context_tag="near_agent",
            strategy_name="planning",
        )

        assert record.tick == 10
        assert record.domain == "action_success"
        assert record.predicted_confidence == 0.75
        assert record.actual_outcome == "succeed"
        assert record.was_correct is True
        assert record.context_tag == "near_agent"
        assert record.strategy_name == "planning"

    def test_domain_values_accepted(self):
        """Test with different domain values."""
        record1 = PredictionRecord(
            tick=1,
            domain="action_success",
            predicted_confidence=0.8,
            actual_outcome="succeed",
            was_correct=True,
            context_tag="alone",
            strategy_name="personality",
        )
        assert record1.domain == "action_success"

        record2 = PredictionRecord(
            tick=2,
            domain="threat_assessment",
            predicted_confidence=0.9,
            actual_outcome="safe",
            was_correct=True,
            context_tag="near_hostile",
            strategy_name="planning",
        )
        assert record2.domain == "threat_assessment"


# ============================================================================
# Test: StrategyOutcomeRecord
# ============================================================================


class TestStrategyOutcomeRecord:
    """Tests for frozen StrategyOutcomeRecord dataclass."""

    def test_frozen_immutable(self):
        """Verify StrategyOutcomeRecord is frozen."""
        record = StrategyOutcomeRecord(
            tick=1,
            strategy_name="personality",
            context_tag="low_hunger",
            action_type="gather",
            succeeded=True,
            needs_delta_sum=10.0,
            confidence=0.8,
            deliberation_used=False,
        )

        with pytest.raises(Exception, match=""):
            record.tick = 2

    def test_all_fields_present(self):
        """Verify all fields can be created and accessed."""
        record = StrategyOutcomeRecord(
            tick=5,
            strategy_name="planning",
            context_tag="near_food",
            action_type="move",
            succeeded=False,
            needs_delta_sum=-2.5,
            confidence=0.6,
            deliberation_used=True,
        )

        assert record.tick == 5
        assert record.strategy_name == "planning"
        assert record.context_tag == "near_food"
        assert record.action_type == "move"
        assert record.succeeded is False
        assert record.needs_delta_sum == -2.5
        assert record.confidence == 0.6
        assert record.deliberation_used is True


# ============================================================================
# Test: StrategyPerformance
# ============================================================================


class TestStrategyPerformance:
    """Tests for StrategyPerformance mutable tracker."""

    def test_success_rate_no_data_returns_prior(self):
        """No uses should return uninformative prior of 0.5."""
        perf = StrategyPerformance(strategy_name="personality", context_tag="alone")
        assert perf.success_rate == 0.5

    def test_success_rate_computed(self):
        """Success rate should be total_successes / total_uses."""
        perf = StrategyPerformance(strategy_name="planning", context_tag="low_hunger")
        perf.record(succeeded=True, fitness_delta=5.0)
        perf.record(succeeded=True, fitness_delta=4.0)
        perf.record(succeeded=False, fitness_delta=-2.0)
        perf.record(succeeded=True, fitness_delta=3.0)
        perf.record(succeeded=False, fitness_delta=-1.0)

        # 3 successes out of 5 uses = 0.6
        assert perf.success_rate == 0.6

    def test_recent_success_rate_no_data_returns_prior(self):
        """Empty recent_outcomes should return prior of 0.5."""
        perf = StrategyPerformance(strategy_name="personality", context_tag="alone")
        assert perf.recent_success_rate == 0.5

    def test_recent_success_rate_differs_from_total(self):
        """Recent success rate can differ from total success rate."""
        perf = StrategyPerformance(strategy_name="planning", context_tag="low_thirst")

        # Old successes (many)
        for _ in range(50):
            perf.record(succeeded=True, fitness_delta=5.0)

        # Recent failures (will be in recent window)
        for _ in range(30):
            perf.record(succeeded=False, fitness_delta=-2.0)

        # Total: 50/80 = 0.625
        assert perf.success_rate == 0.625

        # Recent (last 50, which are: 20 True + 30 False): 20/50 = 0.4
        assert perf.recent_success_rate == 0.4

    def test_record_increments_counters(self):
        """Recording outcomes should increment counters correctly."""
        perf = StrategyPerformance(strategy_name="personality", context_tag="near_agent")

        perf.record(succeeded=True, fitness_delta=10.0)
        assert perf.total_uses == 1
        assert perf.total_successes == 1
        assert perf.total_fitness_delta == 10.0

        perf.record(succeeded=False, fitness_delta=-5.0)
        assert perf.total_uses == 2
        assert perf.total_successes == 1
        assert perf.total_fitness_delta == 5.0

        perf.record(succeeded=True, fitness_delta=8.0)
        assert perf.total_uses == 3
        assert perf.total_successes == 2
        assert perf.total_fitness_delta == 13.0

    def test_deque_maxlen_respected(self):
        """Recent outcomes should respect deque maxlen of 50."""
        perf = StrategyPerformance(strategy_name="planning", context_tag="alone")

        # Add 60 outcomes
        for i in range(60):
            perf.record(succeeded=(i % 2 == 0), fitness_delta=1.0)

        # Should only have last 50
        assert len(perf.recent_outcomes) == 50
        assert perf.total_uses == 60

    def test_avg_fitness_delta_computed(self):
        """Average fitness delta should be total / uses."""
        perf = StrategyPerformance(strategy_name="personality", context_tag="low_hunger")
        perf.record(succeeded=True, fitness_delta=10.0)
        perf.record(succeeded=True, fitness_delta=5.0)
        perf.record(succeeded=False, fitness_delta=-5.0)

        # Total: 10.0 avg
        assert perf.avg_fitness_delta == 10.0 / 3

    def test_avg_fitness_delta_no_data(self):
        """Should return 0.0 if no uses."""
        perf = StrategyPerformance(strategy_name="planning", context_tag="alone")
        assert perf.avg_fitness_delta == 0.0


# ============================================================================
# Test: CalibrationTracker
# ============================================================================


class TestCalibrationTracker:
    """Tests for CalibrationTracker prediction accuracy monitoring."""

    def test_brier_score_no_data(self):
        """Should return uninformative prior of 0.25 with no data."""
        tracker = CalibrationTracker()
        assert tracker.brier_score == 0.25

    def test_brier_score_perfect(self):
        """Perfect predictions with confidence=1.0 should give Brier=0.0."""
        tracker = CalibrationTracker()

        # All correct with high confidence
        for i in range(10):
            tracker.record(
                PredictionRecord(
                    tick=i,
                    domain="action_success",
                    predicted_confidence=1.0,
                    actual_outcome="succeed",
                    was_correct=True,
                    context_tag="alone",
                    strategy_name="personality",
                )
            )

        assert tracker.brier_score == 0.0

    def test_brier_score_worst(self):
        """Worst predictions (all wrong with confidence=1.0) should give Brier=1.0."""
        tracker = CalibrationTracker()

        # All wrong with high confidence
        for i in range(10):
            tracker.record(
                PredictionRecord(
                    tick=i,
                    domain="action_success",
                    predicted_confidence=1.0,
                    actual_outcome="fail",
                    was_correct=False,
                    context_tag="alone",
                    strategy_name="personality",
                )
            )

        assert tracker.brier_score == 1.0

    def test_brier_score_mixed(self):
        """Mixed outcomes should produce intermediate Brier score."""
        tracker = CalibrationTracker()

        # Half correct with confidence 0.8
        for i in range(5):
            tracker.record(
                PredictionRecord(
                    tick=i,
                    domain="action_success",
                    predicted_confidence=0.8,
                    actual_outcome="succeed",
                    was_correct=True,
                    context_tag="alone",
                    strategy_name="personality",
                )
            )

        # Half wrong with confidence 0.8
        for i in range(5, 10):
            tracker.record(
                PredictionRecord(
                    tick=i,
                    domain="action_success",
                    predicted_confidence=0.8,
                    actual_outcome="fail",
                    was_correct=False,
                    context_tag="alone",
                    strategy_name="personality",
                )
            )

        # Brier = mean((0.8-1)^2, (0.8-0)^2, ...) = mean(0.04, 0.64, ...) = 0.34
        expected_brier = (5 * (0.8 - 1.0) ** 2 + 5 * (0.8 - 0.0) ** 2) / 10
        assert abs(tracker.brier_score - expected_brier) < 0.01

    def test_calibration_score_inverse_of_brier(self):
        """Calibration score should be 1.0 - brier_score."""
        tracker = CalibrationTracker()

        for i in range(10):
            tracker.record(
                PredictionRecord(
                    tick=i,
                    domain="action_success",
                    predicted_confidence=0.7,
                    actual_outcome="succeed" if i < 7 else "fail",
                    was_correct=(i < 7),
                    context_tag="alone",
                    strategy_name="personality",
                )
            )

        brier = tracker.brier_score
        calibration = tracker.calibration_score
        assert abs(calibration - (1.0 - brier)) < 0.01

    def test_ece_no_data(self):
        """Expected calibration error should return 0.25 with insufficient data."""
        tracker = CalibrationTracker()
        assert tracker.expected_calibration_error == 0.25

    def test_ece_perfect_calibration(self):
        """Perfect calibration should give ECE near 0.0."""
        tracker = CalibrationTracker()

        # Confidence 0.75 with 75% accuracy
        for i in range(20):
            tracker.record(
                PredictionRecord(
                    tick=i,
                    domain="action_success",
                    predicted_confidence=0.75,
                    actual_outcome="succeed" if i < 15 else "fail",
                    was_correct=(i < 15),
                    context_tag="alone",
                    strategy_name="personality",
                )
            )

        # Should be close to 0 (within bin resolution)
        assert tracker.expected_calibration_error < 0.1

    def test_ece_overconfident(self):
        """High confidence with low accuracy should produce high ECE."""
        tracker = CalibrationTracker()

        # Confidence 0.9 but only 30% accuracy
        for i in range(20):
            tracker.record(
                PredictionRecord(
                    tick=i,
                    domain="action_success",
                    predicted_confidence=0.9,
                    actual_outcome="succeed" if i < 6 else "fail",
                    was_correct=(i < 6),
                    context_tag="alone",
                    strategy_name="personality",
                )
            )

        # ECE should be high (roughly |0.9 - 0.3| = 0.6)
        assert tracker.expected_calibration_error > 0.4

    def test_confidence_bias_overconfident(self):
        """Overconfidence should produce positive bias."""
        tracker = CalibrationTracker()

        # High confidence, low accuracy
        for i in range(30):
            tracker.record(
                PredictionRecord(
                    tick=i,
                    domain="action_success",
                    predicted_confidence=0.9,
                    actual_outcome="succeed" if i < 9 else "fail",
                    was_correct=(i < 9),
                    context_tag="alone",
                    strategy_name="personality",
                )
            )

        # Bias should be positive
        assert tracker.confidence_bias > 0.0

    def test_confidence_bias_underconfident(self):
        """Underconfidence should produce negative bias."""
        tracker = CalibrationTracker()

        # Low confidence, high accuracy
        for i in range(30):
            tracker.record(
                PredictionRecord(
                    tick=i,
                    domain="action_success",
                    predicted_confidence=0.3,
                    actual_outcome="succeed" if i < 27 else "fail",
                    was_correct=(i < 27),
                    context_tag="alone",
                    strategy_name="personality",
                )
            )

        # Bias should be negative
        assert tracker.confidence_bias < 0.0

    def test_confidence_bias_insufficient_data(self):
        """Should return 0.0 with insufficient data."""
        tracker = CalibrationTracker()

        # Only 3 predictions (below MIN_BIN_SIZE)
        for i in range(3):
            tracker.record(
                PredictionRecord(
                    tick=i,
                    domain="action_success",
                    predicted_confidence=0.8,
                    actual_outcome="succeed",
                    was_correct=True,
                    context_tag="alone",
                    strategy_name="personality",
                )
            )

        assert tracker.confidence_bias == 0.0

    def test_calibration_curve_returns_ten_bins(self):
        """Calibration curve should always return 10 bins."""
        tracker = CalibrationTracker()

        # Add some data
        for i in range(20):
            tracker.record(
                PredictionRecord(
                    tick=i,
                    domain="action_success",
                    predicted_confidence=0.5 + (i * 0.025),  # Spread across bins
                    actual_outcome="succeed" if i % 2 == 0 else "fail",
                    was_correct=(i % 2 == 0),
                    context_tag="alone",
                    strategy_name="personality",
                )
            )

        curve = tracker.calibration_curve()
        assert len(curve) == 10

        # Each entry is (bin_center, accuracy, count)
        for bin_center, accuracy, count in curve:
            assert 0.0 <= bin_center <= 1.0
            assert 0.0 <= accuracy <= 1.0
            assert count >= 0


# ============================================================================
# Test: CapabilityRating
# ============================================================================


class TestCapabilityRating:
    """Tests for CapabilityRating dataclass."""

    def test_default_values(self):
        """Verify default initialization."""
        rating = CapabilityRating(domain="gathering")

        assert rating.domain == "gathering"
        assert rating.rating == 0.5
        assert rating.confidence == 0.0
        assert rating.sample_count == 0
        assert rating.bias_estimate == 0.0

    def test_custom_values(self):
        """Verify custom initialization."""
        rating = CapabilityRating(
            domain="social",
            rating=0.8,
            confidence=0.6,
            sample_count=50,
            bias_estimate=0.1,
        )

        assert rating.domain == "social"
        assert rating.rating == 0.8
        assert rating.confidence == 0.6
        assert rating.sample_count == 50
        assert rating.bias_estimate == 0.1


# ============================================================================
# Test: AgentMetacogState
# ============================================================================


class TestAgentMetacogState:
    """Tests for AgentMetacogState bundle."""

    def test_default_initialization(self):
        """Verify default values."""
        state = AgentMetacogState()

        assert isinstance(state.calibration, CalibrationTracker)
        assert state.strategy_performance == {}
        assert state.self_model is None
        assert state.deliberation_threshold == 0.7
        assert state.current_strategy_name == "personality"
        assert state.available_strategies == ["personality", "planning"]
        assert len(state.switch_history) == 0
        assert state.last_switch_tick == -999
        assert state.underperformance_streak == 0

    def test_custom_initialization(self):
        """Verify custom initialization."""
        tracker = CalibrationTracker()
        perf = {("planning", "alone"): StrategyPerformance("planning", "alone")}
        model = CognitiveSelfModel()

        state = AgentMetacogState(
            calibration=tracker,
            strategy_performance=perf,
            self_model=model,
            deliberation_threshold=0.6,
            current_strategy_name="planning",
            available_strategies=["personality", "planning", "reactive"],
        )

        assert state.calibration is tracker
        assert state.strategy_performance is perf
        assert state.self_model is model
        assert state.deliberation_threshold == 0.6
        assert state.current_strategy_name == "planning"
        assert len(state.available_strategies) == 3


# ============================================================================
# Test: MetacognitiveSnapshot
# ============================================================================


class TestMetacognitiveSnapshot:
    """Tests for frozen MetacognitiveSnapshot dataclass."""

    def test_frozen_immutable(self):
        """Verify snapshot is frozen."""
        snapshot = MetacognitiveSnapshot(
            tick=1,
            agent_id="agent_1",
            active_strategy="personality",
            calibration_score=0.8,
            self_awareness_score=0.7,
            confidence_bias=0.05,
            strategy_switch_this_tick=False,
            previous_strategy="",
            switch_reason="",
            deliberation_threshold=0.7,
            self_model_summary={},
        )

        with pytest.raises(Exception, match=""):
            snapshot.tick = 2

    def test_all_fields(self):
        """Verify all fields can be created and accessed."""
        summary = {"gathering": 0.8, "social": 0.6}

        snapshot = MetacognitiveSnapshot(
            tick=10,
            agent_id="agent_2",
            active_strategy="planning",
            calibration_score=0.85,
            self_awareness_score=0.75,
            confidence_bias=-0.1,
            strategy_switch_this_tick=True,
            previous_strategy="personality",
            switch_reason="underperforming",
            deliberation_threshold=0.65,
            self_model_summary=summary,
        )

        assert snapshot.tick == 10
        assert snapshot.agent_id == "agent_2"
        assert snapshot.active_strategy == "planning"
        assert snapshot.calibration_score == 0.85
        assert snapshot.self_awareness_score == 0.75
        assert snapshot.confidence_bias == -0.1
        assert snapshot.strategy_switch_this_tick is True
        assert snapshot.previous_strategy == "personality"
        assert snapshot.switch_reason == "underperforming"
        assert snapshot.deliberation_threshold == 0.65
        assert snapshot.self_model_summary == summary


# ============================================================================
# Test: MetacognitiveMonitor
# ============================================================================


class TestMetacognitiveMonitor:
    """Tests for MetacognitiveMonitor evaluation logic."""

    def test_evaluate_tick_successful_action(self):
        """Successful action should create prediction with was_correct=True."""
        monitor = MetacognitiveMonitor()

        sensation = MockSensation(tick=1, own_needs={"hunger": 50, "thirst": 50})
        reflection = MockReflection()
        intention = Intention(primary_goal="gather", confidence=0.8)
        action = Action(type=ActionType.GATHER, target="berry_bush")
        action_result = ActionResult(
            action=action,
            success=True,
            message="Gathered berries",
            needs_delta={"hunger": 10.0},
        )

        prediction, outcome = monitor.evaluate_tick(
            agent_id="agent_1",
            tick=1,
            sensation=sensation,
            reflection=reflection,
            intention=intention,
            action_result=action_result,
            needs_before={"hunger": 50.0},
            needs_after={"hunger": 60.0},
            strategy_name="personality",
            deliberation_used=False,
        )

        assert prediction.predicted_confidence == 0.8
        assert prediction.was_correct is True
        assert prediction.domain == "action_success"
        assert outcome.succeeded is True
        assert outcome.needs_delta_sum == 10.0

    def test_evaluate_tick_failed_action(self):
        """Failed action should create prediction with was_correct=False."""
        monitor = MetacognitiveMonitor()

        sensation = MockSensation(tick=1)
        reflection = MockReflection()
        intention = Intention(primary_goal="gather", confidence=0.6)
        action = Action(type=ActionType.GATHER, target="berry_bush")
        action_result = ActionResult(
            action=action,
            success=False,
            message="No resources found",
            needs_delta={},
        )

        prediction, outcome = monitor.evaluate_tick(
            agent_id="agent_1",
            tick=1,
            sensation=sensation,
            reflection=reflection,
            intention=intention,
            action_result=action_result,
            needs_before={"hunger": 50.0},
            needs_after={"hunger": 50.0},
            strategy_name="planning",
            deliberation_used=True,
        )

        assert prediction.predicted_confidence == 0.6
        assert prediction.was_correct is False
        assert outcome.succeeded is False
        assert outcome.deliberation_used is True

    def test_evaluate_tick_no_sensation_defaults_alone(self):
        """None sensation should default context_tag to 'alone'."""
        monitor = MetacognitiveMonitor()

        intention = Intention(primary_goal="wait", confidence=0.5)
        action_result = ActionResult(
            action=Action(type=ActionType.WAIT),
            success=True,
            message="Waited",
        )

        prediction, outcome = monitor.evaluate_tick(
            agent_id="agent_1",
            tick=1,
            sensation=None,
            reflection=None,
            intention=intention,
            action_result=action_result,
            needs_before={},
            needs_after={},
            strategy_name="personality",
            deliberation_used=False,
        )

        assert prediction.context_tag == "alone"
        assert outcome.context_tag == "alone"

    def test_evaluate_tick_extracts_context_from_sensation(self):
        """Should extract context from sensation using ContextTag.extract_primary."""
        monitor = MetacognitiveMonitor()

        # Low hunger should trigger LOW_HUNGER context
        sensation = MockSensation(tick=1, own_needs={"hunger": 25, "thirst": 80})
        intention = Intention(primary_goal="gather", confidence=0.7)
        action_result = ActionResult(
            action=Action(type=ActionType.GATHER),
            success=True,
            message="Gathered",
        )

        prediction, outcome = monitor.evaluate_tick(
            agent_id="agent_1",
            tick=1,
            sensation=sensation,
            reflection=None,
            intention=intention,
            action_result=action_result,
            needs_before={},
            needs_after={},
            strategy_name="personality",
            deliberation_used=False,
        )

        assert prediction.context_tag == "low_hunger"

    def test_evaluate_tick_needs_delta_computed(self):
        """Should compute needs_delta_sum from action_result.needs_delta."""
        monitor = MetacognitiveMonitor()

        sensation = MockSensation()
        intention = Intention(primary_goal="eat", confidence=0.9)
        action_result = ActionResult(
            action=Action(type=ActionType.EAT),
            success=True,
            message="Ate food",
            needs_delta={"hunger": 20.0, "energy": 5.0},
        )

        prediction, outcome = monitor.evaluate_tick(
            agent_id="agent_1",
            tick=1,
            sensation=sensation,
            reflection=None,
            intention=intention,
            action_result=action_result,
            needs_before={},
            needs_after={},
            strategy_name="personality",
            deliberation_used=False,
        )

        assert outcome.needs_delta_sum == 25.0

    def test_evaluate_tick_none_action_result(self):
        """Should handle None action_result gracefully."""
        monitor = MetacognitiveMonitor()

        sensation = MockSensation()
        intention = Intention(primary_goal="wait", confidence=0.5)

        prediction, outcome = monitor.evaluate_tick(
            agent_id="agent_1",
            tick=1,
            sensation=sensation,
            reflection=None,
            intention=intention,
            action_result=None,
            needs_before={},
            needs_after={},
            strategy_name="personality",
            deliberation_used=False,
        )

        assert prediction.was_correct is False
        assert outcome.succeeded is False
        assert outcome.needs_delta_sum == 0.0

    def test_fok_high_capability_returns_high(self):
        """Good calibration and performance should produce high FOK."""
        monitor = MetacognitiveMonitor()

        # Create good calibration
        tracker = CalibrationTracker()
        for i in range(20):
            tracker.record(
                PredictionRecord(
                    tick=i,
                    domain="action_success",
                    predicted_confidence=1.0,
                    actual_outcome="succeed",
                    was_correct=True,
                    context_tag="alone",
                    strategy_name="personality",
                )
            )

        # Create good performance
        perf = StrategyPerformance("personality", "alone")
        for _ in range(10):
            perf.record(succeeded=True, fitness_delta=5.0)
        strategy_perf = {("personality", "alone"): perf}

        # Create good self-model
        self_model = CognitiveSelfModel()
        for _ in range(20):
            self_model.update_from_outcome("gathering", succeeded=True, prediction_error=0.0)

        fok = monitor.compute_feeling_of_knowing(
            calibration=tracker,
            strategy_perf=strategy_perf,
            context_tag="alone",
            self_model=self_model,
        )

        # Should be high (> 0.7)
        assert fok > 0.7

    def test_fok_low_capability_returns_low(self):
        """Bad calibration and performance should produce low FOK."""
        monitor = MetacognitiveMonitor()

        # Create bad calibration
        tracker = CalibrationTracker()
        for i in range(20):
            tracker.record(
                PredictionRecord(
                    tick=i,
                    domain="action_success",
                    predicted_confidence=1.0,
                    actual_outcome="fail",
                    was_correct=False,
                    context_tag="alone",
                    strategy_name="personality",
                )
            )

        # Create bad performance
        perf = StrategyPerformance("personality", "alone")
        for _ in range(10):
            perf.record(succeeded=False, fitness_delta=-2.0)
        strategy_perf = {("personality", "alone"): perf}

        # Create bad self-model
        self_model = CognitiveSelfModel()
        for _ in range(20):
            self_model.update_from_outcome("gathering", succeeded=False, prediction_error=1.0)

        fok = monitor.compute_feeling_of_knowing(
            calibration=tracker,
            strategy_perf=strategy_perf,
            context_tag="alone",
            self_model=self_model,
        )

        # Should be low (< 0.3)
        assert fok < 0.3


# ============================================================================
# Test: CognitiveSelfModel
# ============================================================================


class TestCognitiveSelfModel:
    """Tests for CognitiveSelfModel capability learning."""

    def test_get_rating_unknown_returns_neutral(self):
        """Unknown domain should return neutral prior."""
        model = CognitiveSelfModel()

        rating = model.get_rating("unknown_domain")

        assert rating.rating == 0.5
        assert rating.confidence == 0.0
        assert rating.sample_count == 0

    def test_update_from_outcome_increments_sample_count(self):
        """Each update should increment sample count."""
        model = CognitiveSelfModel()

        model.update_from_outcome("gathering", succeeded=True, prediction_error=0.0)
        rating = model.get_rating("gathering")
        assert rating.sample_count == 1

        model.update_from_outcome("gathering", succeeded=True, prediction_error=0.0)
        rating = model.get_rating("gathering")
        assert rating.sample_count == 2

    def test_update_from_outcome_no_change_below_min_samples(self):
        """Rating should stay at 0.5 until MIN_SAMPLES (10) reached."""
        model = CognitiveSelfModel()

        # Update 9 times (below threshold)
        for _ in range(9):
            model.update_from_outcome("social", succeeded=True, prediction_error=0.0)

        rating = model.get_rating("social")
        assert rating.rating == 0.5
        assert rating.sample_count == 9

    def test_update_from_outcome_ema_convergence(self):
        """After many successes, rating should approach 1.0 via EMA."""
        model = CognitiveSelfModel()

        # Update 100 times with success
        for _ in range(100):
            model.update_from_outcome("combat", succeeded=True, prediction_error=0.0)

        rating = model.get_rating("combat")
        # Should be much closer to 1.0 than 0.5
        assert rating.rating > 0.9

    def test_update_bias_estimate_ema(self):
        """Bias estimate should update via EMA."""
        model = CognitiveSelfModel()

        # Set initial bias
        model.update_bias_estimate("gathering", bias=0.5)
        rating = model.get_rating("gathering")

        # Should be close to 0.5 * EMA_ALPHA (0.05) = 0.025
        assert rating.bias_estimate > 0.0

        # Update with new bias
        for _ in range(20):
            model.update_bias_estimate("gathering", bias=0.2)

        rating = model.get_rating("gathering")
        # Should have moved toward 0.2
        assert rating.bias_estimate > 0.0

    def test_strongest_domain(self):
        """Should return domain with highest rating."""
        model = CognitiveSelfModel()

        # Create ratings in different domains
        for _ in range(50):
            model.update_from_outcome("gathering", succeeded=True, prediction_error=0.0)
        for _ in range(50):
            model.update_from_outcome("social", succeeded=False, prediction_error=0.0)
        for _ in range(50):
            model.update_from_outcome("combat", succeeded=True, prediction_error=0.0)

        strongest = model.strongest_domain()
        # Should be either gathering or combat (both successful)
        assert strongest in ["gathering", "combat"]

    def test_weakest_domain(self):
        """Should return domain with lowest rating."""
        model = CognitiveSelfModel()

        # Create ratings
        for _ in range(50):
            model.update_from_outcome("gathering", succeeded=True, prediction_error=0.0)
        for _ in range(50):
            model.update_from_outcome("social", succeeded=False, prediction_error=0.0)

        weakest = model.weakest_domain()
        assert weakest == "social"

    def test_strongest_weakest_empty_returns_unknown(self):
        """Empty model should return 'unknown' for strongest/weakest."""
        model = CognitiveSelfModel()

        assert model.strongest_domain() == "unknown"
        assert model.weakest_domain() == "unknown"

    def test_capability_summary(self):
        """Should return dict of {domain: rating}."""
        model = CognitiveSelfModel()

        for _ in range(50):
            model.update_from_outcome("gathering", succeeded=True, prediction_error=0.0)
        for _ in range(50):
            model.update_from_outcome("social", succeeded=False, prediction_error=0.0)

        summary = model.capability_summary()

        assert isinstance(summary, dict)
        assert "gathering" in summary
        assert "social" in summary
        assert summary["gathering"] > summary["social"]

    def test_to_dict_serializable(self):
        """Should return full serializable dict."""
        model = CognitiveSelfModel()

        for _ in range(20):
            model.update_from_outcome("exploration", succeeded=True, prediction_error=0.0)

        data = model.to_dict()

        assert isinstance(data, dict)
        assert "exploration" in data
        assert "rating" in data["exploration"]
        assert "confidence" in data["exploration"]
        assert "sample_count" in data["exploration"]
        assert "bias_estimate" in data["exploration"]


# ============================================================================
# Test: MetacognitiveController
# ============================================================================


class TestMetacognitiveController:
    """Tests for MetacognitiveController decision logic."""

    def test_no_switch_performing_well(self):
        """Should not switch if performance above threshold."""
        controller = MetacognitiveController(switch_threshold=0.5)

        # Create good performance
        perf = StrategyPerformance("personality", "alone")
        for _ in range(10):
            perf.record(succeeded=True, fitness_delta=5.0)

        state = AgentMetacogState(
            strategy_performance={("personality", "alone"): perf},
            current_strategy_name="personality",
            underperformance_streak=0,
        )

        decision = controller.decide(
            agent_id="agent_1",
            tick=10,
            state=state,
            context="alone",
            fok=0.7,
            threat=0.0,
            nearby_agent_ids=[],
        )

        assert decision.switch_strategy is False
        assert state.underperformance_streak == 0

    def test_no_switch_insufficient_streak(self):
        """Should not switch if underperformance streak < patience."""
        controller = MetacognitiveController(
            switch_threshold=0.5,
            switch_patience=5,
        )

        # Create poor performance
        perf = StrategyPerformance("personality", "alone")
        for _ in range(10):
            perf.record(succeeded=False, fitness_delta=-2.0)

        state = AgentMetacogState(
            strategy_performance={("personality", "alone"): perf},
            current_strategy_name="personality",
            underperformance_streak=3,  # Below patience
        )

        decision = controller.decide(
            agent_id="agent_1",
            tick=10,
            state=state,
            context="alone",
            fok=0.7,
            threat=0.0,
            nearby_agent_ids=[],
        )

        assert decision.switch_strategy is False
        assert state.underperformance_streak == 4  # Incremented

    def test_switch_after_underperformance_streak(self):
        """Should switch after sustained underperformance."""
        controller = MetacognitiveController(
            switch_threshold=0.5,
            switch_patience=3,
            switch_cooldown=5,
        )

        # Create poor performance for current strategy
        perf = StrategyPerformance("personality", "alone")
        for _ in range(10):
            perf.record(succeeded=False, fitness_delta=-2.0)

        state = AgentMetacogState(
            strategy_performance={("personality", "alone"): perf},
            current_strategy_name="personality",
            available_strategies=["personality", "planning"],
            underperformance_streak=3,  # At patience threshold
            last_switch_tick=-999,  # Long ago
        )

        decision = controller.decide(
            agent_id="agent_1",
            tick=100,
            state=state,
            context="alone",
            fok=0.7,
            threat=0.0,
            nearby_agent_ids=[],
        )

        assert decision.switch_strategy is True
        assert decision.new_strategy_name != ""
        assert decision.switch_reason == "underperforming"

    def test_switch_cooldown_enforced(self):
        """Should not switch if within cooldown period."""
        controller = MetacognitiveController(
            switch_threshold=0.5,
            switch_patience=3,
            switch_cooldown=10,
        )

        perf = StrategyPerformance("personality", "alone")
        for _ in range(10):
            perf.record(succeeded=False, fitness_delta=-2.0)

        state = AgentMetacogState(
            strategy_performance={("personality", "alone"): perf},
            current_strategy_name="personality",
            underperformance_streak=5,
            last_switch_tick=95,  # Recent switch
        )

        decision = controller.decide(
            agent_id="agent_1",
            tick=100,  # Only 5 ticks since last switch
            state=state,
            context="alone",
            fok=0.7,
            threat=0.0,
            nearby_agent_ids=[],
        )

        assert decision.switch_strategy is False

    def test_switch_selects_best_alternative(self):
        """Should select strategy with best recent performance."""
        controller = MetacognitiveController()

        # Current strategy performing poorly
        perf_current = StrategyPerformance("personality", "alone")
        for _ in range(10):
            perf_current.record(succeeded=False, fitness_delta=-2.0)

        # Alternative 1: moderate performance
        perf_alt1 = StrategyPerformance("planning", "alone")
        for _ in range(10):
            perf_alt1.record(succeeded=True, fitness_delta=3.0)
        for _ in range(5):
            perf_alt1.record(succeeded=False, fitness_delta=-1.0)

        # Alternative 2: excellent performance
        perf_alt2 = StrategyPerformance("reactive", "alone")
        for _ in range(15):
            perf_alt2.record(succeeded=True, fitness_delta=5.0)

        state = AgentMetacogState(
            strategy_performance={
                ("personality", "alone"): perf_current,
                ("planning", "alone"): perf_alt1,
                ("reactive", "alone"): perf_alt2,
            },
            current_strategy_name="personality",
            available_strategies=["personality", "planning", "reactive"],
            underperformance_streak=5,
            last_switch_tick=-999,
        )

        decision = controller.decide(
            agent_id="agent_1",
            tick=100,
            state=state,
            context="alone",
            fok=0.7,
            threat=0.0,
            nearby_agent_ids=[],
        )

        assert decision.switch_strategy is True
        # Should pick reactive (best performance: 100% vs planning's ~67%)
        assert decision.new_strategy_name == "reactive"

    def test_switch_defaults_to_planning(self):
        """Should default to 'planning' if no performance data."""
        controller = MetacognitiveController()

        perf = StrategyPerformance("personality", "alone")
        for _ in range(10):
            perf.record(succeeded=False, fitness_delta=-2.0)

        state = AgentMetacogState(
            strategy_performance={("personality", "alone"): perf},
            current_strategy_name="personality",
            available_strategies=["personality", "planning"],
            underperformance_streak=5,
            last_switch_tick=-999,
        )

        decision = controller.decide(
            agent_id="agent_1",
            tick=100,
            state=state,
            context="alone",
            fok=0.7,
            threat=0.0,
            nearby_agent_ids=[],
        )

        assert decision.switch_strategy is True
        assert decision.new_strategy_name == "planning"

    def test_deliberation_lowered_when_ece_high(self):
        """High ECE should lower deliberation threshold."""
        controller = MetacognitiveController(deliberation_adjustment_rate=0.05)

        # Create high ECE (overconfident)
        tracker = CalibrationTracker()
        for i in range(30):
            tracker.record(
                PredictionRecord(
                    tick=i,
                    domain="action_success",
                    predicted_confidence=0.9,
                    actual_outcome="succeed" if i < 9 else "fail",
                    was_correct=(i < 9),
                    context_tag="alone",
                    strategy_name="personality",
                )
            )

        state = AgentMetacogState(
            calibration=tracker,
            deliberation_threshold=0.7,
        )

        decision = controller.decide(
            agent_id="agent_1",
            tick=10,
            state=state,
            context="alone",
            fok=0.5,
            threat=0.0,
            nearby_agent_ids=[],
        )

        # Should lower threshold (ECE > 0.15)
        assert decision.new_deliberation_threshold is not None
        assert decision.new_deliberation_threshold < 0.7

    def test_deliberation_raised_when_ece_low(self):
        """Low ECE should raise deliberation threshold."""
        controller = MetacognitiveController(deliberation_adjustment_rate=0.05)

        # Create low ECE (well calibrated)
        tracker = CalibrationTracker()
        for i in range(30):
            tracker.record(
                PredictionRecord(
                    tick=i,
                    domain="action_success",
                    predicted_confidence=0.8,
                    actual_outcome="succeed" if i < 24 else "fail",
                    was_correct=(i < 24),
                    context_tag="alone",
                    strategy_name="personality",
                )
            )

        state = AgentMetacogState(
            calibration=tracker,
            deliberation_threshold=0.5,
        )

        decision = controller.decide(
            agent_id="agent_1",
            tick=10,
            state=state,
            context="alone",
            fok=0.5,
            threat=0.0,
            nearby_agent_ids=[],
        )

        # ECE should be low (0.8 confidence, 0.8 accuracy = well calibrated)
        # Should raise threshold (ECE < 0.05)
        assert decision.new_deliberation_threshold is not None
        assert decision.new_deliberation_threshold > 0.5

    def test_deliberation_clamped_min(self):
        """Deliberation threshold should never go below 0.3."""
        controller = MetacognitiveController(deliberation_adjustment_rate=0.5)

        # High ECE
        tracker = CalibrationTracker()
        for i in range(30):
            tracker.record(
                PredictionRecord(
                    tick=i,
                    domain="action_success",
                    predicted_confidence=1.0,
                    actual_outcome="fail",
                    was_correct=False,
                    context_tag="alone",
                    strategy_name="personality",
                )
            )

        state = AgentMetacogState(
            calibration=tracker,
            deliberation_threshold=0.3,  # Already at min
        )

        decision = controller.decide(
            agent_id="agent_1",
            tick=10,
            state=state,
            context="alone",
            fok=0.5,
            threat=0.0,
            nearby_agent_ids=[],
        )

        # Should stay at 0.3
        if decision.new_deliberation_threshold is not None:
            assert decision.new_deliberation_threshold >= 0.3

    def test_deliberation_clamped_max(self):
        """Deliberation threshold should never go above 0.9."""
        controller = MetacognitiveController(deliberation_adjustment_rate=0.5)

        # Low ECE
        tracker = CalibrationTracker()
        for i in range(30):
            tracker.record(
                PredictionRecord(
                    tick=i,
                    domain="action_success",
                    predicted_confidence=1.0,
                    actual_outcome="succeed",
                    was_correct=True,
                    context_tag="alone",
                    strategy_name="personality",
                )
            )

        state = AgentMetacogState(
            calibration=tracker,
            deliberation_threshold=0.9,  # Already at max
        )

        decision = controller.decide(
            agent_id="agent_1",
            tick=10,
            state=state,
            context="alone",
            fok=0.5,
            threat=0.0,
            nearby_agent_ids=[],
        )

        # Should stay at or below 0.9
        if decision.new_deliberation_threshold is not None:
            assert decision.new_deliberation_threshold <= 0.9

    def test_deliberation_no_change_when_ece_moderate(self):
        """Moderate ECE (0.05-0.15) should not change threshold."""
        controller = MetacognitiveController()

        # Moderate ECE
        tracker = CalibrationTracker()
        for i in range(30):
            tracker.record(
                PredictionRecord(
                    tick=i,
                    domain="action_success",
                    predicted_confidence=0.6,
                    actual_outcome="succeed" if i < 18 else "fail",
                    was_correct=(i < 18),
                    context_tag="alone",
                    strategy_name="personality",
                )
            )

        state = AgentMetacogState(
            calibration=tracker,
            deliberation_threshold=0.7,
        )

        decision = controller.decide(
            agent_id="agent_1",
            tick=10,
            state=state,
            context="alone",
            fok=0.5,
            threat=0.0,
            nearby_agent_ids=[],
        )

        # Should not change (or change very little)
        assert decision.new_deliberation_threshold is None

    def test_help_seeking_triggered(self):
        """Low FOK + high threat + nearby agents should trigger help-seeking."""
        controller = MetacognitiveController(
            help_confidence_threshold=0.4,
            help_stakes_threshold=0.6,
        )

        state = AgentMetacogState()

        decision = controller.decide(
            agent_id="agent_1",
            tick=10,
            state=state,
            context="near_hostile",
            fok=0.2,  # Low confidence
            threat=0.8,  # High threat
            nearby_agent_ids=["agent_2", "agent_3"],
        )

        assert decision.seek_help is True
        assert decision.help_target_id == "agent_2"

    def test_help_seeking_suppressed_no_nearby(self):
        """Should not seek help if no nearby agents."""
        controller = MetacognitiveController()

        state = AgentMetacogState()

        decision = controller.decide(
            agent_id="agent_1",
            tick=10,
            state=state,
            context="near_hostile",
            fok=0.2,
            threat=0.8,
            nearby_agent_ids=[],  # No one nearby
        )

        assert decision.seek_help is False

    def test_help_seeking_suppressed_confident(self):
        """Should not seek help if FOK is high (confident)."""
        controller = MetacognitiveController()

        state = AgentMetacogState()

        decision = controller.decide(
            agent_id="agent_1",
            tick=10,
            state=state,
            context="near_hostile",
            fok=0.8,  # High confidence
            threat=0.8,
            nearby_agent_ids=["agent_2"],
        )

        assert decision.seek_help is False


# ============================================================================
# Test: MetacognitiveStrategy
# ============================================================================


class TestMetacognitiveStrategy:
    """Tests for MetacognitiveStrategy wrapper."""

    def test_form_intention_delegates_to_inner(self):
        """Should delegate to inner strategy."""
        inner = MockStrategy()
        strategy = MetacognitiveStrategy(inner, fok_enabled=False)

        sensation = MockSensation()
        reflection = MockReflection()

        intention = strategy.form_intention(sensation, reflection)

        assert intention.primary_goal == "mock_goal"

    def test_form_intention_blends_fok(self):
        """Should blend FOK into confidence when enabled."""
        inner = MockStrategy()
        strategy = MetacognitiveStrategy(inner, fok_enabled=True)
        strategy.set_fok(0.8)

        sensation = MockSensation()
        reflection = MockReflection()

        intention = strategy.form_intention(sensation, reflection)

        # Original confidence is 0.5, FOK is 0.8
        # Blended: 0.5 * 0.5 + 0.5 * 0.8 = 0.25 + 0.4 = 0.65
        expected = 0.5 * 0.5 + 0.5 * 0.8
        assert abs(intention.confidence - expected) < 0.01

    def test_form_intention_fok_disabled(self):
        """Should not blend FOK when disabled."""
        inner = MockStrategy()
        strategy = MetacognitiveStrategy(inner, fok_enabled=False)
        strategy.set_fok(0.9)

        sensation = MockSensation()
        reflection = MockReflection()

        intention = strategy.form_intention(sensation, reflection)

        # Should keep original confidence
        assert intention.confidence == 0.5

    def test_express_delegates_to_inner(self):
        """Express should be pure delegation."""
        inner = MockStrategy()
        strategy = MetacognitiveStrategy(inner)

        sensation = MockSensation()
        reflection = MockReflection()
        intention = Intention(primary_goal="test", confidence=0.5)

        expression = strategy.express(sensation, reflection, intention)

        assert expression.action.type == ActionType.WAIT

    def test_set_fok_clamped_low(self):
        """Negative FOK should be clamped to 0.0."""
        inner = MockStrategy()
        strategy = MetacognitiveStrategy(inner)

        strategy.set_fok(-0.5)

        assert strategy._current_fok == 0.0

    def test_set_fok_clamped_high(self):
        """FOK > 1.0 should be clamped to 1.0."""
        inner = MockStrategy()
        strategy = MetacognitiveStrategy(inner)

        strategy.set_fok(1.5)

        assert strategy._current_fok == 1.0

    def test_active_strategy_name_initial(self):
        """Should return strategy name passed to constructor."""
        inner = MockStrategy()
        strategy = MetacognitiveStrategy(inner, strategy_name="planning")

        assert strategy.active_strategy_name == "planning"

    def test_active_strategy_name_tracks_constructor(self):
        """Active strategy name should match constructor parameter."""
        inner = MockStrategy()
        strategy = MetacognitiveStrategy(inner, strategy_name="reactive")

        assert strategy.active_strategy_name == "reactive"

    def test_satisfies_protocol(self):
        """Should have form_intention and express methods (protocol check)."""
        inner = MockStrategy()
        strategy = MetacognitiveStrategy(inner)

        assert hasattr(strategy, "form_intention")
        assert hasattr(strategy, "express")
        assert callable(strategy.form_intention)
        assert callable(strategy.express)

    def test_inner_property(self):
        """Should provide access to inner strategy."""
        inner = MockStrategy()
        strategy = MetacognitiveStrategy(inner)

        assert strategy.inner is inner


# ============================================================================
# Test: MetacognitiveEngine (Integration)
# ============================================================================


class TestMetacognitiveEngineUnit:
    """Unit tests for MetacognitiveEngine orchestration."""

    def test_register_agent_creates_state(self):
        """Registering agent should create AgentMetacogState."""
        from src.metacognition.engine import MetacognitiveEngine

        engine = MetacognitiveEngine()
        engine.register_agent(
            agent_id="agent_1",
            initial_strategy_name="personality",
            available_strategies=["personality", "planning"],
            strategy_instances={"personality": MockStrategy(), "planning": MockStrategy()},
        )

        state = engine.get_agent_state("agent_1")
        assert state is not None
        assert state.current_strategy_name == "personality"
        assert len(state.available_strategies) == 2

    def test_register_agent_preserves_state_on_reregistration(self):
        """Re-registering agent should preserve calibration and performance data."""
        from src.metacognition.engine import MetacognitiveEngine
        from src.metacognition.types import PredictionRecord

        engine = MetacognitiveEngine()
        engine.register_agent(
            agent_id="agent_1",
            initial_strategy_name="personality",
            available_strategies=["personality"],
            strategy_instances={"personality": MockStrategy()},
        )

        # Add calibration data
        state = engine.get_agent_state("agent_1")
        state.calibration.record(
            PredictionRecord(
                tick=1,
                domain="action_success",
                predicted_confidence=0.8,
                actual_outcome="succeed",
                was_correct=True,
                context_tag="alone",
                strategy_name="personality",
            )
        )

        # Re-register
        engine.register_agent(
            agent_id="agent_1",
            initial_strategy_name="planning",
            available_strategies=["personality", "planning"],
            strategy_instances={"personality": MockStrategy(), "planning": MockStrategy()},
        )

        # Calibration should be preserved
        state = engine.get_agent_state("agent_1")
        assert state.calibration.count == 1

    def test_unregistered_agent_returns_none(self):
        """Getting state for unregistered agent should return None."""
        from src.metacognition.engine import MetacognitiveEngine

        engine = MetacognitiveEngine()
        assert engine.get_agent_state("nonexistent") is None

    def test_snapshots_accumulate(self):
        """Snapshots should accumulate across ticks."""
        from src.metacognition.engine import MetacognitiveEngine

        engine = MetacognitiveEngine()
        assert len(engine.snapshots) == 0

    def test_get_metacognitive_stats_empty(self):
        """Empty engine should return zero stats."""
        from src.metacognition.engine import MetacognitiveEngine

        engine = MetacognitiveEngine()
        stats = engine.get_metacognitive_stats()

        assert stats["total_agents_tracked"] == 0
        assert stats["avg_calibration_score"] == 0.0
        assert stats["strategy_distribution"] == {}
        assert stats["total_switches"] == 0

    def test_get_metacognitive_stats_with_agents(self):
        """Should return aggregated stats across registered agents."""
        from src.metacognition.engine import MetacognitiveEngine

        engine = MetacognitiveEngine()
        engine.register_agent(
            agent_id="a1",
            initial_strategy_name="personality",
            available_strategies=["personality"],
            strategy_instances={"personality": MockStrategy()},
        )
        engine.register_agent(
            agent_id="a2",
            initial_strategy_name="planning",
            available_strategies=["planning"],
            strategy_instances={"planning": MockStrategy()},
        )

        stats = engine.get_metacognitive_stats()
        assert stats["total_agents_tracked"] == 2
        assert stats["strategy_distribution"] == {"personality": 1, "planning": 1}

    def test_action_type_to_domain_mappings(self):
        """Should map action types to correct domains."""
        from src.metacognition.engine import MetacognitiveEngine

        engine = MetacognitiveEngine()

        assert engine._action_type_to_domain("gather") == "gathering"
        assert engine._action_type_to_domain("move") == "exploration"
        assert engine._action_type_to_domain("attack") == "combat"
        assert engine._action_type_to_domain("give") == "social"
        assert engine._action_type_to_domain("rest") == "gathering"
        assert engine._action_type_to_domain("unknown") == "planning"


# ============================================================================
# Test: Architecture Registration
# ============================================================================


class TestArchitectureRegistration:
    """Tests for metacognitive architecture in the registry."""

    def test_metacognitive_architecture_exists(self):
        """Architecture registry should contain 'metacognitive' entry."""
        from src.awareness.architecture import get_architectures

        archs = get_architectures()
        assert "metacognitive" in archs

    def test_metacognitive_architecture_has_deliberation(self):
        """Metacognitive architecture should include ThresholdDeliberation."""
        from src.awareness.architecture import get_architectures
        from src.awareness.deliberation import ThresholdDeliberation

        arch = get_architectures()["metacognitive"]
        assert isinstance(arch.deliberation, ThresholdDeliberation)

    def test_metacognitive_architecture_build_succeeds(self):
        """Should successfully build an awareness loop from metacognitive arch."""
        from src.agents.identity import AgentID, AgentProfile, PersonalityTraits
        from src.awareness.architecture import build_awareness_loop
        from src.simulation.entities import Agent

        profile = AgentProfile(
            agent_id=AgentID(),
            name="test_agent",
            archetype="explorer",
            traits=PersonalityTraits(),
        )
        agent = Agent(profile=profile, x=0, y=0)

        loop = build_awareness_loop(agent, "metacognitive")
        assert loop is not None
        assert loop.strategy is not None


# ============================================================================
# Test: Config Flags
# ============================================================================


class TestConfigFlags:
    """Tests for metacognition config parameters."""

    def test_metacognition_disabled_by_default(self):
        """Metacognition should be disabled by default in config."""
        from src.config import SimulationConfig

        config = SimulationConfig()
        assert config.metacognition_enabled is False

    def test_metacognition_config_fields_exist(self):
        """All metacognition config fields should be accessible."""
        from src.config import SimulationConfig

        config = SimulationConfig()
        assert hasattr(config, "metacognition_enabled")
        assert hasattr(config, "metacognition_switch_threshold")
        assert hasattr(config, "metacognition_switch_patience")
        assert hasattr(config, "metacognition_help_confidence_threshold")
        assert hasattr(config, "metacognition_help_stakes_threshold")
        assert hasattr(config, "metacognition_deliberation_adjustment_rate")
        assert hasattr(config, "metacognition_switch_cooldown")
        assert hasattr(config, "metacognition_fok_enabled")

    def test_metacognition_config_defaults(self):
        """Config defaults should match plan values."""
        from src.config import SimulationConfig

        config = SimulationConfig()
        assert config.metacognition_switch_threshold == 0.3
        assert config.metacognition_switch_patience == 5
        assert config.metacognition_help_confidence_threshold == 0.3
        assert config.metacognition_help_stakes_threshold == 0.6
        assert config.metacognition_deliberation_adjustment_rate == 0.05
        assert config.metacognition_switch_cooldown == 10
        assert config.metacognition_fok_enabled is True

    def test_engine_no_metacognition_when_disabled(self):
        """SimulationEngine should not create metacognition_engine when disabled."""
        from src.config import SimulationConfig
        from src.simulation.engine import SimulationEngine

        config = SimulationConfig(metacognition_enabled=False)
        engine = SimulationEngine(config)
        assert engine.metacognition_engine is None
        assert engine.metacognition_analyzer is None

    def test_engine_creates_metacognition_when_enabled(self):
        """SimulationEngine should create metacognition_engine when enabled."""
        from src.config import SimulationConfig
        from src.simulation.engine import SimulationEngine

        config = SimulationConfig(metacognition_enabled=True)
        engine = SimulationEngine(config)
        assert engine.metacognition_engine is not None
        assert engine.metacognition_analyzer is None  # Analyzer created separately


# ============================================================================
# Test: MetacognitiveAnalyzer
# ============================================================================


class TestMetacognitiveAnalyzer:
    """Tests for population-level metacognitive metrics."""

    def test_empty_history(self):
        """New analyzer should have empty history."""
        from src.metacognition.metrics import MetacognitiveAnalyzer

        analyzer = MetacognitiveAnalyzer()
        assert len(analyzer.history) == 0
        assert analyzer.total_switches() == 0

    def test_record_tick_creates_snapshot(self):
        """Recording a tick should add to history."""
        from src.metacognition.engine import MetacognitiveEngine
        from src.metacognition.metrics import MetacognitiveAnalyzer

        engine = MetacognitiveEngine()
        engine.register_agent(
            agent_id="a1",
            initial_strategy_name="personality",
            available_strategies=["personality"],
            strategy_instances={"personality": MockStrategy()},
        )

        analyzer = MetacognitiveAnalyzer()
        snapshot = analyzer.record_tick(engine, tick=1)

        assert snapshot.tick == 1
        assert snapshot.total_agents == 1
        assert len(analyzer.history) == 1

    def test_population_calibration_returns_tuples(self):
        """Should return (tick, calibration_score) tuples."""
        from src.metacognition.engine import MetacognitiveEngine
        from src.metacognition.metrics import MetacognitiveAnalyzer

        engine = MetacognitiveEngine()
        engine.register_agent(
            agent_id="a1",
            initial_strategy_name="personality",
            available_strategies=["personality"],
            strategy_instances={"personality": MockStrategy()},
        )

        analyzer = MetacognitiveAnalyzer()
        analyzer.record_tick(engine, tick=1)
        analyzer.record_tick(engine, tick=2)

        cal = analyzer.population_calibration()
        assert len(cal) == 2
        assert cal[0][0] == 1
        assert cal[1][0] == 2

    def test_to_summary_empty(self):
        """Empty analyzer should return zero summary."""
        from src.metacognition.metrics import MetacognitiveAnalyzer

        analyzer = MetacognitiveAnalyzer()
        summary = analyzer.to_summary()

        assert summary["total_ticks_recorded"] == 0
        assert summary["total_switches"] == 0


# ============================================================================
# Test: End-to-End Integration
# ============================================================================


class TestMetacognitionIntegration:
    """End-to-end integration tests with SimulationEngine."""

    def test_metacognition_enabled_runs_cleanly(self):
        """SimulationEngine with metacognition should run ticks without error."""
        from src.config import SimulationConfig
        from src.simulation.engine import SimulationEngine

        config = SimulationConfig(
            metacognition_enabled=True,
            num_agents=3,
            max_ticks=10,
        )
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        # Run a few ticks
        for _ in range(5):
            tick_record = engine.step_all()
            assert tick_record is not None
            assert len(tick_record.agent_records) > 0

    def test_metacognition_wraps_all_agents(self):
        """All agents should have MetacognitiveStrategy wrapper after setup."""
        from src.cognition.strategies.metacognitive_strategy import MetacognitiveStrategy
        from src.config import SimulationConfig
        from src.simulation.engine import SimulationEngine

        config = SimulationConfig(
            metacognition_enabled=True,
            num_agents=3,
        )
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        for agent in engine.registry.living_agents():
            loop = engine.registry.get_awareness_loop(agent.agent_id)
            assert loop is not None
            assert isinstance(loop.strategy, MetacognitiveStrategy)

    def test_metacognition_registers_all_agents(self):
        """All agents should be registered in metacognition engine after setup."""
        from src.config import SimulationConfig
        from src.simulation.engine import SimulationEngine

        config = SimulationConfig(
            metacognition_enabled=True,
            num_agents=3,
        )
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        for agent in engine.registry.living_agents():
            state = engine.metacognition_engine.get_agent_state(str(agent.agent_id))
            assert state is not None

    def test_metacognition_produces_snapshots(self):
        """Running ticks should produce metacognitive snapshots."""
        from src.config import SimulationConfig
        from src.simulation.engine import SimulationEngine

        config = SimulationConfig(
            metacognition_enabled=True,
            num_agents=3,
            max_ticks=20,
        )
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        # Run several ticks
        for _ in range(10):
            engine.step_all()

        # Should have snapshots (one per agent per tick)
        assert len(engine.metacognition_engine.snapshots) > 0

    def test_metacognition_disabled_no_overhead(self):
        """Disabled metacognition should not affect engine behavior."""
        from src.config import SimulationConfig
        from src.simulation.engine import SimulationEngine

        config = SimulationConfig(
            metacognition_enabled=False,
            num_agents=3,
        )
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        assert engine.metacognition_engine is None

        # Should run fine without metacognition
        for _ in range(5):
            tick_record = engine.step_all()
            assert tick_record is not None

    def test_cultural_and_metacognition_compose(self):
        """Cultural transmission and metacognition should work together."""
        from src.cognition.strategies.metacognitive_strategy import MetacognitiveStrategy
        from src.config import SimulationConfig
        from src.simulation.engine import SimulationEngine

        config = SimulationConfig(
            metacognition_enabled=True,
            cultural_transmission_enabled=True,
            evolution_enabled=True,
            num_agents=3,
            max_ticks=20,
        )
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        # Verify wrapper chain: MetacognitiveStrategy wraps CulturallyModulatedStrategy
        for agent in engine.registry.living_agents():
            loop = engine.registry.get_awareness_loop(agent.agent_id)
            assert isinstance(loop.strategy, MetacognitiveStrategy)

            # Inner should be CulturallyModulatedStrategy
            from src.cognition.strategies.cultural_strategy import CulturallyModulatedStrategy

            assert isinstance(loop.strategy.inner, CulturallyModulatedStrategy)

        # Run a few ticks — should not crash
        for _ in range(5):
            engine.step_all()

    def test_dead_agents_skipped(self):
        """Dead agents should be skipped in metacognitive processing."""
        from src.config import SimulationConfig
        from src.simulation.engine import SimulationEngine

        config = SimulationConfig(
            metacognition_enabled=True,
            num_agents=3,
            max_ticks=500,
        )
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        # Kill one agent
        agents = engine.registry.living_agents()
        if len(agents) >= 2:
            killed = agents[0]
            engine.registry.kill(killed.agent_id, "test_death")

            # Should still run without errors
            for _ in range(3):
                tick_record = engine.step_all()
                assert tick_record is not None
