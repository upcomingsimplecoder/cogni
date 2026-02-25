"""Tests for performance timing instrumentation."""

from src.metrics.timing import PerformanceMonitor, TickTiming


def test_tick_timing_creation():
    """Test TickTiming dataclass creation."""
    timing = TickTiming(
        decision_ms=10.5,
        action_execution_ms=5.2,
        message_delivery_ms=1.3,
        need_decay_ms=0.8,
        memory_update_ms=2.1,
        emergence_detection_ms=3.0,
        total_ms=22.9,
    )
    assert timing.decision_ms == 10.5
    assert timing.action_execution_ms == 5.2
    assert timing.total_ms == 22.9


def test_tick_timing_defaults():
    """Test TickTiming has sensible defaults."""
    timing = TickTiming()
    assert timing.decision_ms == 0.0
    assert timing.total_ms == 0.0


def test_performance_monitor_records_ticks():
    """Test PerformanceMonitor records tick timings."""
    monitor = PerformanceMonitor()
    timing1 = TickTiming(decision_ms=10.0, total_ms=20.0)
    timing2 = TickTiming(decision_ms=12.0, total_ms=25.0)

    monitor.record_tick(timing1)
    monitor.record_tick(timing2)

    summary = monitor.summary
    assert summary["total_ticks"] == 2
    assert summary["avg_tick_ms"] == 22.5
    assert summary["avg_decision_ms"] == 11.0
    assert summary["slowest_tick_ms"] == 25.0


def test_performance_monitor_summary_fields():
    """Test PerformanceMonitor summary has expected fields."""
    monitor = PerformanceMonitor()
    timing = TickTiming(
        decision_ms=10.0,
        action_execution_ms=5.0,
        total_ms=20.0,
    )
    monitor.record_tick(timing)

    summary = monitor.summary
    assert "total_ticks" in summary
    assert "avg_tick_ms" in summary
    assert "avg_decision_ms" in summary
    assert "avg_action_ms" in summary
    assert "slowest_tick_ms" in summary
    assert "strategy_breakdown" in summary
    assert "llm_calls" in summary
    assert "llm_avg_ms" in summary
    assert "llm_parse_failures" in summary


def test_performance_monitor_empty_summary():
    """Test PerformanceMonitor returns empty dict when no data."""
    monitor = PerformanceMonitor()
    assert monitor.summary == {}


def test_strategy_call_tracking():
    """Test PerformanceMonitor tracks strategy calls."""
    monitor = PerformanceMonitor()

    monitor.record_strategy_call("PersonalityStrategy", 5.0)
    monitor.record_strategy_call("PersonalityStrategy", 7.0)
    monitor.record_strategy_call("LLMStrategy", 120.0)

    # Need at least one tick for summary to be generated
    monitor.record_tick(TickTiming(total_ms=10.0))

    summary = monitor.summary
    breakdown = summary["strategy_breakdown"]

    assert "PersonalityStrategy" in breakdown
    assert breakdown["PersonalityStrategy"]["calls"] == 2
    assert breakdown["PersonalityStrategy"]["total_ms"] == 12.0
    assert breakdown["PersonalityStrategy"]["avg_ms"] == 6.0

    assert "LLMStrategy" in breakdown
    assert breakdown["LLMStrategy"]["calls"] == 1
    assert breakdown["LLMStrategy"]["total_ms"] == 120.0
    assert breakdown["LLMStrategy"]["avg_ms"] == 120.0


def test_strategy_breakdown_division_by_zero():
    """Test strategy breakdown handles zero calls gracefully."""
    monitor = PerformanceMonitor()
    monitor.record_tick(TickTiming(total_ms=10.0))

    summary = monitor.summary
    # Should not crash even with no strategy calls
    assert "strategy_breakdown" in summary
    assert summary["strategy_breakdown"] == {}


def test_llm_avg_with_zero_calls():
    """Test LLM avg calculation handles zero calls."""
    monitor = PerformanceMonitor()
    monitor.record_tick(TickTiming(total_ms=10.0))

    summary = monitor.summary
    # Should not crash with division by zero
    assert summary["llm_calls"] == 0
    assert summary["llm_avg_ms"] == 0.0
