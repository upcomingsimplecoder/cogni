"""Tests for emergence detection and metrics collection."""

from __future__ import annotations

from src.emergence.detector import EmergenceDetector
from src.emergence.events import EmergentEvent
from src.emergence.metrics import MetricsCollector, TickMetrics
from src.simulation.entities import AgentNeeds


class MockAgent:
    """Mock agent for spatial tests."""

    def __init__(self, agent_id: str, x: int, y: int):
        self.agent_id = agent_id
        self.x = x
        self.y = y


class MockAgentWithNeeds:
    """Mock agent with needs for metrics tests."""

    def __init__(self, agent_id: str, needs: AgentNeeds | None = None):
        self.agent_id = agent_id
        self.needs = needs or AgentNeeds()


class TestEmergentEvent:
    """Test EmergentEvent data structure."""

    def test_event_str_format(self):
        """EmergentEvent.__str__() should format correctly."""
        event = EmergentEvent(
            tick=42,
            pattern_type="cluster",
            agents_involved=["agent1", "agent2"],
            description="2 agents clustered at (10, 15)",
        )
        result = str(event)
        assert result == "[Tick 42] cluster: 2 agents clustered at (10, 15)"

    def test_event_stores_all_fields(self):
        """EmergentEvent should store tick, pattern_type, agents_involved, description, data."""
        event = EmergentEvent(
            tick=100,
            pattern_type="sharing_network",
            agents_involved=["a1", "a2", "a3"],
            description="3-agent sharing network",
            data={"exchange_count": 5},
        )
        assert event.tick == 100
        assert event.pattern_type == "sharing_network"
        assert event.agents_involved == ["a1", "a2", "a3"]
        assert event.description == "3-agent sharing network"
        assert event.data == {"exchange_count": 5}


class TestMetricsCollector:
    """Test MetricsCollector for aggregate stats."""

    def test_collect_returns_tick_metrics(self):
        """MetricsCollector.collect() should return TickMetrics."""
        collector = MetricsCollector()
        agent = MockAgentWithNeeds("agent1")

        metrics = collector.collect(
            tick=1,
            living_agents=[agent],
            dead_count=0,
        )

        assert isinstance(metrics, TickMetrics)
        assert metrics.tick == 1
        assert metrics.agents_alive == 1
        assert metrics.agents_dead == 0

    def test_collect_computes_averages_correctly(self):
        """MetricsCollector.collect() should compute correct averages."""
        collector = MetricsCollector()

        agent1 = MockAgentWithNeeds("a1")
        agent1.needs.health = 80.0
        agent1.needs.hunger = 60.0
        agent1.needs.thirst = 50.0
        agent1.needs.energy = 40.0

        agent2 = MockAgentWithNeeds("a2")
        agent2.needs.health = 100.0
        agent2.needs.hunger = 80.0
        agent2.needs.thirst = 70.0
        agent2.needs.energy = 60.0

        metrics = collector.collect(
            tick=10,
            living_agents=[agent1, agent2],
            dead_count=1,
        )

        assert metrics.agents_alive == 2
        assert metrics.agents_dead == 1
        assert metrics.avg_health == 90.0
        assert metrics.avg_hunger == 70.0
        assert metrics.avg_thirst == 60.0
        assert metrics.avg_energy == 50.0

    def test_collect_counts_cooperation_and_aggression(self):
        """MetricsCollector.collect() should count GIVE and ATTACK actions."""
        collector = MetricsCollector()
        agent = MockAgentWithNeeds("a1")

        tick_actions = [
            ("give", True),
            ("give", True),
            ("attack", True),
            ("move", True),
            ("give", False),  # failed give doesn't count
        ]

        metrics = collector.collect(
            tick=5,
            living_agents=[agent],
            dead_count=0,
            tick_actions=tick_actions,
        )

        assert metrics.cooperation_events == 2
        assert metrics.aggression_events == 1
        assert metrics.resource_sharing_rate == 2.0  # 2 successful gives / 1 agent

    def test_trend_returns_stable_with_less_than_2_data_points(self):
        """MetricsCollector.trend() should return 'stable' with < 2 data points."""
        collector = MetricsCollector()

        # No data
        assert collector.trend("avg_health") == "stable"

        # One data point
        collector.collect(tick=1, living_agents=[MockAgentWithNeeds("a1")], dead_count=0)
        assert collector.trend("avg_health") == "stable"

    def test_trend_detects_increasing(self):
        """MetricsCollector.trend() should detect increasing trend."""
        collector = MetricsCollector()

        # Create 10 metrics with increasing cooperation_events
        for i in range(10):
            agent = MockAgentWithNeeds(f"a{i}")
            collector.collect(
                tick=i,
                living_agents=[agent],
                dead_count=0,
                tick_actions=[("give", True)] * i,  # 0, 1, 2, 3... gives
            )

        trend = collector.trend("cooperation_events", window=10)
        assert trend == "increasing"

    def test_trend_detects_decreasing(self):
        """MetricsCollector.trend() should detect decreasing trend."""
        collector = MetricsCollector()

        # Create 10 metrics with decreasing avg_health
        for i in range(10):
            agent = MockAgentWithNeeds(f"a{i}")
            agent.needs.health = 100.0 - (i * 5)  # 100, 95, 90, 85...
            collector.collect(
                tick=i,
                living_agents=[agent],
                dead_count=0,
            )

        trend = collector.trend("avg_health", window=10)
        assert trend == "decreasing"

    def test_latest_returns_most_recent(self):
        """MetricsCollector.latest() should return most recent metrics."""
        collector = MetricsCollector()

        # No data
        assert collector.latest() is None

        # Add some metrics
        collector.collect(tick=1, living_agents=[MockAgentWithNeeds("a1")], dead_count=0)
        collector.collect(tick=2, living_agents=[MockAgentWithNeeds("a1")], dead_count=0)
        collector.collect(tick=3, living_agents=[MockAgentWithNeeds("a1")], dead_count=0)

        latest = collector.latest()
        assert latest is not None
        assert latest.tick == 3


class TestEmergenceDetector:
    """Test EmergenceDetector pattern detection."""

    def test_detects_spatial_cluster_when_agents_within_distance(self):
        """EmergenceDetector should detect cluster when 2+ agents within distance."""
        detector = EmergenceDetector(cluster_distance=3, cluster_sustain=1)

        # Two agents within distance
        agents = [
            MockAgent("a1", 10, 10),
            MockAgent("a2", 12, 10),  # distance = 2
        ]

        events = detector.detect(tick=1, agents=agents)

        # First detection doesn't trigger event yet (needs sustain)
        assert len(events) == 0

        # Sustain for one more tick
        events = detector.detect(tick=2, agents=agents)

        assert len(events) == 1
        assert events[0].pattern_type == "cluster"
        assert set(events[0].agents_involved) == {"a1", "a2"}

    def test_cluster_sustained_for_n_ticks_triggers_event(self):
        """EmergenceDetector should trigger event when cluster sustained for N ticks."""
        detector = EmergenceDetector(cluster_distance=3, cluster_sustain=5)

        agents = [
            MockAgent("a1", 10, 10),
            MockAgent("a2", 12, 10),
        ]

        # Run for sustain ticks (need to reach duration = 5 from first_seen)
        # first_seen will be tick 1, so we need to reach tick 6 (duration = 5)
        for tick in range(1, 7):
            events = detector.detect(tick=tick, agents=agents)
            cluster_events = [e for e in events if e.pattern_type == "cluster"]
            if tick < 6:
                # No event yet
                assert len(cluster_events) == 0

        # At tick 6, duration = 6-1 = 5, should trigger
        events = detector.detect(tick=6, agents=agents)
        cluster_events = [e for e in events if e.pattern_type == "cluster"]
        assert len(cluster_events) > 0

    def test_detects_sharing_network(self):
        """EmergenceDetector should detect sharing network.

        3+ GIVE events between pair in 50 ticks.
        """
        detector = EmergenceDetector()

        agents = [
            MockAgent("a1", 10, 10),
            MockAgent("a2", 12, 12),
        ]

        # Simulate 3 GIVE actions between a1 and a2
        tick_actions = [
            ("a1", "give", True, "a2"),
            ("a1", "give", True, "a2"),
            ("a2", "give", True, "a1"),
        ]

        events = detector.detect(tick=10, agents=agents, tick_actions=tick_actions)

        sharing_events = [e for e in events if e.pattern_type == "sharing_network"]
        assert len(sharing_events) == 1
        assert set(sharing_events[0].agents_involved) == {"a1", "a2"}
        assert sharing_events[0].data["exchange_count"] == 3

    def test_detects_territory(self):
        """EmergenceDetector should detect territory (agent staying in bounded area + threats)."""
        detector = EmergenceDetector(territory_radius=5, territory_sustain=20)

        agent = MockAgent("a1", 10, 10)

        # Agent stays in small area for 20 ticks
        for tick in range(1, 21):
            # Move slightly but stay within radius
            agent.x = 10 + (tick % 3)
            agent.y = 10 + (tick % 2)
            detector.detect(tick=tick, agents=[agent])

        # Add a threat (via message tracking)
        detector._threats_sent["a1"] = 2

        events = detector.detect(tick=21, agents=[agent])

        territory_events = [e for e in events if e.pattern_type == "territory"]
        assert len(territory_events) == 1
        assert territory_events[0].agents_involved == ["a1"]
        assert territory_events[0].data["threats"] == 2

    def test_detects_specialization(self):
        """EmergenceDetector should detect specialization (>60% actions in one category)."""
        detector = EmergenceDetector()

        agent = MockAgent("a1", 10, 10)

        # Agent performs mostly gathering actions
        tick_actions = (
            [("a1", "gather", True, None)] * 20
            + [("a1", "eat", True, None)] * 5
            + [("a1", "move", True, None)] * 5
        )

        events = detector.detect(tick=30, agents=[agent], tick_actions=tick_actions)

        spec_events = [e for e in events if e.pattern_type == "specialization"]
        assert len(spec_events) == 1
        assert spec_events[0].agents_involved == ["a1"]
        assert spec_events[0].data["category"] == "gathering"
        assert spec_events[0].data["ratio"] > 0.6

    def test_no_events_when_no_patterns(self):
        """EmergenceDetector should return empty list when no patterns detected."""
        detector = EmergenceDetector()

        # Single agent, far apart, no sustained behavior
        agents = [MockAgent("a1", 10, 10)]

        events = detector.detect(tick=1, agents=agents)

        assert events == []
