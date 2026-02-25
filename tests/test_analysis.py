"""Tests for the analysis toolkit."""

import tempfile
from pathlib import Path

import pytest

from src.analysis.aggregator import DatasetAggregator
from src.analysis.behavioral import BehavioralAnalyzer
from src.analysis.correlations import TraitBehaviorAnalyzer
from src.analysis.reports import AnalysisReportGenerator
from src.analysis.survival import SurvivalAnalyzer
from src.trajectory.schema import (
    AgentSnapshot,
    RunMetadata,
    TrajectoryDataset,
)

# Test helpers to build synthetic datasets


def make_snapshot(
    tick: int,
    agent_id: str,
    agent_name: str,
    archetype: str,
    alive: bool = True,
    action_type: str = "wait",
    traits: dict | None = None,
) -> AgentSnapshot:
    """Create a test agent snapshot."""
    if traits is None:
        traits = {
            "cooperation_tendency": 0.5,
            "curiosity": 0.5,
            "risk_tolerance": 0.5,
            "resource_sharing": 0.5,
            "aggression": 0.5,
            "sociability": 0.5,
        }

    return AgentSnapshot(
        tick=tick,
        agent_id=agent_id,
        agent_name=agent_name,
        archetype=archetype,
        position=(0, 0),
        alive=alive,
        hunger=50.0,
        thirst=50.0,
        energy=50.0,
        health=100.0,
        traits=traits,
        sensation_summary={"visible_agent_count": 0},
        reflection={"threat_level": 0.0},
        intention={"primary_goal": "survive"},
        action_type=action_type,
        action_target=None,
        action_target_agent=None,
        action_succeeded=True,
        needs_delta={},
        inventory={},
        messages_sent=[],
        messages_received=[],
        internal_monologue="",
        trait_changes=[],
    )


def make_dataset(snapshots: list[AgentSnapshot], run_id: str = "test_run") -> TrajectoryDataset:
    """Create a test trajectory dataset."""
    metadata = RunMetadata(
        run_id=run_id,
        timestamp="2026-01-01T00:00:00",
        seed=42,
        config={},
        num_agents=len(set(s.agent_id for s in snapshots)),
        max_ticks=1000,
        actual_ticks=max((s.tick for s in snapshots), default=0),
        agents=[],
        architecture="SRIE",
        final_state={"agents_alive": 0, "agents_dead": 0},
    )

    return TrajectoryDataset(
        metadata=metadata,
        agent_snapshots=snapshots,
        emergence_events=[],
    )


# Correlation tests


def test_trait_action_correlation_perfect_positive():
    """Test correlation with perfect positive relationship."""
    # Create agents where high cooperation → high give frequency
    snapshots = []

    for i in range(10):
        agent_id = f"agent_{i}"
        coop_value = i / 10.0  # 0.0 to 0.9

        # Number of give actions proportional to cooperation
        give_count = int(coop_value * 10)
        _wait_count = 10 - give_count

        for tick in range(give_count):
            snapshots.append(
                make_snapshot(
                    tick=tick,
                    agent_id=agent_id,
                    agent_name=f"Agent {i}",
                    archetype="cooperator",
                    action_type="give",
                    traits={"cooperation_tendency": coop_value},
                )
            )

        for tick in range(give_count, 10):
            snapshots.append(
                make_snapshot(
                    tick=tick,
                    agent_id=agent_id,
                    agent_name=f"Agent {i}",
                    archetype="cooperator",
                    action_type="wait",
                    traits={"cooperation_tendency": coop_value},
                )
            )

    dataset = make_dataset(snapshots)
    analyzer = TraitBehaviorAnalyzer()

    result = analyzer.trait_action_correlation(dataset)

    # Should be strong positive correlation between cooperation and give
    assert "cooperation_tendency" in result
    assert "give" in result["cooperation_tendency"]
    assert result["cooperation_tendency"]["give"] > 0.8


def test_trait_action_correlation_perfect_negative():
    """Test correlation with perfect negative relationship."""
    # Create agents where high aggression → low wait frequency
    snapshots = []

    for i in range(10):
        agent_id = f"agent_{i}"
        aggression = i / 10.0

        # Wait count inversely proportional to aggression
        wait_count = int((1.0 - aggression) * 10)
        attack_count = 10 - wait_count

        for tick in range(attack_count):
            snapshots.append(
                make_snapshot(
                    tick=tick,
                    agent_id=agent_id,
                    agent_name=f"Agent {i}",
                    archetype="aggressive",
                    action_type="attack",
                    traits={"aggression": aggression},
                )
            )

        for tick in range(attack_count, 10):
            snapshots.append(
                make_snapshot(
                    tick=tick,
                    agent_id=agent_id,
                    agent_name=f"Agent {i}",
                    archetype="aggressive",
                    action_type="wait",
                    traits={"aggression": aggression},
                )
            )

    dataset = make_dataset(snapshots)
    analyzer = TraitBehaviorAnalyzer()

    result = analyzer.trait_action_correlation(dataset)

    # Should be strong negative correlation between aggression and wait
    assert result["aggression"]["wait"] < -0.8


def test_trait_survival_correlation():
    """Test trait-survival correlation."""
    # Create agents where high cooperation → longer survival
    snapshots = []

    for i in range(10):
        agent_id = f"agent_{i}"
        coop_value = i / 10.0

        # Survival proportional to cooperation (10 to 100 ticks)
        survival_ticks = 10 + int(coop_value * 90)

        for tick in range(survival_ticks):
            snapshots.append(
                make_snapshot(
                    tick=tick,
                    agent_id=agent_id,
                    agent_name=f"Agent {i}",
                    archetype="cooperator",
                    traits={"cooperation_tendency": coop_value},
                )
            )

    dataset = make_dataset(snapshots)
    analyzer = TraitBehaviorAnalyzer()

    result = analyzer.trait_survival_correlation(dataset)

    # Should be strong positive correlation
    assert result["cooperation_tendency"] > 0.8


def test_optimal_trait_profile():
    """Test optimal trait profile across multiple runs."""
    datasets = []

    # Create 3 runs where high curiosity agents survive longer
    for run in range(3):
        snapshots = []

        for i in range(10):
            agent_id = f"agent_{i}"
            curiosity = i / 10.0
            survival_ticks = 10 + int(curiosity * 90)

            for tick in range(survival_ticks):
                snapshots.append(
                    make_snapshot(
                        tick=tick,
                        agent_id=agent_id,
                        agent_name=f"Agent {i}",
                        archetype="explorer",
                        traits={"curiosity": curiosity},
                    )
                )

        datasets.append(make_dataset(snapshots, run_id=f"run_{run}"))

    analyzer = TraitBehaviorAnalyzer()
    result = analyzer.optimal_trait_profile(datasets)

    # Top survivors should have high curiosity
    assert result["curiosity"] > 0.6


# Survival analysis tests


def test_survival_curve():
    """Test survival curve generation."""
    # 5 agents die at specific ticks (last tick they appear)
    snapshots = []

    death_ticks = [10, 20, 30, 40, 50]

    for i, death_tick in enumerate(death_ticks):
        agent_id = f"agent_{i}"
        for tick in range(death_tick + 1):
            snapshots.append(
                make_snapshot(
                    tick=tick,
                    agent_id=agent_id,
                    agent_name=f"Agent {i}",
                    archetype="gatherer",
                )
            )

    dataset = make_dataset(snapshots)
    analyzer = SurvivalAnalyzer()

    curve = analyzer.survival_curve(dataset)

    # Verify key points
    curve_dict = dict(curve)

    # At tick 0, all alive
    assert curve_dict[0] == 1.0

    # At tick 10, all still alive (agent 0's last tick)
    assert curve_dict[10] == 1.0

    # At tick 11, only 4/5 alive (agent 0 is gone)
    assert curve_dict.get(11, 0.8) == 0.8

    # At tick 20, 4/5 still alive (agent 1's last tick)
    assert curve_dict[20] == 0.8

    # At tick 50, 1/5 alive (agent 4's last tick, they're still counted)
    assert curve_dict[50] == 0.2


def test_death_cause_distribution():
    """Test death cause detection."""
    snapshots = []

    # Agent 0: dies of hunger
    for tick in range(10):
        health = 100.0 if tick < 9 else 50.0
        hunger = 50.0 if tick < 9 else 0.0
        snapshots.append(
            make_snapshot(
                tick=tick,
                agent_id="agent_0",
                agent_name="Agent 0",
                archetype="gatherer",
                alive=(tick < 9),
            )
        )
        snapshots[-1].health = health
        snapshots[-1].hunger = hunger

    # Agent 1: dies of thirst
    for tick in range(15):
        health = 100.0 if tick < 14 else 50.0
        thirst = 50.0 if tick < 14 else 0.0
        snapshots.append(
            make_snapshot(
                tick=tick,
                agent_id="agent_1",
                agent_name="Agent 1",
                archetype="gatherer",
                alive=(tick < 14),
            )
        )
        snapshots[-1].health = health
        snapshots[-1].thirst = thirst

    # Agent 2: dies of health
    for tick in range(20):
        health = 50.0 if tick < 19 else 0.0
        snapshots.append(
            make_snapshot(
                tick=tick,
                agent_id="agent_2",
                agent_name="Agent 2",
                archetype="gatherer",
                alive=(tick < 19),
            )
        )
        snapshots[-1].health = health

    dataset = make_dataset(snapshots)
    analyzer = SurvivalAnalyzer()

    causes = analyzer.death_cause_distribution(dataset)

    assert causes.get("hunger", 0) == 1
    assert causes.get("thirst", 0) == 1
    assert causes.get("health", 0) == 1


def test_survival_predictors():
    """Test survival predictor identification."""
    datasets = []

    # Create 2 runs where high risk_tolerance → shorter survival
    for run in range(2):
        snapshots = []

        for i in range(10):
            agent_id = f"agent_{i}"
            risk = i / 10.0
            survival_ticks = 100 - int(risk * 80)  # Higher risk = shorter survival

            for tick in range(survival_ticks):
                snapshots.append(
                    make_snapshot(
                        tick=tick,
                        agent_id=agent_id,
                        agent_name=f"Agent {i}",
                        archetype="explorer",
                        traits={"risk_tolerance": risk},
                    )
                )

        datasets.append(make_dataset(snapshots, run_id=f"run_{run}"))

    analyzer = SurvivalAnalyzer()
    result = analyzer.survival_predictors(datasets)

    # Should identify risk_tolerance as a predictor
    assert "risk_tolerance" in result
    assert result["risk_tolerance"] > 0.0  # Has some information gain


# Behavioral analysis tests


def test_behavioral_fingerprint_gather_only():
    """Test fingerprint for agent that only gathers."""
    snapshots = []

    for tick in range(100):
        snapshots.append(
            make_snapshot(
                tick=tick,
                agent_id="agent_0",
                agent_name="Gatherer",
                archetype="gatherer",
                action_type="gather",
            )
        )

    dataset = make_dataset(snapshots)
    analyzer = BehavioralAnalyzer()

    fingerprint = analyzer.behavioral_fingerprint(dataset, "agent_0")

    # Should be 100% gather
    assert fingerprint["action_distribution"]["gather"] == 1.0
    assert fingerprint["social_ratio"] == 0.0  # No social actions
    assert fingerprint["exploration_ratio"] == 0.0  # No movement


def test_behavioral_shift_detection():
    """Test detection of behavioral shifts."""
    snapshots = []

    # Agent does gather for 100 ticks, then switches to move for 100 ticks
    for tick in range(100):
        snapshots.append(
            make_snapshot(
                tick=tick,
                agent_id="agent_0",
                agent_name="Shifter",
                archetype="explorer",
                action_type="gather",
            )
        )

    for tick in range(100, 200):
        snapshots.append(
            make_snapshot(
                tick=tick,
                agent_id="agent_0",
                agent_name="Shifter",
                archetype="explorer",
                action_type="move",
            )
        )

    dataset = make_dataset(snapshots)
    analyzer = BehavioralAnalyzer()

    shifts = analyzer.behavioral_shift_detection(dataset, "agent_0", window=40)

    # Should detect shift around tick 100
    assert len(shifts) > 0

    # Find shift closest to tick 100
    shift_ticks = [s["tick"] for s in shifts]
    closest_shift = min(shift_ticks, key=lambda t: abs(t - 100))

    # Should be within reasonable range of tick 100
    assert 80 <= closest_shift <= 120


def test_archetype_fidelity():
    """Test archetype fidelity measurement."""
    snapshots = []

    # Agent 0: perfect gatherer (mostly gather, some eat/drink)
    for tick in range(50):
        snapshots.append(
            make_snapshot(
                tick=tick,
                agent_id="agent_0",
                agent_name="Perfect Gatherer",
                archetype="gatherer",
                action_type="gather",
            )
        )

    for tick in range(50, 65):
        snapshots.append(
            make_snapshot(
                tick=tick,
                agent_id="agent_0",
                agent_name="Perfect Gatherer",
                archetype="gatherer",
                action_type="eat",
            )
        )

    # Agent 1: bad gatherer (only attacks)
    for tick in range(100):
        snapshots.append(
            make_snapshot(
                tick=tick,
                agent_id="agent_1",
                agent_name="Bad Gatherer",
                archetype="gatherer",
                action_type="attack",
            )
        )

    dataset = make_dataset(snapshots)
    analyzer = BehavioralAnalyzer()

    fidelity = analyzer.archetype_fidelity(dataset)

    # Agent 0 should have higher fidelity than agent 1
    assert fidelity["agent_0"] > fidelity["agent_1"]


# Aggregation tests


def test_dataset_combination():
    """Test combining multiple datasets."""
    datasets = []

    for run in range(3):
        snapshots = []
        for i in range(5):
            for tick in range(10):
                snapshots.append(
                    make_snapshot(
                        tick=tick,
                        agent_id=f"agent_{i}",
                        agent_name=f"Agent {i}",
                        archetype="gatherer",
                    )
                )

        datasets.append(make_dataset(snapshots, run_id=f"run_{run}"))

    aggregator = DatasetAggregator()
    agg = aggregator.combine(datasets)

    # Should have 3 runs
    assert len(agg.run_ids) == 3

    # Should have 3 * 5 * 10 = 150 snapshots
    assert len(agg.agent_snapshots) == 150

    # Each run should be tagged
    run_ids_in_snapshots = set(run_id for run_id, _ in agg.agent_snapshots)
    assert len(run_ids_in_snapshots) == 3


def test_personality_behavior_matrix():
    """Test personality-behavior pivot table."""
    snapshots = []

    # Create agents with different cooperation levels
    for i in range(15):
        agent_id = f"agent_{i}"
        coop = i / 15.0  # 0.0 to ~0.93

        # High cooperation → more give actions
        if coop < 0.33:  # Low bucket
            action_type = "wait"
        elif coop < 0.67:  # Med bucket
            action_type = "gather"
        else:  # High bucket
            action_type = "give"

        for tick in range(10):
            snapshots.append(
                make_snapshot(
                    tick=tick,
                    agent_id=agent_id,
                    agent_name=f"Agent {i}",
                    archetype="cooperator",
                    action_type=action_type,
                    traits={"cooperation_tendency": coop},
                )
            )

    dataset = make_dataset(snapshots)
    aggregator = DatasetAggregator()
    agg = aggregator.combine([dataset])

    matrix = aggregator.personality_behavior_matrix(agg)

    # Should have cooperation_tendency
    assert "cooperation_tendency" in matrix
    assert "low" in matrix["cooperation_tendency"]
    assert "med" in matrix["cooperation_tendency"]
    assert "high" in matrix["cooperation_tendency"]

    # High bucket should have more give than low bucket
    high_give = matrix["cooperation_tendency"]["high"].get("give", 0.0)
    low_give = matrix["cooperation_tendency"]["low"].get("give", 0.0)
    assert high_give > low_give


def test_architecture_comparison():
    """Test architecture comparison."""
    datasets = []

    # Create 2 SRIE runs and 1 reactive run
    for run_id, arch, survival in [
        ("run_0", "SRIE", 100),
        ("run_1", "SRIE", 120),
        ("run_2", "reactive", 80),
    ]:
        snapshots = []
        for i in range(5):
            for tick in range(survival):
                snapshots.append(
                    make_snapshot(
                        tick=tick,
                        agent_id=f"agent_{i}",
                        agent_name=f"Agent {i}",
                        archetype="gatherer",
                    )
                )

        dataset = make_dataset(snapshots, run_id=run_id)
        dataset.metadata.architecture = arch
        dataset.metadata.actual_ticks = survival
        datasets.append(dataset)

    aggregator = DatasetAggregator()
    agg = aggregator.combine(datasets)

    comparison = aggregator.architecture_comparison(agg)

    # Should have 2 architectures
    assert "SRIE" in comparison
    assert "reactive" in comparison

    # SRIE should have 2 runs, reactive 1
    assert comparison["SRIE"]["num_runs"] == 2
    assert comparison["reactive"]["num_runs"] == 1

    # SRIE average survival should be (100 + 120) / 2 = 110
    assert comparison["SRIE"]["avg_survival_ticks"] == 110.0

    # Reactive should be 80
    assert comparison["reactive"]["avg_survival_ticks"] == 80.0


# Report generation tests


def test_run_report_generation():
    """Test single-run report generation."""
    snapshots = []

    for i in range(5):
        for tick in range(50):
            snapshots.append(
                make_snapshot(
                    tick=tick,
                    agent_id=f"agent_{i}",
                    agent_name=f"Agent {i}",
                    archetype="gatherer",
                    action_type="gather" if tick < 25 else "eat",
                )
            )

    dataset = make_dataset(snapshots)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "report.md"

        generator = AnalysisReportGenerator()
        generator.generate_run_report(dataset, str(output_path))

        # Verify file was created
        assert output_path.exists()

        # Verify content contains key sections
        content = output_path.read_text()
        assert "# AUTOCOG Trajectory Analysis Report" in content
        assert "## Run Summary" in content
        assert "## Trait-Behavior Correlations" in content
        assert "## Survival Analysis" in content
        assert "## Behavioral Analysis" in content


def test_comparison_report_generation():
    """Test multi-run comparison report."""
    datasets = []

    for run in range(2):
        snapshots = []
        for i in range(5):
            for tick in range(50):
                snapshots.append(
                    make_snapshot(
                        tick=tick,
                        agent_id=f"agent_{i}",
                        agent_name=f"Agent {i}",
                        archetype="gatherer",
                    )
                )

        datasets.append(make_dataset(snapshots, run_id=f"run_{run}"))

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "comparison.md"

        generator = AnalysisReportGenerator()
        generator.generate_comparison_report(datasets, str(output_path))

        # Verify file was created
        assert output_path.exists()

        # Verify content
        content = output_path.read_text()
        assert "# AUTOCOG Multi-Run Comparison Report" in content
        assert "## Run Summaries" in content
        assert "## Architecture Comparison" in content
        assert "## Optimal Trait Profile" in content


def test_pearson_correlation_edge_cases():
    """Test Pearson correlation with edge cases."""
    analyzer = TraitBehaviorAnalyzer()

    # Empty lists
    assert analyzer._pearson_correlation([], []) == 0.0

    # Mismatched lengths
    assert analyzer._pearson_correlation([1, 2, 3], [1, 2]) == 0.0

    # Zero variance
    assert analyzer._pearson_correlation([1, 1, 1], [2, 3, 4]) == 0.0

    # Perfect correlation
    result = analyzer._pearson_correlation([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
    assert abs(result - 1.0) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
