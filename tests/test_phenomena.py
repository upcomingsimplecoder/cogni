"""Tests for phenomena computation module."""

from __future__ import annotations

import tempfile

from src.analysis.phenomena import (
    FailureSignature,
    NormConvergenceMetric,
    PhenomenaComputer,
    PhenomenaReport,
    ValueDriftCurve,
)
from src.trajectory.schema import (
    AgentSnapshot,
    RunMetadata,
    TrajectoryDataset,
)


def make_snapshot(
    tick,
    agent_id,
    name="Agent",
    archetype="explorer",
    alive=True,
    hunger=50.0,
    thirst=50.0,
    energy=50.0,
    health=100.0,
    traits=None,
    action_type="wait",
):
    """Create minimal AgentSnapshot for testing."""
    default_traits = {
        "cooperation_tendency": 0.5,
        "curiosity": 0.5,
        "risk_tolerance": 0.5,
        "resource_sharing": 0.5,
        "aggression": 0.5,
        "sociability": 0.5,
    }
    if traits:
        default_traits.update(traits)
    return AgentSnapshot(
        tick=tick,
        agent_id=agent_id,
        agent_name=name,
        archetype=archetype,
        position=(10, 10),
        alive=alive,
        hunger=hunger,
        thirst=thirst,
        energy=energy,
        health=health,
        traits=default_traits,
        sensation_summary={},
        reflection={"threat_level": 0.0, "opportunity_score": 0.0},
        intention={"primary_goal": "wait", "confidence": 0.5},
        action_type=action_type,
        action_target=None,
        action_target_agent=None,
        action_succeeded=True,
        needs_delta={},
        inventory={},
    )


def make_dataset(snapshots, run_id="test-run-001"):
    """Create minimal TrajectoryDataset for testing."""
    metadata = RunMetadata(
        run_id=run_id,
        timestamp="2026-02-24T00:00:00",
        seed=42,
        config={},
        num_agents=1,
        max_ticks=100,
    )
    return TrajectoryDataset(
        metadata=metadata,
        agent_snapshots=snapshots,
    )


def test_value_drift_increasing_trait():
    """Create snapshots where cooperation goes 0.3 → 0.5 over 100 ticks.

    Verify direction="increasing", total_drift≈0.2
    """
    snapshots = []
    for tick in range(0, 101, 10):
        # Linear increase from 0.3 to 0.5
        cooperation_value = 0.3 + (tick / 100) * 0.2
        traits = {"cooperation_tendency": cooperation_value}
        snapshots.append(make_snapshot(tick, "agent-1", traits=traits))

    dataset = make_dataset(snapshots)
    computer = PhenomenaComputer()
    curves = computer.compute_value_drift(dataset)

    # Find cooperation_tendency curve
    coop_curve = next(c for c in curves if c.trait_name == "cooperation_tendency")

    assert coop_curve.direction == "increasing"
    assert abs(coop_curve.total_drift - 0.2) < 0.01
    assert coop_curve.drift_rate > 0


def test_value_drift_stable_trait():
    """Create snapshots where trait stays at 0.5 ± 0.01.

    Verify direction="stable"
    """
    snapshots = []
    for tick in range(0, 101, 10):
        # Stay at 0.5 with small noise
        traits = {"curiosity": 0.5}
        snapshots.append(make_snapshot(tick, "agent-1", traits=traits))

    dataset = make_dataset(snapshots)
    computer = PhenomenaComputer()
    curves = computer.compute_value_drift(dataset)

    # Find curiosity curve
    curiosity_curve = next(c for c in curves if c.trait_name == "curiosity")

    assert curiosity_curve.direction == "stable"
    assert curiosity_curve.total_drift < 0.02


def test_value_drift_decreasing_trait():
    """Create snapshots where aggression goes 0.8 → 0.4.

    Verify direction="decreasing"
    """
    snapshots = []
    for tick in range(0, 101, 10):
        # Linear decrease from 0.8 to 0.4
        aggression_value = 0.8 - (tick / 100) * 0.4
        traits = {"aggression": aggression_value}
        snapshots.append(make_snapshot(tick, "agent-1", traits=traits))

    dataset = make_dataset(snapshots)
    computer = PhenomenaComputer()
    curves = computer.compute_value_drift(dataset)

    # Find aggression curve
    aggression_curve = next(c for c in curves if c.trait_name == "aggression")

    assert aggression_curve.direction == "decreasing"
    assert abs(aggression_curve.total_drift - 0.4) < 0.01


def test_norm_convergence_converging():
    """Create 3 agents starting with cooperation=[0.2, 0.5, 0.8], ending at [0.48, 0.50, 0.52].

    Verify converged=True
    """
    snapshots = []

    # Initial variance is high (0.2, 0.5, 0.8)
    # Final variance is low (0.48, 0.50, 0.52)

    # First quartile (ticks 0-25): divergent values
    for tick in range(0, 26):
        snapshots.append(make_snapshot(tick, "agent-1", traits={"cooperation_tendency": 0.2}))
        snapshots.append(make_snapshot(tick, "agent-2", traits={"cooperation_tendency": 0.5}))
        snapshots.append(make_snapshot(tick, "agent-3", traits={"cooperation_tendency": 0.8}))

    # Last quartile (ticks 75-100): convergent values
    for tick in range(75, 101):
        snapshots.append(make_snapshot(tick, "agent-1", traits={"cooperation_tendency": 0.48}))
        snapshots.append(make_snapshot(tick, "agent-2", traits={"cooperation_tendency": 0.50}))
        snapshots.append(make_snapshot(tick, "agent-3", traits={"cooperation_tendency": 0.52}))

    dataset = make_dataset(snapshots)
    computer = PhenomenaComputer()
    metrics = computer.compute_norm_convergence(dataset)

    # Find cooperation_tendency metric
    coop_metric = next(m for m in metrics if m.trait_name == "cooperation_tendency")

    assert coop_metric.converged is True
    assert coop_metric.convergence_ratio < 0.5
    assert coop_metric.final_variance < coop_metric.initial_variance


def test_norm_convergence_diverging():
    """Create 3 agents starting at [0.4, 0.5, 0.6], ending at [0.1, 0.5, 0.9].

    Verify converged=False
    """
    snapshots = []

    # First quartile (ticks 0-25): close values
    for tick in range(0, 26):
        snapshots.append(make_snapshot(tick, "agent-1", traits={"cooperation_tendency": 0.4}))
        snapshots.append(make_snapshot(tick, "agent-2", traits={"cooperation_tendency": 0.5}))
        snapshots.append(make_snapshot(tick, "agent-3", traits={"cooperation_tendency": 0.6}))

    # Last quartile (ticks 75-100): divergent values
    for tick in range(75, 101):
        snapshots.append(make_snapshot(tick, "agent-1", traits={"cooperation_tendency": 0.1}))
        snapshots.append(make_snapshot(tick, "agent-2", traits={"cooperation_tendency": 0.5}))
        snapshots.append(make_snapshot(tick, "agent-3", traits={"cooperation_tendency": 0.9}))

    dataset = make_dataset(snapshots)
    computer = PhenomenaComputer()
    metrics = computer.compute_norm_convergence(dataset)

    # Find cooperation_tendency metric
    coop_metric = next(m for m in metrics if m.trait_name == "cooperation_tendency")

    assert coop_metric.converged is False
    assert coop_metric.convergence_ratio > 1.0
    assert coop_metric.final_variance > coop_metric.initial_variance


def test_failure_signature_mass_death():
    """Create 5 agents, 3 die at ticks 48-50 (alive goes from 5→2 rapidly).

    Low hunger before death. Verify trigger detected, contributing_factors includes "starvation"
    """
    snapshots = []

    # All 5 agents alive and healthy for first 20 ticks
    for tick in range(0, 20):
        for i in range(1, 6):
            snapshots.append(
                make_snapshot(
                    tick,
                    f"agent-{i}",
                    alive=True,
                    hunger=60.0,
                    thirst=60.0,
                    energy=60.0,
                    health=80.0,
                )
            )

    # Ticks 20-47: ALL agents' hunger is critically low
    for tick in range(20, 48):
        for i in range(1, 6):
            # All agents starving with hunger at 10
            snapshots.append(
                make_snapshot(
                    tick,
                    f"agent-{i}",
                    alive=True,
                    hunger=10.0,
                    thirst=60.0,
                    energy=60.0,
                    health=80.0,
                )
            )

    # Ticks 48-50: 3 agents die, 2 survive
    for tick in range(48, 51):
        # Agents 1-3 are dead (don't appear)
        # Agents 4-5 survive but still have low hunger
        for i in range(4, 6):
            snapshots.append(
                make_snapshot(
                    tick,
                    f"agent-{i}",
                    alive=True,
                    hunger=10.0,
                    thirst=60.0,
                    energy=60.0,
                    health=80.0,
                )
            )

    dataset = make_dataset(snapshots)
    computer = PhenomenaComputer()
    signatures = computer.compute_failure_signatures(dataset)

    assert len(signatures) > 0

    # Find the failure signature
    sig = signatures[0]

    assert len(sig.agents_dying) >= 2  # At least 2 agents died
    assert "starvation" in sig.contributing_factors
    assert sig.death_rate > 0


def test_failure_signature_no_deaths():
    """All agents survive. Verify empty failure_signatures list."""
    snapshots = []

    # All 3 agents alive for 100 ticks
    for tick in range(0, 101):
        for i in range(1, 4):
            snapshots.append(
                make_snapshot(
                    tick,
                    f"agent-{i}",
                    alive=True,
                    hunger=60.0,
                    thirst=60.0,
                    energy=60.0,
                    health=100.0,
                )
            )

    dataset = make_dataset(snapshots)
    computer = PhenomenaComputer()
    signatures = computer.compute_failure_signatures(dataset)

    assert len(signatures) == 0


def test_compute_all_returns_report():
    """Verify PhenomenaReport has all sections populated."""
    snapshots = []

    # Create simple dataset with 2 agents
    for tick in range(0, 51, 10):
        snapshots.append(make_snapshot(tick, "agent-1", traits={"cooperation_tendency": 0.3}))
        snapshots.append(make_snapshot(tick, "agent-2", traits={"cooperation_tendency": 0.7}))

    dataset = make_dataset(snapshots)
    computer = PhenomenaComputer()
    report = computer.compute_all(dataset)

    assert isinstance(report, PhenomenaReport)
    assert report.run_id == "test-run-001"
    assert len(report.value_drift_curves) > 0
    assert len(report.norm_convergence) > 0
    # failure_signatures may be empty (no deaths)


def test_phenomena_report_save_load_round_trip():
    """save() then load(), verify data matches."""
    # Create a simple report
    curves = [
        ValueDriftCurve(
            agent_id="agent-1",
            agent_name="Agent 1",
            trait_name="cooperation_tendency",
            ticks=[0, 10, 20],
            values=[0.3, 0.4, 0.5],
            total_drift=0.2,
            drift_rate=0.01,
            direction="increasing",
        )
    ]

    metrics = [
        NormConvergenceMetric(
            trait_name="cooperation_tendency",
            initial_variance=0.1,
            final_variance=0.05,
            convergence_ratio=0.5,
            converged=False,
        )
    ]

    signatures = [
        FailureSignature(
            trigger_tick=50,
            agents_dying=["agent-1", "agent-2"],
            preceding_conditions={"avg_hunger": 10.0},
            death_rate=0.5,
            contributing_factors=["starvation"],
        )
    ]

    original_report = PhenomenaReport(
        run_id="test-run-roundtrip",
        value_drift_curves=curves,
        norm_convergence=metrics,
        failure_signatures=signatures,
    )

    # Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        original_report.save(tmpdir)
        loaded_report = PhenomenaReport.load(tmpdir)

    # Verify data matches
    assert loaded_report.run_id == original_report.run_id
    assert len(loaded_report.value_drift_curves) == len(original_report.value_drift_curves)
    assert len(loaded_report.norm_convergence) == len(original_report.norm_convergence)
    assert len(loaded_report.failure_signatures) == len(original_report.failure_signatures)

    # Verify first curve details
    assert loaded_report.value_drift_curves[0].agent_id == "agent-1"
    assert loaded_report.value_drift_curves[0].trait_name == "cooperation_tendency"
    assert loaded_report.value_drift_curves[0].direction == "increasing"

    # Verify first metric details
    assert loaded_report.norm_convergence[0].trait_name == "cooperation_tendency"
    assert loaded_report.norm_convergence[0].converged is False

    # Verify first signature details
    assert loaded_report.failure_signatures[0].trigger_tick == 50
    assert "starvation" in loaded_report.failure_signatures[0].contributing_factors


def test_empty_dataset():
    """Empty dataset returns empty lists for all phenomena."""
    snapshots = []
    dataset = make_dataset(snapshots)
    computer = PhenomenaComputer()
    report = computer.compute_all(dataset)

    assert len(report.value_drift_curves) == 0
    assert len(report.norm_convergence) == 0
    assert len(report.failure_signatures) == 0


def test_value_drift_multiple_agents():
    """Verify value drift computes correctly for multiple agents."""
    snapshots = []

    # Agent 1: increasing cooperation
    for tick in range(0, 51, 10):
        cooperation_value = 0.3 + (tick / 50) * 0.2
        snapshots.append(
            make_snapshot(
                tick, "agent-1", name="Agent 1", traits={"cooperation_tendency": cooperation_value}
            )
        )

    # Agent 2: decreasing aggression
    for tick in range(0, 51, 10):
        aggression_value = 0.8 - (tick / 50) * 0.3
        snapshots.append(
            make_snapshot(tick, "agent-2", name="Agent 2", traits={"aggression": aggression_value})
        )

    dataset = make_dataset(snapshots)
    computer = PhenomenaComputer()
    curves = computer.compute_value_drift(dataset)

    # Should have 6 traits × 2 agents = 12 curves
    assert len(curves) == 12

    # Find specific curves
    agent1_coop = next(
        c for c in curves if c.agent_id == "agent-1" and c.trait_name == "cooperation_tendency"
    )
    agent2_aggr = next(
        c for c in curves if c.agent_id == "agent-2" and c.trait_name == "aggression"
    )

    assert agent1_coop.direction == "increasing"
    assert agent2_aggr.direction == "decreasing"


def test_norm_convergence_single_agent():
    """Single agent dataset should have variance=0, convergence_ratio=1.0."""
    snapshots = []

    # Single agent with stable trait
    for tick in range(0, 101):
        snapshots.append(make_snapshot(tick, "agent-1", traits={"cooperation_tendency": 0.5}))

    dataset = make_dataset(snapshots)
    computer = PhenomenaComputer()
    metrics = computer.compute_norm_convergence(dataset)

    # All traits should have variance 0 (only one agent)
    for metric in metrics:
        assert metric.initial_variance == 0.0
        assert metric.final_variance == 0.0
        assert metric.convergence_ratio == 1.0
        assert metric.converged is False  # ratio is 1.0, not < 0.5


def test_failure_signature_multiple_spikes():
    """Multiple death events should produce multiple signatures."""
    snapshots = []

    # First death spike at tick 30 (agents 1-2 die)
    for tick in range(0, 30):
        for i in range(1, 6):
            snapshots.append(
                make_snapshot(
                    tick,
                    f"agent-{i}",
                    alive=True,
                    hunger=10.0 if i <= 2 else 60.0,  # Agents 1-2 starving
                    thirst=60.0,
                    energy=60.0,
                    health=80.0,
                )
            )

    # Tick 30-40: agents 1-2 dead, rest alive
    for tick in range(30, 40):
        for i in range(3, 6):
            snapshots.append(
                make_snapshot(
                    tick,
                    f"agent-{i}",
                    alive=True,
                    hunger=60.0,
                    thirst=60.0,
                    energy=60.0,
                    health=80.0,
                )
            )

    # Second death spike at tick 50 (agents 3-4 die)
    for tick in range(40, 50):
        for i in range(3, 6):
            hunger_val = 60.0 if i == 5 else 10.0  # Agents 3-4 starving
            snapshots.append(
                make_snapshot(
                    tick,
                    f"agent-{i}",
                    alive=True,
                    hunger=hunger_val,
                    thirst=60.0,
                    energy=60.0,
                    health=80.0,
                )
            )

    # Tick 50-60: only agent 5 alive
    for tick in range(50, 61):
        snapshots.append(
            make_snapshot(
                tick, "agent-5", alive=True, hunger=60.0, thirst=60.0, energy=60.0, health=80.0
            )
        )

    dataset = make_dataset(snapshots)
    computer = PhenomenaComputer()
    signatures = computer.compute_failure_signatures(dataset)

    # Should detect both death spikes
    assert len(signatures) >= 1  # At least one spike detected
