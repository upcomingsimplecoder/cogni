"""Tests for visualization dashboard generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.trajectory.schema import (
    AgentSnapshot,
    EmergenceSnapshot,
    RunMetadata,
    TrajectoryDataset,
)
from src.visualization.dashboard import ARCHETYPE_COLORS, DashboardGenerator


@pytest.fixture
def sample_dataset() -> TrajectoryDataset:
    """Create a small sample dataset for testing."""
    metadata = RunMetadata(
        run_id="test-run-123",
        timestamp="2026-01-01T00:00:00Z",
        seed=42,
        config={"world_width": 32, "world_height": 32},
        num_agents=2,
        max_ticks=10,
        actual_ticks=5,
        agents=[
            {"id": "agent-1", "name": "Alice", "archetype": "gatherer"},
            {"id": "agent-2", "name": "Bob", "archetype": "explorer"},
        ],
    )

    snapshots = [
        AgentSnapshot(
            tick=0,
            agent_id="agent-1",
            agent_name="Alice",
            archetype="gatherer",
            position=(5, 5),
            alive=True,
            hunger=80.0,
            thirst=70.0,
            energy=90.0,
            health=100.0,
            traits={"cooperation_tendency": 0.5},
            sensation_summary={"visible_agent_count": 1},
            reflection={"threat_level": 0.0},
            intention={"primary_goal": "gather_food"},
            action_type="gather",
            action_target="berries",
            action_target_agent=None,
            action_succeeded=True,
            needs_delta={"energy": -0.5},
            inventory={"berries": 3},
            internal_monologue="I need to gather more food",
        ),
        AgentSnapshot(
            tick=0,
            agent_id="agent-2",
            agent_name="Bob",
            archetype="explorer",
            position=(10, 10),
            alive=True,
            hunger=75.0,
            thirst=80.0,
            energy=85.0,
            health=95.0,
            traits={"curiosity": 0.9},
            sensation_summary={"visible_agent_count": 1},
            reflection={"opportunity_score": 0.5},
            intention={"primary_goal": "explore"},
            action_type="move",
            action_target=None,
            action_target_agent=None,
            action_succeeded=True,
            needs_delta={"energy": -1.0},
            inventory={},
            internal_monologue="Let's see what's over there",
        ),
        AgentSnapshot(
            tick=1,
            agent_id="agent-1",
            agent_name="Alice",
            archetype="gatherer",
            position=(5, 6),
            alive=True,
            hunger=78.5,
            thirst=69.3,
            energy=89.5,
            health=100.0,
            traits={"cooperation_tendency": 0.5},
            sensation_summary={"visible_agent_count": 0},
            reflection={"threat_level": 0.0},
            intention={"primary_goal": "gather_food"},
            action_type="move",
            action_target=None,
            action_target_agent=None,
            action_succeeded=True,
            needs_delta={"energy": -1.0},
            inventory={"berries": 3},
            internal_monologue="Moving to a better spot",
        ),
    ]

    events = [
        EmergenceSnapshot(
            tick=1,
            pattern_type="clustering",
            agents_involved=["agent-1", "agent-2"],
            description="Agents forming cluster near resources",
            data={"cluster_size": 2},
        )
    ]

    return TrajectoryDataset(metadata=metadata, agent_snapshots=snapshots, emergence_events=events)


def test_dashboard_generator_init(sample_dataset):
    """Test DashboardGenerator initialization."""
    generator = DashboardGenerator(sample_dataset)
    assert generator.dataset == sample_dataset


def test_prepare_data_metadata(sample_dataset):
    """Test metadata extraction from dataset."""
    generator = DashboardGenerator(sample_dataset)
    data = generator._prepare_data()

    assert data["metadata"]["run_id"] == "test-run-123"
    assert data["metadata"]["seed"] == 42
    assert data["metadata"]["num_agents"] == 2
    assert data["metadata"]["actual_ticks"] == 5
    assert data["metadata"]["world_width"] == 32
    assert data["metadata"]["world_height"] == 32


def test_agent_metadata_extraction(sample_dataset):
    """Test agent metadata extraction and color assignment."""
    generator = DashboardGenerator(sample_dataset)
    agents = generator._agent_metadata()

    assert len(agents) == 2

    alice = next(a for a in agents if a["name"] == "Alice")
    assert alice["id"] == "agent-1"
    assert alice["archetype"] == "gatherer"
    assert alice["color"] == ARCHETYPE_COLORS["gatherer"]

    bob = next(a for a in agents if a["name"] == "Bob")
    assert bob["id"] == "agent-2"
    assert bob["archetype"] == "explorer"
    assert bob["color"] == ARCHETYPE_COLORS["explorer"]


def test_compact_tick_data(sample_dataset):
    """Test tick data grouping and field compression."""
    generator = DashboardGenerator(sample_dataset)
    ticks = generator._compact_tick_data()

    # Check tick 0 has 2 agents
    assert 0 in ticks
    assert len(ticks[0]) == 2

    # Check tick 1 has 1 agent
    assert 1 in ticks
    assert len(ticks[1]) == 1

    # Verify field compression
    alice_t0 = next(s for s in ticks[0] if s["id"] == "agent-1")
    assert alice_t0["pos"] == (5, 5)
    assert alice_t0["alive"] is True
    assert alice_t0["hunger"] == 80.0
    assert alice_t0["action"] == "gather"
    assert alice_t0["success"] is True
    assert alice_t0["inventory"] == {"berries": 3}
    assert alice_t0["monologue"] == "I need to gather more food"
    assert alice_t0["intention"] == "gather_food"


def test_compact_tick_data_rounding(sample_dataset):
    """Test that needs values are rounded to 1 decimal place."""
    generator = DashboardGenerator(sample_dataset)
    ticks = generator._compact_tick_data()

    alice_t1 = ticks[1][0]
    assert alice_t1["hunger"] == 78.5
    assert alice_t1["thirst"] == 69.3  # Should be rounded from 69.3


def test_emergence_events_extraction(sample_dataset):
    """Test emergence events are correctly extracted."""
    generator = DashboardGenerator(sample_dataset)
    data = generator._prepare_data()

    assert len(data["emergence_events"]) == 1
    event = data["emergence_events"][0]
    assert event["tick"] == 1
    assert event["type"] == "clustering"
    assert event["agents"] == ["agent-1", "agent-2"]
    assert "cluster" in event["description"].lower()


def test_generate_creates_file(sample_dataset, tmp_path):
    """Test that generate() creates HTML file with embedded data."""
    generator = DashboardGenerator(sample_dataset)
    output_path = tmp_path / "dashboard.html"

    generator.generate(str(output_path))

    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")

    # Verify HTML structure
    assert "<!DOCTYPE html>" in content
    assert "<title>AUTOCOG Trajectory Dashboard</title>" in content
    assert "const DATA =" in content

    # Verify data is embedded (not the placeholder)
    assert "/*__DATA__*/" not in content


def test_generate_creates_parent_dirs(sample_dataset, tmp_path):
    """Test that generate() creates parent directories if needed."""
    generator = DashboardGenerator(sample_dataset)
    output_path = tmp_path / "nested" / "dir" / "dashboard.html"

    generator.generate(str(output_path))

    assert output_path.exists()
    assert output_path.parent.exists()


def test_archetype_color_fallback(sample_dataset):
    """Test unknown archetype gets white color."""
    # Add snapshot with unknown archetype
    unknown_snap = AgentSnapshot(
        tick=0,
        agent_id="agent-3",
        agent_name="Charlie",
        archetype="unknown_type",
        position=(0, 0),
        alive=True,
        hunger=50.0,
        thirst=50.0,
        energy=50.0,
        health=50.0,
        traits={},
        sensation_summary={},
        reflection={},
        intention={},
        action_type="wait",
        action_target=None,
        action_target_agent=None,
        action_succeeded=True,
        needs_delta={},
        inventory={},
    )
    sample_dataset.agent_snapshots.append(unknown_snap)

    generator = DashboardGenerator(sample_dataset)
    agents = generator._agent_metadata()

    charlie = next(a for a in agents if a["name"] == "Charlie")
    assert charlie["color"] == "#ffffff"  # Fallback white


def test_empty_dataset_handling():
    """Test handling of dataset with no snapshots."""
    metadata = RunMetadata(
        run_id="empty-run",
        timestamp="2026-01-01T00:00:00Z",
        seed=1,
        config={"world_width": 10, "world_height": 10},
        num_agents=0,
        max_ticks=0,
        actual_ticks=0,
    )
    dataset = TrajectoryDataset(metadata=metadata)

    generator = DashboardGenerator(dataset)
    data = generator._prepare_data()

    assert data["metadata"]["num_agents"] == 0
    assert len(data["agents"]) == 0
    assert len(data["ticks"]) == 0
    assert len(data["emergence_events"]) == 0


def test_dead_agent_in_snapshot(sample_dataset):
    """Test handling of dead agents in snapshots."""
    dead_snap = AgentSnapshot(
        tick=2,
        agent_id="agent-1",
        agent_name="Alice",
        archetype="gatherer",
        position=(5, 6),
        alive=False,  # Agent died
        hunger=0.0,
        thirst=0.0,
        energy=0.0,
        health=0.0,
        traits={},
        sensation_summary={},
        reflection={},
        intention={},
        action_type="wait",
        action_target=None,
        action_target_agent=None,
        action_succeeded=False,
        needs_delta={},
        inventory={},
    )
    sample_dataset.agent_snapshots.append(dead_snap)

    generator = DashboardGenerator(sample_dataset)
    ticks = generator._compact_tick_data()

    dead_state = ticks[2][0]
    assert dead_state["alive"] is False
    assert dead_state["health"] == 0.0


def test_compact_tick_data_includes_cultural_fields(sample_dataset):
    """Test that cultural data is included when present on a snapshot."""
    # Set cultural data on first agent (agent-1) at tick 0
    alice_snap = sample_dataset.agent_snapshots[0]
    alice_snap.cultural_learning_style = "prestige"
    alice_snap.cultural_group_id = 1
    alice_snap.cultural_repertoire = {
        "low_hunger:gather": {"adopted": True, "context": "low_hunger", "action": "gather"},
        "danger:flee": {"adopted": False},
    }
    alice_snap.transmission_events_this_tick = [
        {
            "variant_id": "low_hunger:gather",
            "actor_id": "agent-2",
            "bias_type": "prestige",
            "adopted": True,
            "probability": 0.75,
        }
    ]

    generator = DashboardGenerator(sample_dataset)
    ticks = generator._compact_tick_data()

    # Find agent-1 in tick 0
    alice_t0 = next(s for s in ticks[0] if s["id"] == "agent-1")

    assert "cultural" in alice_t0
    assert alice_t0["cultural"]["group"] == 1
    assert alice_t0["cultural"]["style"] == "prestige"
    assert alice_t0["cultural"]["repertoire_size"] == 2
    assert alice_t0["cultural"]["adopted_count"] == 1  # only low_hunger:gather has adopted=True
    assert len(alice_t0["cultural"]["events"]) == 1


def test_compact_tick_data_omits_cultural_when_absent(sample_dataset):
    """Test that cultural key is not present when no cultural data exists."""
    generator = DashboardGenerator(sample_dataset)
    ticks = generator._compact_tick_data()

    # Check all agents at tick 0 have no cultural key
    for agent_state in ticks[0]:
        assert "cultural" not in agent_state


def test_cultural_summary_empty_without_cultural_data(sample_dataset):
    """Test that cultural_summary returns empty dict when no cultural data exists."""
    generator = DashboardGenerator(sample_dataset)
    summary = generator._cultural_summary()

    assert summary == {}


def test_cultural_summary_aggregates_correctly(sample_dataset):
    """Test that cultural_summary correctly aggregates data from multiple agents."""
    # Set cultural data on both agents at tick 0
    alice_snap = sample_dataset.agent_snapshots[0]  # agent-1
    bob_snap = sample_dataset.agent_snapshots[1]  # agent-2

    alice_snap.cultural_learning_style = "prestige"
    alice_snap.cultural_group_id = 0
    alice_snap.cultural_repertoire = {"a:b": {"adopted": True}}

    bob_snap.cultural_learning_style = "conformist"
    bob_snap.cultural_group_id = 0
    bob_snap.cultural_repertoire = {"a:b": {"adopted": True}, "c:d": {"adopted": False}}

    generator = DashboardGenerator(sample_dataset)
    summary = generator._cultural_summary()

    assert 0 in summary
    assert summary[0]["group_count"] == 1  # both in group 0
    assert summary[0]["total_adopted"] == 2  # one from each agent
    assert summary[0]["total_variants"] == 3  # 1 + 2 repertoire entries
    assert summary[0]["style_distribution"] == {"prestige": 1, "conformist": 1}


def test_prepare_data_includes_cultural_summary(sample_dataset):
    """Test that _prepare_data includes cultural_summary key."""
    generator = DashboardGenerator(sample_dataset)
    data = generator._prepare_data()

    assert "cultural_summary" in data
    assert data["cultural_summary"] == {}  # no cultural data in sample_dataset


def test_cultural_data_survives_json_roundtrip(sample_dataset):
    """Test that cultural data survives JSON serialization roundtrip."""
    # Set cultural data on first agent
    alice_snap = sample_dataset.agent_snapshots[0]
    alice_snap.cultural_learning_style = "prestige"
    alice_snap.cultural_group_id = 1
    alice_snap.cultural_repertoire = {"low_hunger:gather": {"adopted": True}}
    alice_snap.transmission_events_this_tick = []

    generator = DashboardGenerator(sample_dataset)
    data = generator._prepare_data()

    # JSON roundtrip
    json_str = json.dumps(data)
    recovered_data = json.loads(json_str)

    # Find agent-1 in tick 0
    alice_t0 = next(s for s in recovered_data["ticks"]["0"] if s["id"] == "agent-1")

    assert "cultural" in alice_t0
    assert alice_t0["cultural"]["group"] == 1
    assert alice_t0["cultural"]["style"] == "prestige"


def test_dashboard_generates_without_cultural_data(sample_dataset, tmp_path):
    """Test that dashboard generates successfully without cultural data."""
    generator = DashboardGenerator(sample_dataset)
    output_path = tmp_path / "dashboard_no_cultural.html"

    generator.generate(str(output_path))

    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content


def test_dashboard_generates_with_cultural_data(sample_dataset, tmp_path):
    """Test that dashboard generates successfully with cultural data embedded."""
    # Set cultural data on first agent
    alice_snap = sample_dataset.agent_snapshots[0]
    alice_snap.cultural_learning_style = "prestige"
    alice_snap.cultural_group_id = 1
    alice_snap.cultural_repertoire = {"low_hunger:gather": {"adopted": True}}
    alice_snap.transmission_events_this_tick = []

    generator = DashboardGenerator(sample_dataset)
    output_path = tmp_path / "dashboard_with_cultural.html"

    generator.generate(str(output_path))

    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "cultural" in content  # Should appear in the embedded JSON


class TestDashboardTemplateAndGeneration:
    """Test dashboard template loading and HTML generation."""

    def test_dashboard_generator_load_template_exists(self):
        """Template file should exist and be loadable."""

        # Get template path
        dashboard_module = Path(__file__).parent.parent / "src" / "visualization" / "dashboard.py"
        template_path = dashboard_module.parent / "templates" / "dashboard.html"

        assert template_path.exists(), f"Template not found at {template_path}"

        # Verify it's readable
        content = template_path.read_text(encoding="utf-8")
        assert len(content) > 0
        assert "/*__DATA__*/" in content  # Should have data placeholder

    def test_dashboard_generator_produces_valid_html(self, sample_dataset, tmp_path):
        """Generated HTML should be valid and complete."""
        generator = DashboardGenerator(sample_dataset)
        output_path = tmp_path / "test_dashboard.html"

        generator.generate(str(output_path))

        content = output_path.read_text(encoding="utf-8")

        # Verify HTML structure
        assert content.startswith("<!DOCTYPE html>") or "<html" in content[:100]
        assert "</html>" in content
        assert "<head>" in content
        assert "<body>" in content
        assert "</body>" in content

        # Verify JavaScript is present
        assert "<script>" in content
        assert "const DATA =" in content

    def test_dashboard_data_contains_all_sections(self, sample_dataset, tmp_path):
        """Embedded data should contain all required sections."""
        generator = DashboardGenerator(sample_dataset)
        output_path = tmp_path / "dashboard.html"

        generator.generate(str(output_path))

        content = output_path.read_text(encoding="utf-8")

        # Extract the embedded JSON data
        # Look for: const DATA = {...};
        start_marker = "const DATA ="
        start_idx = content.find(start_marker)
        assert start_idx != -1, "DATA constant not found in HTML"

        # Find the JSON object (simple approach - look for opening brace after DATA =)
        json_start = content.find("{", start_idx)
        assert json_start != -1

        # Verify key sections are present in the content
        assert '"metadata"' in content
        assert '"agents"' in content
        assert '"ticks"' in content
        assert '"emergence_events"' in content
        assert '"run_id"' in content
        assert sample_dataset.metadata.run_id in content

    def test_dashboard_from_trajectory_file(self, tmp_path):
        """DashboardGenerator.from_file() should load and generate dashboard."""
        from src.trajectory.exporter import TrajectoryExporter
        from src.trajectory.schema import AgentSnapshot, RunMetadata, TrajectoryDataset

        # Create a simple dataset
        metadata = RunMetadata(
            run_id="test-from-file",
            timestamp="2026-02-22T00:00:00Z",
            seed=123,
            config={"world_width": 16, "world_height": 16},
            num_agents=1,
            max_ticks=5,
            actual_ticks=2,
        )

        snapshots = [
            AgentSnapshot(
                tick=0,
                agent_id="agent-1",
                agent_name="Test",
                archetype="survivalist",
                position=(5, 5),
                alive=True,
                hunger=50.0,
                thirst=50.0,
                energy=50.0,
                health=100.0,
                traits={},
                sensation_summary={},
                reflection={},
                intention={},
                action_type="wait",
                action_target=None,
                action_target_agent=None,
                action_succeeded=True,
                needs_delta={},
                inventory={},
            ),
            AgentSnapshot(
                tick=1,
                agent_id="agent-1",
                agent_name="Test",
                archetype="survivalist",
                position=(5, 6),
                alive=True,
                hunger=49.0,
                thirst=49.0,
                energy=49.0,
                health=100.0,
                traits={},
                sensation_summary={},
                reflection={},
                intention={},
                action_type="move",
                action_target=None,
                action_target_agent=None,
                action_succeeded=True,
                needs_delta={},
                inventory={},
            ),
        ]

        dataset = TrajectoryDataset(metadata=metadata, agent_snapshots=snapshots)

        # Export to JSONL
        trajectory_path = tmp_path / "trajectory.jsonl"
        TrajectoryExporter.to_jsonl(dataset, str(trajectory_path))

        # Generate dashboard from file
        generator = DashboardGenerator.from_file(str(trajectory_path))
        output_path = tmp_path / "dashboard.html"
        generator.generate(str(output_path))

        # Verify it was created
        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert "test-from-file" in content
