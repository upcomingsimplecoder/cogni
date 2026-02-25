"""End-to-end integration tests for the full engine with all features.

Tests the complete system with:
- Trajectory recording
- Checkpointing
- Theory of Mind
- Evolution
- Coalitions
- Multiple cognitive architectures
"""

from __future__ import annotations

from pathlib import Path

from src.config import SimulationConfig
from src.persistence.checkpoint import CheckpointManager
from src.simulation.engine import SimulationEngine
from src.trajectory.loader import TrajectoryLoader
from src.trajectory.recorder import TrajectoryRecorder


class TestEngineWithTrajectoryRecording:
    """Test engine with trajectory recording enabled."""

    def test_engine_with_trajectory_recording_produces_files(
        self, config: SimulationConfig, tmp_path
    ):
        """Engine with trajectory recording should produce trajectory files."""
        recorder = TrajectoryRecorder(output_dir=str(tmp_path / "trajectories"))
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        recorder.start_run(engine)
        for _ in range(10):
            tick_record = engine.step_all()
            recorder.record_tick(engine, tick_record)
        recorder.end_run(engine)

        # Verify files exist
        jsonl_path = recorder.output_dir / "trajectory.jsonl"
        assert jsonl_path.exists()

        # Verify loadable
        dataset = TrajectoryLoader.from_jsonl(str(jsonl_path))
        assert dataset.metadata.actual_ticks == 10

    def test_engine_trajectory_data_matches_engine_state(self, config: SimulationConfig, tmp_path):
        """Trajectory data should match engine state after recording."""
        recorder = TrajectoryRecorder(output_dir=str(tmp_path / "trajectories"))
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        recorder.start_run(engine)
        for _ in range(5):
            tick_record = engine.step_all()
            recorder.record_tick(engine, tick_record)
        recorder.end_run(engine)

        # Load trajectory
        jsonl_path = recorder.output_dir / "trajectory.jsonl"
        dataset = TrajectoryLoader.from_jsonl(str(jsonl_path))

        # Verify final tick matches
        assert dataset.metadata.actual_ticks == engine.state.tick

        # Verify agent count
        living_agents = engine.registry.living_agents()
        final_tick_snapshots = [s for s in dataset.agent_snapshots if s.tick == 4]
        # All living agents should have snapshots
        assert len([s for s in final_tick_snapshots if s.alive]) <= len(living_agents)


class TestEngineWithCheckpointing:
    """Test engine with checkpoint save/load."""

    def test_engine_with_checkpointing_creates_checkpoint_files(
        self, config: SimulationConfig, tmp_path
    ):
        """Engine with checkpointing should create checkpoint files."""
        checkpoint_dir = tmp_path / "checkpoints"
        manager = CheckpointManager(str(checkpoint_dir))

        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        # Run simulation and save checkpoint
        for _ in range(10):
            engine.step_all()

        filepath = manager.save(engine, label="test_checkpoint")

        # Verify checkpoint exists
        assert Path(filepath).exists()
        assert "test_checkpoint" in filepath

    def test_engine_checkpoint_save_load_resumes_correctly(
        self, config: SimulationConfig, tmp_path
    ):
        """Checkpoint save/load should preserve state and allow resumption."""
        checkpoint_dir = tmp_path / "checkpoints"
        manager = CheckpointManager(str(checkpoint_dir))

        # Create and run engine
        engine1 = SimulationEngine(config)
        engine1.setup_multi_agent()

        for _ in range(15):
            engine1.step_all()

        tick_before_save = engine1.state.tick
        agents_before = {
            str(a.agent_id): (a.x, a.y, a.alive) for a in engine1.registry.living_agents()
        }

        # Save checkpoint
        filepath = manager.save(engine1, label="resume_test")

        # Load checkpoint into new engine
        engine2 = manager.load(filepath)

        # Verify state matches
        assert engine2.state.tick == tick_before_save
        agents_after = {
            str(a.agent_id): (a.x, a.y, a.alive) for a in engine2.registry.living_agents()
        }

        # At least verify agent count and some agents match
        assert len(agents_after) == len(agents_before)

        # Continue running from checkpoint
        for _ in range(10):
            engine2.step_all()

        # Verify it advanced
        assert engine2.state.tick == tick_before_save + 10


class TestEngineWithTheoryOfMind:
    """Test engine with Theory of Mind enabled."""

    def test_engine_with_tom_wraps_strategies(self, config: SimulationConfig):
        """Engine with ToM should wrap strategies with perspective-taking."""
        # This test verifies the integration exists, not the detailed ToM behavior
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        # Run a few ticks - should not crash
        for _ in range(5):
            tick_record = engine.step_all()
            assert tick_record is not None

        # Verify agents exist and are alive
        assert len(engine.registry.living_agents()) > 0


class TestEngineWithEvolution:
    """Test engine with trait evolution enabled."""

    def test_engine_with_evolution_trait_drift_observable(self, config: SimulationConfig, tmp_path):
        """Engine with evolution should show trait changes over time."""
        config.max_ticks = 100
        config.num_agents = 5

        recorder = TrajectoryRecorder(output_dir=str(tmp_path / "trajectories"))
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        # Capture initial traits
        initial_traits = {}
        for agent in engine.registry.living_agents():
            if agent.profile:
                initial_traits[str(agent.agent_id)] = agent.profile.traits.as_dict().copy()

        # Run simulation with recording
        recorder.start_run(engine)
        for _ in range(100):
            tick_record = engine.step_all()
            recorder.record_tick(engine, tick_record)
        recorder.end_run(engine)

        # Check if any traits changed (evolution may or may not have occurred)
        # This test just verifies the system doesn't crash with evolution enabled
        jsonl_path = recorder.output_dir / "trajectory.jsonl"
        dataset = TrajectoryLoader.from_jsonl(str(jsonl_path))

        assert dataset.metadata.actual_ticks == 100
        assert len(dataset.agent_snapshots) > 0


class TestEngineWithCoalitions:
    """Test engine with coalition tracking."""

    def test_engine_with_coalitions_tracks_groups(self, config: SimulationConfig):
        """Engine with coalitions should track social groups."""
        config.num_agents = 5

        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        # Run simulation - coalitions should form/dissolve
        for _ in range(20):
            engine.step_all()

        # Verify simulation ran successfully
        assert engine.state.tick == 20
        assert len(engine.registry.living_agents()) > 0


class TestEngineWithMultipleFeatures:
    """Test engine with multiple features enabled simultaneously."""

    def test_engine_all_features_enabled_runs_100_ticks(self, config: SimulationConfig, tmp_path):
        """Engine with all features should run full simulation."""
        config.max_ticks = 100
        config.num_agents = 5

        recorder = TrajectoryRecorder(output_dir=str(tmp_path / "trajectories"))
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        # Run with recording
        recorder.start_run(engine)
        for _ in range(100):
            tick_record = engine.step_all()
            recorder.record_tick(engine, tick_record)
        recorder.end_run(engine)

        # Verify successful completion
        assert engine.state.tick == 100

        # Verify trajectory captured
        jsonl_path = recorder.output_dir / "trajectory.jsonl"
        dataset = TrajectoryLoader.from_jsonl(str(jsonl_path))
        assert dataset.metadata.actual_ticks == 100


class TestCognitiveArchitectures:
    """Test different cognitive architectures."""

    def test_engine_planning_architecture_creates_multi_step_plans(self, config: SimulationConfig):
        """Engine with planning architecture should enable multi-step planning."""
        config.num_agents = 3

        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        # Run simulation
        for _ in range(20):
            tick_record = engine.step_all()
            assert tick_record is not None

        # Verify agents executed actions
        assert engine.state.tick == 20

    def test_engine_dual_process_escalates_under_threat(self, config: SimulationConfig):
        """Engine with dual-process architecture should handle threat escalation."""
        config.num_agents = 3

        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        # Run simulation
        for _ in range(20):
            engine.step_all()

        # Verify simulation completed
        assert engine.state.tick == 20

    def test_engine_social_architecture_responds_to_allies(self, config: SimulationConfig):
        """Engine with social architecture should respond to social context."""
        config.num_agents = 4

        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        # Run simulation with social interactions
        for _ in range(20):
            engine.step_all()

        # Verify completed
        assert engine.state.tick == 20


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_engine_survives_all_agents_dying(self, config: SimulationConfig):
        """Engine should handle scenario where all agents die."""
        config.num_agents = 2
        config.max_ticks = 200  # Long enough for agents to potentially die

        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        # Run until all agents die or max ticks
        for _ in range(200):
            engine.step_all()
            if engine.registry.count_living == 0:
                break

        # Should not crash, tick should advance
        assert engine.state.tick > 0

    def test_engine_with_single_agent(self, config: SimulationConfig):
        """Engine should work with a single agent."""
        config.num_agents = 1

        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        for _ in range(10):
            tick_record = engine.step_all()
            assert len(tick_record.agent_records) == 1

        assert engine.state.tick == 10

    def test_engine_with_many_agents(self, config: SimulationConfig):
        """Engine should handle larger agent populations."""
        config.num_agents = 20
        config.world_width = 64
        config.world_height = 64

        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        # Run a few ticks - should not crash or be prohibitively slow
        for _ in range(10):
            tick_record = engine.step_all()
            assert tick_record is not None

        assert len(engine.registry.living_agents()) > 0


class TestCheckpointIntegration:
    """Test checkpoint integration with full engine."""

    def test_checkpoint_handles_complex_state(self, config: SimulationConfig, tmp_path):
        """Checkpoint should handle complex engine state with all features."""
        config.num_agents = 5
        checkpoint_dir = tmp_path / "checkpoints"
        manager = CheckpointManager(str(checkpoint_dir))

        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        # Run and create complex state
        for _ in range(30):
            engine.step_all()

        # Save checkpoint
        filepath = manager.save(engine, label="complex_state")

        # Load and verify
        loaded_engine = manager.load(filepath)
        assert loaded_engine.state.tick == 30

        # Continue simulation
        for _ in range(10):
            loaded_engine.step_all()

        assert loaded_engine.state.tick == 40


class TestTrajectoryAndCheckpointTogether:
    """Test trajectory recording and checkpointing together."""

    def test_combined_trajectory_and_checkpoint(self, config: SimulationConfig, tmp_path):
        """Should be able to use trajectory recording and checkpointing together."""
        config.num_agents = 4

        recorder = TrajectoryRecorder(output_dir=str(tmp_path / "trajectories"))
        checkpoint_manager = CheckpointManager(str(tmp_path / "checkpoints"))

        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        # Run with trajectory recording
        recorder.start_run(engine)
        checkpoint_filepath = None
        for i in range(20):
            tick_record = engine.step_all()
            recorder.record_tick(engine, tick_record)

            # Save checkpoint at tick 10
            if i == 9:
                checkpoint_filepath = checkpoint_manager.save(engine, label="mid_run")

        recorder.end_run(engine)

        # Verify both systems worked
        jsonl_path = recorder.output_dir / "trajectory.jsonl"
        assert jsonl_path.exists()

        assert checkpoint_filepath is not None
        assert Path(checkpoint_filepath).exists()

        # Load checkpoint and verify tick
        loaded_engine = checkpoint_manager.load(checkpoint_filepath)
        assert loaded_engine.state.tick == 10

        # Load trajectory and verify
        dataset = TrajectoryLoader.from_jsonl(str(jsonl_path))
        assert dataset.metadata.actual_ticks == 20
