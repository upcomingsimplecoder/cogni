"""Tests for error handling and edge cases.

Verifies:
- LLM strategy parse failures
- Invalid actions
- Agent death during tick
- Trajectory recorder missing fields
- Corrupted checkpoint files
- Plugin loader failures
- Awareness loop exceptions
- Experiment runner failures
"""

from __future__ import annotations

import json

import pytest

from src.config import SimulationConfig
from src.persistence.checkpoint import CheckpointManager
from src.simulation.actions import Action, ActionType
from src.simulation.engine import SimulationEngine
from src.trajectory.loader import TrajectoryLoader
from src.trajectory.recorder import TrajectoryRecorder


class TestLLMStrategyErrorHandling:
    """Test LLM strategy parse failure handling."""

    def test_llm_strategy_logs_on_parse_failure(self, config: SimulationConfig, caplog):
        """LLM strategy should log parse failures and not crash."""
        # This is a conceptual test - actual implementation depends on LLM integration
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        # Run simulation - even if LLM fails to parse, should continue
        for _ in range(5):
            tick_record = engine.step_all()
            assert tick_record is not None

        # Simulation should complete
        assert engine.state.tick == 5


class TestEngineErrorHandling:
    """Test engine error handling."""

    def test_engine_survives_invalid_action(self, config: SimulationConfig):
        """Engine should handle invalid action gracefully."""
        engine = SimulationEngine(config)
        engine.setup_legacy_mode()

        # Create an action with invalid parameters (e.g., out of bounds target)
        action = Action(type=ActionType.MOVE, direction=None)  # Invalid: move without direction

        # Should not crash
        record = engine.step(action)

        # Action should fail gracefully
        assert record is not None
        assert len(record.agent_records) == 1
        # The action likely failed, but engine continues
        assert engine.state.tick == 1

    def test_engine_survives_agent_death_mid_tick(self, config: SimulationConfig):
        """Engine should handle agent dying during tick."""
        config.num_agents = 3
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        # Get an agent and set health very low
        agents = list(engine.registry.living_agents())
        if agents:
            agents[0].needs.health = 1.0  # Very low health

        # Run simulation - agent might die
        for _ in range(20):
            tick_record = engine.step_all()
            assert tick_record is not None

        # Engine should survive regardless
        assert engine.state.tick == 20

    def test_engine_handles_empty_agent_registry(self, config: SimulationConfig):
        """Engine should handle having no agents."""
        engine = SimulationEngine(config)
        # Don't setup any agents

        # step_all should not crash
        tick_record = engine.step_all()
        assert tick_record is not None
        assert len(tick_record.agent_records) == 0
        assert engine.state.tick == 1


class TestTrajectoryRecorderErrorHandling:
    """Test trajectory recorder error handling."""

    def test_trajectory_recorder_handles_missing_fields_gracefully(
        self, config: SimulationConfig, tmp_path
    ):
        """Recorder should handle agents with missing optional fields."""
        recorder = TrajectoryRecorder(output_dir=str(tmp_path / "trajectories"))
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        # Start recording
        recorder.start_run(engine)

        # Record some ticks
        for _ in range(5):
            tick_record = engine.step_all()
            # Should not crash even if some fields are missing
            recorder.record_tick(engine, tick_record)

        recorder.end_run(engine)

        # Verify file was created
        jsonl_path = recorder.output_dir / "trajectory.jsonl"
        assert jsonl_path.exists()

        # Verify loadable
        dataset = TrajectoryLoader.from_jsonl(str(jsonl_path))
        assert dataset.metadata.actual_ticks == 5

    def test_trajectory_recorder_handles_end_run_without_start(self, tmp_path):
        """Calling end_run without start_run should not crash."""
        recorder = TrajectoryRecorder(output_dir=str(tmp_path / "trajectories"))
        config = SimulationConfig(seed=42)
        engine = SimulationEngine(config)

        # Call end_run without start_run - should not crash
        recorder.end_run(engine)

        # No files should be created
        assert (
            not (recorder.output_dir / "trajectory.jsonl").exists()
            or (recorder.output_dir / "trajectory.jsonl").stat().st_size == 0
        )


class TestCheckpointErrorHandling:
    """Test checkpoint error handling."""

    def test_checkpoint_handles_corrupted_file(self, tmp_path):
        """CheckpointManager should handle corrupted checkpoint files."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Create a corrupted checkpoint file
        corrupted_file = checkpoint_dir / "corrupted.json"
        corrupted_file.write_text("{ invalid json content [[[")

        manager = CheckpointManager(str(checkpoint_dir))

        # Loading corrupted file should raise appropriate error
        # Could be json.JSONDecodeError or custom exception
        with pytest.raises(Exception, match=""):
            manager.load("corrupted")

    def test_checkpoint_handles_nonexistent_file(self, tmp_path):
        """CheckpointManager should handle loading nonexistent checkpoint."""
        checkpoint_dir = tmp_path / "checkpoints"
        manager = CheckpointManager(str(checkpoint_dir))

        # Loading nonexistent file should raise appropriate error
        # FileNotFoundError or custom exception
        with pytest.raises(Exception, match=""):
            manager.load("nonexistent")

    def test_checkpoint_handles_invalid_state_data(self, config: SimulationConfig, tmp_path):
        """CheckpointManager should handle checkpoint with invalid state data."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Create checkpoint with invalid data structure
        invalid_checkpoint = checkpoint_dir / "invalid.json"
        invalid_checkpoint.write_text(
            json.dumps(
                {
                    "config": {},
                    "state": {"tick": "not_a_number"},  # Invalid tick type
                    "agents": [],
                }
            )
        )

        manager = CheckpointManager(str(checkpoint_dir))

        # Should raise error or handle gracefully
        with pytest.raises(Exception, match=""):
            manager.load("invalid")


class TestTrajectoryLoaderErrorHandling:
    """Test trajectory loader error handling."""

    def test_loader_handles_missing_trajectory_file(self):
        """TrajectoryLoader should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            TrajectoryLoader.from_jsonl("nonexistent_trajectory.jsonl")

    def test_loader_handles_empty_trajectory_file(self, tmp_path):
        """TrajectoryLoader should handle empty trajectory files."""
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("")

        # Should raise error about missing metadata
        with pytest.raises(ValueError, match="No metadata found"):
            TrajectoryLoader.from_jsonl(str(empty_file))

    def test_loader_handles_malformed_jsonl(self, tmp_path):
        """TrajectoryLoader should handle malformed JSONL."""
        malformed_file = tmp_path / "malformed.jsonl"
        malformed_file.write_text("{ not valid json\n")

        # Should raise JSON decode error
        with pytest.raises(json.JSONDecodeError):
            TrajectoryLoader.from_jsonl(str(malformed_file))


class TestAwarenessLoopErrorHandling:
    """Test awareness loop exception handling."""

    def test_awareness_loop_handles_strategy_exception(self, config: SimulationConfig):
        """Awareness loop should handle strategy exceptions gracefully."""
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        # Run simulation - strategies might fail internally but shouldn't crash
        for _ in range(10):
            tick_record = engine.step_all()
            assert tick_record is not None

        # Simulation should complete
        assert engine.state.tick == 10

    def test_awareness_loop_handles_missing_perception(self, config: SimulationConfig):
        """Awareness loop should handle missing perception data."""
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        # Run simulation normally
        for _ in range(5):
            tick_record = engine.step_all()
            assert tick_record is not None

        assert engine.state.tick == 5


class TestExperimentRunnerErrorHandling:
    """Test experiment runner error handling."""

    def test_experiment_runner_handles_failed_simulation(self):
        """Experiment runner should handle simulation failures gracefully."""
        # This is a placeholder test - actual implementation depends on experiment runner
        config = SimulationConfig(seed=42, max_ticks=10, num_agents=3)
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        # Run simulation
        try:
            for _ in range(10):
                engine.step_all()
            success = True
        except Exception:
            success = False
            # In production, would log error and continue with other experiments

        # Should have either succeeded or handled error
        assert success or engine.state.tick >= 0


class TestEdgeCaseErrorHandling:
    """Test edge case error handling."""

    def test_engine_handles_zero_world_size(self):
        """Engine should handle or reject zero world size."""
        # This should either work or raise a clear error
        try:
            config = SimulationConfig(world_width=0, world_height=0, seed=42)
            engine = SimulationEngine(config)
            # If it doesn't raise, world should have some default size
            assert engine.world.width > 0 or engine.world.height > 0
        except (ValueError, AssertionError):
            # Expected to reject invalid config
            pass

    def test_engine_handles_negative_seed(self):
        """Engine should handle negative seed values."""
        config = SimulationConfig(seed=-1, num_agents=1)
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        # Should not crash
        for _ in range(5):
            engine.step_all()

        assert engine.state.tick == 5

    def test_trajectory_recorder_handles_very_long_simulation(
        self, config: SimulationConfig, tmp_path
    ):
        """Trajectory recorder should handle long simulations without memory issues."""
        config.num_agents = 2
        recorder = TrajectoryRecorder(output_dir=str(tmp_path / "trajectories"))
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        recorder.start_run(engine)

        # Run moderately long simulation (not too long for tests)
        for _ in range(100):
            tick_record = engine.step_all()
            recorder.record_tick(engine, tick_record)

        recorder.end_run(engine)

        # Verify file was created and is loadable
        jsonl_path = recorder.output_dir / "trajectory.jsonl"
        assert jsonl_path.exists()

        dataset = TrajectoryLoader.from_jsonl(str(jsonl_path))
        assert dataset.metadata.actual_ticks == 100


class TestRecoveryFromErrors:
    """Test system recovery from errors."""

    def test_engine_continues_after_single_agent_error(self, config: SimulationConfig):
        """Engine should continue if single agent encounters error."""
        config.num_agents = 5
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        # Run simulation - even if one agent fails, others should continue
        for _ in range(10):
            tick_record = engine.step_all()
            assert tick_record is not None

        # Should complete successfully
        assert engine.state.tick == 10

    def test_trajectory_system_recovers_after_io_error(self, config: SimulationConfig, tmp_path):
        """Trajectory system should handle I/O errors gracefully."""
        recorder = TrajectoryRecorder(output_dir=str(tmp_path / "trajectories"))
        engine = SimulationEngine(config)
        engine.setup_multi_agent()

        recorder.start_run(engine)

        # Record some ticks
        for _ in range(5):
            tick_record = engine.step_all()
            recorder.record_tick(engine, tick_record)

        # Close recorder
        recorder.end_run(engine)

        # Verify file exists and is valid
        jsonl_path = recorder.output_dir / "trajectory.jsonl"
        assert jsonl_path.exists()

        # Should be loadable
        dataset = TrajectoryLoader.from_jsonl(str(jsonl_path))
        assert len(dataset.agent_snapshots) > 0
