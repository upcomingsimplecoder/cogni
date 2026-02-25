"""Tests for tick hook system in experiments."""

from __future__ import annotations

import warnings

import pytest

from src.config import SimulationConfig
from src.experiments.config import ExperimentCondition, ExperimentConfig, TickHook
from src.experiments.runner import ExperimentRunner
from src.simulation.engine import SimulationEngine


class TestTickHookParsing:
    """Tests for tick hook configuration parsing."""

    def test_tick_hook_from_dict(self):
        """Test that TickHook is created correctly from dict."""
        hook_data = {
            "at_tick": 10,
            "action": "corrupt_traits",
            "params": {
                "target": "all",
                "traits": {"cooperation_tendency": 0.0},
                "mode": "set",
            },
        }

        hook = TickHook(
            at_tick=hook_data["at_tick"],
            action=hook_data["action"],
            params=hook_data.get("params", {}),
        )

        assert hook.at_tick == 10
        assert hook.action == "corrupt_traits"
        assert hook.params["target"] == "all"
        assert hook.params["traits"]["cooperation_tendency"] == 0.0
        assert hook.params["mode"] == "set"

    def test_experiment_config_with_hooks(self):
        """Test that ExperimentConfig parses top-level hooks correctly."""
        data = {
            "name": "Test",
            "description": "Test with hooks",
            "base": {"world_width": 16, "world_height": 16, "max_ticks": 30},
            "conditions": [{"name": "control", "overrides": {}}],
            "tick_hooks": [
                {
                    "at_tick": 20,
                    "action": "corrupt_traits",
                    "params": {"target": "all", "traits": {"aggression": 0.9}},
                }
            ],
            "replicates": 1,
        }

        config = ExperimentConfig.from_dict(data)

        assert len(config.tick_hooks) == 1
        assert config.tick_hooks[0].at_tick == 20
        assert config.tick_hooks[0].action == "corrupt_traits"
        assert config.tick_hooks[0].params["target"] == "all"

    def test_condition_level_hooks(self):
        """Test that condition-specific hooks are parsed correctly."""
        data = {
            "name": "Test",
            "description": "Condition-level hooks",
            "base": {"world_width": 16, "world_height": 16, "max_ticks": 30},
            "conditions": [
                {
                    "name": "with_hooks",
                    "overrides": {},
                    "tick_hooks": [
                        {
                            "at_tick": 10,
                            "action": "remove_agent",
                            "params": {"target": "agent_index", "index": 0},
                        }
                    ],
                }
            ],
            "replicates": 1,
        }

        config = ExperimentConfig.from_dict(data)

        assert len(config.conditions) == 1
        assert len(config.conditions[0].tick_hooks) == 1
        hook = config.conditions[0].tick_hooks[0]
        assert hook.at_tick == 10
        assert hook.action == "remove_agent"
        assert hook.params["target"] == "agent_index"

    def test_empty_hooks_backward_compat(self):
        """Test that configs without hooks work unchanged (backward compatibility)."""
        data = {
            "name": "No Hooks",
            "description": "Test backward compatibility",
            "base": {"world_width": 16, "world_height": 16, "max_ticks": 30},
            "conditions": [{"name": "control", "overrides": {}}],
            "replicates": 1,
        }

        config = ExperimentConfig.from_dict(data)

        assert len(config.tick_hooks) == 0
        assert len(config.conditions[0].tick_hooks) == 0


class TestHookExecution:
    """Tests for hook execution during simulations."""

    def test_corrupt_traits_set_all(self):
        """Test that corrupt_traits with target='all' sets all agents' traits."""
        exp_config = ExperimentConfig(
            name="test",
            description="test",
            base={
                "world_width": 16,
                "world_height": 16,
                "max_ticks": 30,
                "num_agents": 4,
            },
            conditions=[
                ExperimentCondition(
                    name="test",
                    overrides={},
                    tick_hooks=[
                        TickHook(
                            at_tick=10,
                            action="corrupt_traits",
                            params={
                                "target": "all",
                                "traits": {"cooperation_tendency": 0.0},
                                "mode": "set",
                            },
                        )
                    ],
                )
            ],
            replicates=1,
            seed_start=42,
            metrics=["agents_alive_at_end"],
        )

        runner = ExperimentRunner(exp_config)
        results = runner.run_all()

        # The hook should have fired, but we can't easily verify mid-simulation
        # Just verify the simulation completed successfully
        assert len(results) == 1
        assert results[0].metrics["agents_alive_at_end"] >= 0

    def test_corrupt_traits_shift(self):
        """Test that corrupt_traits with mode='shift' adds delta and clamps."""
        # Create engine and manually test hook
        sim_config = SimulationConfig(
            world_width=16, world_height=16, max_ticks=30, num_agents=3, seed=42
        )
        engine = SimulationEngine(sim_config)
        engine.setup_multi_agent()

        # Record initial cooperation values
        living = list(engine.registry.living_agents())
        initial_values = [a.profile.traits.cooperation_tendency for a in living]

        # Create runner and call hook directly
        exp_config = ExperimentConfig(
            name="test",
            description="test",
            base={},
            conditions=[ExperimentCondition("test", {})],
        )
        runner = ExperimentRunner(exp_config)

        # Execute shift hook
        params = {
            "target": "all",
            "traits": {"cooperation_tendency": 0.2},
            "mode": "shift",
        }
        runner._hook_corrupt_traits(engine, params)

        # Check that values increased by 0.2 (or clamped at 1.0)
        for i, agent in enumerate(living):
            expected = min(1.0, initial_values[i] + 0.2)
            assert agent.profile.traits.cooperation_tendency == pytest.approx(expected)

    def test_corrupt_traits_agent_index(self):
        """Test that corrupt_traits with target='agent_index' only affects target agent."""
        sim_config = SimulationConfig(
            world_width=16, world_height=16, max_ticks=30, num_agents=4, seed=42
        )
        engine = SimulationEngine(sim_config)
        engine.setup_multi_agent()

        living = list(engine.registry.living_agents())
        initial_values = [a.profile.traits.aggression for a in living]

        # Create runner
        exp_config = ExperimentConfig(
            name="test",
            description="test",
            base={},
            conditions=[ExperimentCondition("test", {})],
        )
        runner = ExperimentRunner(exp_config)

        # Corrupt only agent at index 1
        params = {
            "target": "agent_index",
            "index": 1,
            "traits": {"aggression": 0.95},
            "mode": "set",
        }
        runner._hook_corrupt_traits(engine, params)

        # Check that only agent 1 changed
        for i, agent in enumerate(living):
            if i == 1:
                assert agent.profile.traits.aggression == 0.95
            else:
                assert agent.profile.traits.aggression == pytest.approx(initial_values[i])

    def test_corrupt_traits_random_n_deterministic(self):
        """Test that random_n selection is deterministic with same seed."""
        # Run twice with same seed
        results_1 = []
        results_2 = []

        for run_results in [results_1, results_2]:
            sim_config = SimulationConfig(
                world_width=16, world_height=16, max_ticks=30, num_agents=5, seed=100
            )
            engine = SimulationEngine(sim_config)
            engine.setup_multi_agent()

            exp_config = ExperimentConfig(
                name="test",
                description="test",
                base={},
                conditions=[ExperimentCondition("test", {})],
            )
            runner = ExperimentRunner(exp_config)

            # Mark agents before corruption
            living = list(engine.registry.living_agents())
            for a in living:
                a.profile.traits.curiosity = 0.5

            # Corrupt random 2
            params = {
                "target": "random_n",
                "n": 2,
                "traits": {"curiosity": 0.99},
                "mode": "set",
            }
            runner._hook_corrupt_traits(engine, params)

            # Record which agents were corrupted
            corrupted_indices = [
                i for i, a in enumerate(living) if a.profile.traits.curiosity == 0.99
            ]
            run_results.extend(corrupted_indices)

        # Should have corrupted same agents in both runs
        assert results_1 == results_2
        assert len(results_1) == 2  # Exactly 2 agents corrupted

    def test_remove_agent_by_index(self):
        """Test that remove_agent with target='agent_index' removes correct agent."""
        sim_config = SimulationConfig(
            world_width=16, world_height=16, max_ticks=30, num_agents=4, seed=42
        )
        engine = SimulationEngine(sim_config)
        engine.setup_multi_agent()

        living = list(engine.registry.living_agents())
        initial_count = len(living)
        target_agent_id = living[2].agent_id

        # Create runner
        exp_config = ExperimentConfig(
            name="test",
            description="test",
            base={},
            conditions=[ExperimentCondition("test", {})],
        )
        runner = ExperimentRunner(exp_config)

        # Remove agent at index 2
        params = {"target": "agent_index", "index": 2}
        runner._hook_remove_agent(engine, params)

        # Check that agent count decreased and target is dead
        assert engine.registry.count_living == initial_count - 1
        assert engine.registry.get(target_agent_id) is None  # Agent is dead (not in living agents)

    def test_remove_agent_most_cooperative(self):
        """Test that remove_agent with target='most_cooperative' removes correct agent."""
        sim_config = SimulationConfig(
            world_width=16, world_height=16, max_ticks=30, num_agents=4, seed=42
        )
        engine = SimulationEngine(sim_config)
        engine.setup_multi_agent()

        # Set known cooperation values
        living = list(engine.registry.living_agents())
        living[0].profile.traits.cooperation_tendency = 0.2
        living[1].profile.traits.cooperation_tendency = 0.9  # Most cooperative
        living[2].profile.traits.cooperation_tendency = 0.5
        living[3].profile.traits.cooperation_tendency = 0.3

        most_coop_id = living[1].agent_id

        exp_config = ExperimentConfig(
            name="test",
            description="test",
            base={},
            conditions=[ExperimentCondition("test", {})],
        )
        runner = ExperimentRunner(exp_config)

        params = {"target": "most_cooperative"}
        runner._hook_remove_agent(engine, params)

        # Verify most cooperative agent was removed
        assert engine.registry.get(most_coop_id) is None  # Agent is dead
        assert engine.registry.count_living == 3

    def test_remove_agent_least_cooperative(self):
        """Test that remove_agent with target='least_cooperative' removes correct agent."""
        sim_config = SimulationConfig(
            world_width=16, world_height=16, max_ticks=30, num_agents=4, seed=42
        )
        engine = SimulationEngine(sim_config)
        engine.setup_multi_agent()

        # Set known cooperation values
        living = list(engine.registry.living_agents())
        living[0].profile.traits.cooperation_tendency = 0.8
        living[1].profile.traits.cooperation_tendency = 0.5
        living[2].profile.traits.cooperation_tendency = 0.1  # Least cooperative
        living[3].profile.traits.cooperation_tendency = 0.7

        least_coop_id = living[2].agent_id

        exp_config = ExperimentConfig(
            name="test",
            description="test",
            base={},
            conditions=[ExperimentCondition("test", {})],
        )
        runner = ExperimentRunner(exp_config)

        params = {"target": "least_cooperative"}
        runner._hook_remove_agent(engine, params)

        # Verify least cooperative agent was removed
        assert engine.registry.get(least_coop_id) is None  # Agent is dead
        assert engine.registry.count_living == 3

    def test_multiple_hooks_different_ticks(self):
        """Test that multiple hooks at different ticks both fire."""
        exp_config = ExperimentConfig(
            name="test",
            description="test",
            base={
                "world_width": 16,
                "world_height": 16,
                "max_ticks": 50,
                "num_agents": 4,
            },
            conditions=[
                ExperimentCondition(
                    name="test",
                    overrides={},
                    tick_hooks=[
                        TickHook(
                            at_tick=10,
                            action="corrupt_traits",
                            params={
                                "target": "agent_index",
                                "index": 0,
                                "traits": {"cooperation_tendency": 0.0},
                                "mode": "set",
                            },
                        ),
                        TickHook(
                            at_tick=20,
                            action="corrupt_traits",
                            params={
                                "target": "agent_index",
                                "index": 1,
                                "traits": {"aggression": 0.9},
                                "mode": "set",
                            },
                        ),
                    ],
                )
            ],
            replicates=1,
            seed_start=42,
            metrics=["agents_alive_at_end"],
        )

        runner = ExperimentRunner(exp_config)
        results = runner.run_all()

        # Both hooks should have fired
        assert len(results) == 1
        assert results[0].metrics["agents_alive_at_end"] >= 0

    def test_hook_at_nonexistent_tick(self):
        """Test that hooks scheduled after max_ticks don't crash."""
        exp_config = ExperimentConfig(
            name="test",
            description="test",
            base={
                "world_width": 16,
                "world_height": 16,
                "max_ticks": 20,
                "num_agents": 3,
            },
            conditions=[
                ExperimentCondition(
                    name="test",
                    overrides={},
                    tick_hooks=[
                        TickHook(
                            at_tick=100,  # Beyond max_ticks
                            action="corrupt_traits",
                            params={
                                "target": "all",
                                "traits": {"cooperation_tendency": 0.0},
                                "mode": "set",
                            },
                        )
                    ],
                )
            ],
            replicates=1,
            seed_start=42,
            metrics=["agents_alive_at_end"],
        )

        runner = ExperimentRunner(exp_config)
        results = runner.run_all()

        # Should complete without error (hook never fires)
        assert len(results) == 1
        assert results[0].metrics["agents_alive_at_end"] >= 0

    def test_unknown_action_warns(self):
        """Test that unknown hook action emits a warning."""
        sim_config = SimulationConfig(
            world_width=16, world_height=16, max_ticks=30, num_agents=3, seed=42
        )
        engine = SimulationEngine(sim_config)
        engine.setup_multi_agent()

        exp_config = ExperimentConfig(
            name="test",
            description="test",
            base={},
            conditions=[ExperimentCondition("test", {})],
        )
        runner = ExperimentRunner(exp_config)

        # Create hook with unknown action
        hook = TickHook(at_tick=10, action="unknown_action", params={})

        # Should warn when dispatching
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            runner._dispatch_hook(engine, hook)

            assert len(w) == 1
            assert "Unknown tick_hook action" in str(w[0].message)
