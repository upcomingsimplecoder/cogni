"""Experiment runner for executing simulations."""

from __future__ import annotations

import json
import random
import time
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import SimulationConfig
from src.experiments.config import ExperimentCondition, ExperimentConfig, TickHook
from src.experiments.provenance import ExperimentProvenance, capture_provenance
from src.experiments.registry import ExperimentRegistry
from src.simulation.engine import SimulationEngine
from src.trajectory.recorder import TrajectoryRecorder


@dataclass
class RunResult:
    """Result from a single simulation run."""

    condition_name: str
    replicate: int
    seed: int
    metrics: dict[str, float]
    duration_seconds: float
    trajectory_path: str | None = None


class ExperimentRunner:
    """Executes experiment conditions with replicates.

    Manages simulation creation, execution, and metric collection.
    Tracks provenance for reproducibility.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        config_yaml_path: str | None = None,
        registry_path: str | None = None,
    ):
        """Initialize experiment runner.

        Args:
            config: Experiment configuration
            config_yaml_path: Path to YAML config (for provenance tracking)
            registry_path: Path to experiment registry (defaults to standard location)
        """
        self.config = config
        self.config_yaml_path = config_yaml_path or "config_from_code.yaml"
        self._results: list[RunResult] = []
        self._provenance: ExperimentProvenance | None = None
        self._experiment_id = str(uuid.uuid4())
        self._registry = ExperimentRegistry(registry_path)

    def run_all(self, progress_callback: Any = None) -> list[RunResult]:
        """Execute all conditions × replicates.

        Captures provenance before running, saves results + provenance after completion,
        and registers experiment in the global registry.

        Args:
            progress_callback: Optional callback(condition, replicate, total)

        Returns:
            List of RunResult objects
        """
        # Capture provenance before running
        start_time = time.time()
        seed_range = list(
            range(
                self.config.seed_start,
                self.config.seed_start + self.config.replicates,
            )
        )

        conditions = self.config.expand_conditions()
        total_runs = len(conditions) * self.config.replicates

        run_count = 0
        for condition in conditions:
            for replicate in range(self.config.replicates):
                seed = self.config.seed_start + replicate

                if progress_callback:
                    progress_callback(condition.name, replicate, total_runs)

                result = self._run_single(condition, replicate, seed)
                self._results.append(result)
                run_count += 1

        # Capture final provenance with duration
        duration = time.time() - start_time
        self._provenance = capture_provenance(
            experiment_id=self._experiment_id,
            config_yaml_path=self.config_yaml_path,
            config_resolved=self._get_resolved_config(),
            seed_range=seed_range,
            duration_seconds=duration,
        )

        # Save provenance to experiment output directory
        self._save_provenance()

        # Register experiment
        self._registry.register(self._provenance, self.config.output_dir)

        return self._results

    def _run_single(self, condition: ExperimentCondition, replicate: int, seed: int) -> RunResult:
        """Run a single simulation.

        Args:
            condition: Experimental condition
            replicate: Replicate number
            seed: Random seed

        Returns:
            RunResult with collected metrics
        """
        # Build config from base + condition overrides + seed
        config_dict = {**self.config.base, **condition.overrides, "seed": seed}

        # Create SimulationConfig from dict
        sim_config = SimulationConfig(**config_dict)

        # Create engine
        engine = SimulationEngine(sim_config)

        # Setup trajectory recording if enabled
        trajectory_path = None
        recorder = None
        should_record = False
        if self.config.trajectory_sample_count > 0:
            should_record = replicate < self.config.trajectory_sample_count
        elif self.config.record_trajectories:
            should_record = True
        if should_record:
            run_id = f"{condition.name}_{replicate}"
            recorder = TrajectoryRecorder(
                output_dir=str(Path(self.config.output_dir) / "trajectories"),
                run_id=run_id,
            )
            recorder.start_run(engine)
            trajectory_path = str(recorder.output_dir)

        # Setup agents
        engine.setup_multi_agent()

        # Run simulation
        start_time = time.time()
        while not engine.is_over():
            tick_record = engine.step_all()
            if recorder:
                recorder.record_tick(engine, tick_record)
            # Execute tick hooks
            self._execute_hooks(engine, tick_record.tick, condition)

        duration = time.time() - start_time

        # Finalize recording
        if recorder:
            recorder.end_run(engine)

        # Collect metrics
        metrics = self._collect_metrics(engine)

        return RunResult(
            condition_name=condition.name,
            replicate=replicate,
            seed=seed,
            metrics=metrics,
            duration_seconds=duration,
            trajectory_path=trajectory_path,
        )

    def _collect_metrics(self, engine: SimulationEngine) -> dict[str, float]:
        """Extract requested metrics from completed simulation.

        Args:
            engine: Simulation engine after completion

        Returns:
            Dict of metric name → value
        """
        metrics = {}
        for metric_name in self.config.metrics:
            metrics[metric_name] = self._get_metric(engine, metric_name)
        return metrics

    def _get_metric(self, engine: SimulationEngine, name: str) -> float:
        """Extract a single metric from engine.

        Supported metrics:

        Survival Category:
        - agents_alive_at_end: Number of living agents at end
        - avg_survival_ticks: Mean ticks alive across all agents
        - survival_rate: Fraction of agents alive (0-1)
        - avg_final_needs: Mean of all needs (hunger+thirst+energy+health)/4 for living agents
        - avg_final_health: Mean health of living agents at end

        Social Category:
        - total_cooperation_events: Count of GIVE actions
        - total_aggression_events: Count of ATTACK actions
        - cooperation_ratio: cooperation / (cooperation + aggression)
        - avg_trust_network_density: Fraction of relationships above trust 0.3
        - coalition_count: Number of active coalitions
        - max_coalition_size: Size of largest coalition
        - avg_coalition_cohesion: Mean cohesion across coalitions

        Cognitive Category:
        - avg_tom_accuracy: Mean ToM prediction accuracy across all models
        - avg_calibration_score: Mean metacognitive calibration score
        - total_strategy_switches: Sum of strategy switches across all agents
        - deliberation_rate: Fraction of agents who deliberated last tick

        Cultural Category:
        - cultural_diversity: Shannon diversity index of cultural groups
        - convention_count: Number of established linguistic conventions
        - avg_vocabulary_size: Mean lexicon size across agents
        - communication_success_rate: Mean communication success rate
        - innovation_count: Total symbol innovations

        Temporal Category:
        - emergence_event_count: Total emergence events detected
        - emergence_diversity: Number of distinct emergence event types
        - trait_evolution_magnitude: Sum of trait variance across agents

        Args:
            engine: Simulation engine
            name: Metric name

        Returns:
            Metric value as float
        """
        # ======== SURVIVAL CATEGORY ========
        if name == "agents_alive_at_end":
            return float(engine.registry.count_living)

        elif name == "avg_survival_ticks":
            all_agents = engine.registry.all_agents()
            if not all_agents:
                return 0.0
            return sum(a.ticks_alive for a in all_agents) / len(all_agents)

        elif name == "survival_rate":
            all_agents = engine.registry.all_agents()
            if not all_agents:
                return 0.0
            return engine.registry.count_living / len(all_agents)

        elif name == "avg_final_needs":
            living = engine.registry.living_agents()
            if not living:
                return 0.0
            total = 0.0
            for a in living:
                needs_sum = (
                    a.needs.hunger + a.needs.thirst + a.needs.energy + a.needs.health
                ) / 4.0
                total += needs_sum
            return total / len(living)

        elif name == "avg_final_health":
            living = engine.registry.living_agents()
            if not living:
                return 0.0
            return sum(a.needs.health for a in living) / len(living)

        # ======== SOCIAL CATEGORY ========
        elif name == "total_cooperation_events":
            total = 0
            for metric in engine.metrics_collector.history:
                total += metric.cooperation_events
            return float(total)

        elif name == "total_aggression_events":
            total = 0
            for metric in engine.metrics_collector.history:
                total += metric.aggression_events
            return float(total)

        elif name == "cooperation_ratio":
            coop = 0
            aggr = 0
            for metric in engine.metrics_collector.history:
                coop += metric.cooperation_events
                aggr += metric.aggression_events
            if coop + aggr == 0:
                return 0.5  # Neutral if no events
            return coop / (coop + aggr)

        elif name == "avg_trust_network_density":
            living = engine.registry.living_agents()
            if len(living) < 2:
                return 0.0

            total_possible = len(living) * (len(living) - 1)
            trust_relationships = 0

            for agent in living:
                memory_tuple = engine.registry.get_memory(agent.agent_id)
                if memory_tuple:
                    _, social = memory_tuple
                    if social and hasattr(social, "_relationships"):
                        for rel in social._relationships.values():
                            if rel.trust > 0.3:
                                trust_relationships += 1

            if total_possible == 0:
                return 0.0
            return trust_relationships / total_possible

        elif name == "coalition_count":
            if not hasattr(engine, "coalition_manager") or engine.coalition_manager is None:
                return 0.0
            return float(len(engine.coalition_manager.all_coalitions()))

        elif name == "max_coalition_size":
            if not hasattr(engine, "coalition_manager") or engine.coalition_manager is None:
                return 0.0
            coalitions = engine.coalition_manager.all_coalitions()
            if not coalitions:
                return 0.0
            return float(max(len(c.members) for c in coalitions))

        elif name == "avg_coalition_cohesion":
            if not hasattr(engine, "coalition_manager") or engine.coalition_manager is None:
                return 0.0
            coalitions = engine.coalition_manager.all_coalitions()
            if not coalitions:
                return 0.0
            return sum(c.cohesion for c in coalitions) / len(coalitions)

        # ======== COGNITIVE CATEGORY ========
        elif name == "avg_tom_accuracy":
            living = engine.registry.living_agents()
            if not living:
                return 0.0

            total_accuracy = 0.0
            model_count = 0

            for agent in living:
                loop = engine.registry.get_awareness_loop(agent.agent_id)
                if not loop:
                    continue

                # Walk strategy wrapper chain to find TheoryOfMindStrategy
                strategy = loop.strategy
                mind_state = None
                for _ in range(5):  # max wrapper depth
                    if hasattr(strategy, "mind_state"):
                        mind_state = strategy.mind_state
                        break
                    if hasattr(strategy, "inner_strategy"):
                        strategy = strategy.inner_strategy
                    elif hasattr(strategy, "_inner"):
                        strategy = strategy._inner
                    else:
                        break

                if mind_state and hasattr(mind_state, "models"):
                    for model in mind_state.models.values():
                        if hasattr(model, "prediction_accuracy"):
                            total_accuracy += model.prediction_accuracy
                            model_count += 1

            if model_count == 0:
                return 0.0
            return total_accuracy / model_count

        elif name == "avg_calibration_score":
            if not hasattr(engine, "metacognition_engine") or engine.metacognition_engine is None:
                return 0.0

            living = engine.registry.living_agents()
            if not living:
                return 0.0

            total_calibration = 0.0
            count = 0

            for agent in living:
                mc_state = engine.metacognition_engine.get_agent_state(str(agent.agent_id))
                if mc_state and hasattr(mc_state, "calibration"):
                    total_calibration += mc_state.calibration.calibration_score
                    count += 1

            if count == 0:
                return 0.0
            return total_calibration / count

        elif name == "total_strategy_switches":
            if not hasattr(engine, "metacognition_engine") or engine.metacognition_engine is None:
                return 0.0

            living = engine.registry.living_agents()
            total_switches = 0

            for agent in living:
                mc_state = engine.metacognition_engine.get_agent_state(str(agent.agent_id))
                if mc_state and hasattr(mc_state, "switch_history"):
                    total_switches += len(mc_state.switch_history)

            return float(total_switches)

        elif name == "deliberation_rate":
            living = engine.registry.living_agents()
            if not living:
                return 0.0

            deliberated_count = 0

            for agent in living:
                loop = engine.registry.get_awareness_loop(agent.agent_id)
                if (
                    loop
                    and hasattr(loop, "_last_intention")
                    and loop._last_intention
                    and getattr(loop._last_intention, "deliberation_used", False)
                ):
                    deliberated_count += 1

            return deliberated_count / len(living)

        # ======== CULTURAL CATEGORY ========
        elif name == "cultural_diversity":
            if not hasattr(engine, "cultural_engine") or engine.cultural_engine is None:
                return 0.0
            if not hasattr(engine, "cultural_analyzer") or engine.cultural_analyzer is None:
                return 0.0

            import math

            groups = engine.cultural_analyzer.detect_cultural_groups(engine.cultural_engine)

            if not groups:
                return 0.0

            living = engine.registry.living_agents()
            if not living:
                return 0.0

            # Shannon diversity: -Σ(p_i * ln(p_i))
            total_agents = len(living)
            diversity = 0.0

            for group in groups:
                if total_agents > 0:
                    p = len(group) / total_agents
                    if p > 0:
                        diversity -= p * math.log(p)

            return diversity

        elif name == "convention_count":
            if not hasattr(engine, "language_engine") or engine.language_engine is None:
                return 0.0

            established = engine.language_engine.get_established_conventions()
            return float(len(established))

        elif name == "avg_vocabulary_size":
            if not hasattr(engine, "language_engine") or engine.language_engine is None:
                return 0.0

            living = engine.registry.living_agents()
            if not living:
                return 0.0

            total_vocab = 0
            count = 0

            for agent in living:
                lexicon = engine.language_engine.get_lexicon(str(agent.agent_id))
                if lexicon:
                    total_vocab += lexicon.vocabulary_size()
                    count += 1

            if count == 0:
                return 0.0
            return total_vocab / count

        elif name == "communication_success_rate":
            if not hasattr(engine, "language_engine") or engine.language_engine is None:
                return 0.0

            # Get recent outcomes
            recent_outcomes = []
            if hasattr(engine.language_engine, "outcomes_this_tick"):
                recent_outcomes = engine.language_engine.outcomes_this_tick

            if not recent_outcomes and hasattr(engine.language_engine, "_communication_history"):
                # Aggregate across all agents
                for history in engine.language_engine._communication_history.values():
                    if history:
                        recent_outcomes.extend(history[-20:])

            if not recent_outcomes:
                return 0.0

            success_count = sum(1 for o in recent_outcomes if o.success)
            return success_count / len(recent_outcomes)

        elif name == "innovation_count":
            if not hasattr(engine, "language_engine") or engine.language_engine is None:
                return 0.0

            # Sum innovations across all snapshots
            total = 0
            if hasattr(engine.language_engine, "snapshots"):
                for snapshot in engine.language_engine.snapshots:
                    total += snapshot.innovations_this_tick

            return float(total)

        # ======== TEMPORAL CATEGORY ========
        elif name == "emergence_event_count":
            return float(len(engine.emergence_detector.all_events))

        elif name == "emergence_diversity":
            events = engine.emergence_detector.all_events
            if not events:
                return 0.0

            pattern_types = set()
            for event in events:
                if hasattr(event, "pattern_type"):
                    pattern_types.add(event.pattern_type)

            return float(len(pattern_types))

        elif name == "trait_evolution_magnitude":
            # Calculate sum of trait variance across agents
            all_agents = engine.registry.all_agents()
            if not all_agents:
                return 0.0

            # Collect all trait values
            trait_values: dict[str, list[float]] = {}
            for agent in all_agents:
                if (
                    hasattr(agent, "profile")
                    and agent.profile is not None
                    and hasattr(agent.profile, "traits")
                ):
                    traits_dict = agent.profile.traits.as_dict()
                    for trait_name, value in traits_dict.items():
                        if trait_name not in trait_values:
                            trait_values[trait_name] = []
                        trait_values[trait_name].append(value)

            # Calculate variance for each trait and sum
            import math

            total_variance = 0.0
            for values in trait_values.values():
                if len(values) > 1:
                    mean = sum(values) / len(values)
                    variance = sum((x - mean) ** 2 for x in values) / len(values)
                    total_variance += math.sqrt(variance)  # Use std dev as magnitude

            return total_variance

        else:
            # Unknown metric
            return 0.0

    def _execute_hooks(
        self, engine: SimulationEngine, current_tick: int, condition: ExperimentCondition
    ) -> None:
        """Execute all hooks scheduled for the current tick.

        Args:
            engine: Simulation engine
            current_tick: Current simulation tick
            condition: Current experimental condition
        """
        # Merge config-wide hooks and condition-specific hooks
        all_hooks = self.config.tick_hooks + condition.tick_hooks

        # Execute hooks scheduled for this tick
        for hook in all_hooks:
            if hook.at_tick == current_tick:
                self._dispatch_hook(engine, hook)

    def _dispatch_hook(self, engine: SimulationEngine, hook: TickHook) -> None:
        """Route hook to appropriate handler.

        Args:
            engine: Simulation engine
            hook: Hook to execute
        """
        if hook.action == "corrupt_traits":
            self._hook_corrupt_traits(engine, hook.params)
        elif hook.action == "remove_agent":
            self._hook_remove_agent(engine, hook.params)
        else:
            warnings.warn(f"Unknown tick_hook action: {hook.action}", stacklevel=2)

    def _hook_corrupt_traits(self, engine: SimulationEngine, params: dict[str, Any]) -> None:
        """Corrupt agent traits for alignment experiments.

        Args:
            engine: Simulation engine
            params: Hook parameters with keys:
                - target: "all" | "random_n" | "agent_index"
                - traits: dict of trait_name → value
                - mode: "set" (default) or "shift"
                - n: (for random_n) number of agents to target
                - index: (for agent_index) index in living agents list
        """
        target = params.get("target", "all")
        traits = params.get("traits", {})
        mode = params.get("mode", "set")

        living = list(engine.registry.living_agents())
        if not living:
            return

        # Select target agents
        target_agents = []
        if target == "all":
            target_agents = living
        elif target == "random_n":
            n = params.get("n", 1)
            # Use reproducible random selection
            rng = random.Random(engine.config.seed + engine.state.tick)
            target_agents = rng.sample(living, min(n, len(living)))
        elif target == "agent_index":
            index = params.get("index", 0)
            if 0 <= index < len(living):
                target_agents = [living[index]]

        # Apply trait corruption
        for agent in target_agents:
            if agent.profile is None:
                continue
            for trait_name, value in traits.items():
                if hasattr(agent.profile.traits, trait_name):
                    if mode == "set":
                        # Clamp value to [0, 1]
                        clamped_value = max(0.0, min(1.0, value))
                        setattr(agent.profile.traits, trait_name, clamped_value)
                    elif mode == "shift":
                        agent.profile.traits.shift_trait(trait_name, value)

    def _hook_remove_agent(self, engine: SimulationEngine, params: dict[str, Any]) -> None:
        """Remove agent(s) from simulation.

        Args:
            engine: Simulation engine
            params: Hook parameters with keys:
                - target: "agent_index" | "most_cooperative" | "least_cooperative"
                - index: (for agent_index) index in living agents list
        """
        target = params.get("target", "agent_index")
        living = list(engine.registry.living_agents())

        if not living:
            return

        # Select target agent
        target_agent = None
        if target == "agent_index":
            index = params.get("index", 0)
            if 0 <= index < len(living):
                target_agent = living[index]
        elif target == "most_cooperative":
            # Sort by cooperation_tendency descending
            valid_agents = [a for a in living if a.profile is not None]
            if valid_agents:
                # Since we've filtered for non-None profiles, we can assert
                target_agent = max(
                    valid_agents,
                    key=lambda a: (
                        a.profile.traits.cooperation_tendency if a.profile is not None else 0.0
                    ),
                )
        elif target == "least_cooperative":
            # Sort by cooperation_tendency ascending
            valid_agents = [a for a in living if a.profile is not None]
            if valid_agents:
                # Since we've filtered for non-None profiles, we can assert
                target_agent = min(
                    valid_agents,
                    key=lambda a: (
                        a.profile.traits.cooperation_tendency if a.profile is not None else 1.0
                    ),
                )

        # Remove agent
        if target_agent:
            engine.registry.kill(target_agent.agent_id, cause="experiment_hook")

    def _get_resolved_config(self) -> dict:
        """Get full resolved configuration as dictionary.

        Returns:
            Dictionary with all experiment config settings
        """
        return {
            "name": self.config.name,
            "description": self.config.description,
            "base": self.config.base,
            "conditions": [
                {
                    "name": c.name,
                    "overrides": c.overrides,
                    "tick_hooks": [
                        {
                            "at_tick": h.at_tick,
                            "action": h.action,
                            "params": h.params,
                        }
                        for h in c.tick_hooks
                    ],
                }
                for c in self.config.conditions
            ],
            "replicates": self.config.replicates,
            "seed_start": self.config.seed_start,
            "metrics": self.config.metrics,
            "output_dir": self.config.output_dir,
            "formats": self.config.formats,
            "record_trajectories": self.config.record_trajectories,
            "trajectory_sample_count": self.config.trajectory_sample_count,
            "tick_hooks": [
                {
                    "at_tick": h.at_tick,
                    "action": h.action,
                    "params": h.params,
                }
                for h in self.config.tick_hooks
            ],
        }

    def _save_provenance(self) -> None:
        """Save provenance to experiment output directory."""
        if self._provenance is None:
            return

        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save provenance.json
        provenance_path = output_dir / f"provenance_{self._experiment_id}.json"
        with open(provenance_path, "w", encoding="utf-8") as f:
            json.dump(self._provenance.to_dict(), f, indent=2)

        # Save lock file (simplified version for quick reference)
        lock_path = output_dir / f"experiment_{self._experiment_id}.lock"
        with open(lock_path, "w", encoding="utf-8") as f:
            f.write(f"Experiment ID: {self._experiment_id}\n")
            f.write(f"Timestamp: {self._provenance.timestamp}\n")
            f.write(f"Git Commit: {self._provenance.git_commit or 'N/A'}\n")
            f.write(f"Git Dirty: {self._provenance.git_dirty}\n")
            f.write(f"Python: {self._provenance.python_version}\n")
            f.write(f"AUTOCOG: {self._provenance.autocog_version}\n")
            f.write(f"Config Hash: {self._provenance.config_yaml_hash}\n")
            f.write(f"Seeds: {self._provenance.seed_range}\n")
            f.write(f"Duration: {self._provenance.duration_seconds:.2f}s\n")

    def get_provenance(self) -> ExperimentProvenance | None:
        """Get experiment provenance metadata.

        Returns:
            ExperimentProvenance object, or None if not yet run
        """
        return self._provenance

    def get_experiment_id(self) -> str:
        """Get unique experiment ID.

        Returns:
            UUID string
        """
        return self._experiment_id
