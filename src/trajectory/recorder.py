"""Trajectory recorder: captures agent state during simulation.

Hooks into SimulationEngine to record per-tick data and streams to JSONL for crash safety.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

from src.trajectory.schema import (
    AgentSnapshot,
    EmergenceSnapshot,
    RunMetadata,
    TrajectoryDataset,
)

if TYPE_CHECKING:
    from src.simulation.engine import AgentTickRecord, SimulationEngine, TickRecord


class TrajectoryRecorder:
    """Records trajectory data during simulation.

    Hooks into SimulationEngine.step_all() to capture per-tick data.
    Writes incrementally to JSONL for crash safety.
    """

    def __init__(self, output_dir: str = "data/trajectories", run_id: str | None = None):
        self.run_id = run_id or str(uuid.uuid4())[:12]
        self.output_dir = Path(output_dir) / self.run_id
        self._snapshots: list[AgentSnapshot] = []
        self._events: list[EmergenceSnapshot] = []
        self._jsonl_file: TextIO | None = None
        self._metadata: RunMetadata | None = None
        self._prev_traits: dict[str, dict[str, float]] = {}  # agent_id -> previous trait values

    def start_run(self, engine: SimulationEngine) -> None:
        """Initialize recording. Called before first tick."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._metadata = self._build_metadata(engine)
        self._jsonl_file = (self.output_dir / "trajectory.jsonl").open("w")
        # Write metadata as first line
        self._write_jsonl({"type": "metadata", **self._metadata.to_dict()})

    def record_tick(self, engine: SimulationEngine, tick_record: TickRecord) -> None:
        """Record one tick's data. Called after each step_all()."""
        for agent_record in tick_record.agent_records:
            snapshot = self._build_agent_snapshot(engine, agent_record, tick_record.tick)
            self._snapshots.append(snapshot)
            self._write_jsonl({"type": "agent_snapshot", **snapshot.to_dict()})

        # Record emergence events
        if hasattr(tick_record, "emergent_events") and tick_record.emergent_events:
            for event in tick_record.emergent_events:
                es = self._build_emergence_snapshot(event, tick_record.tick)
                self._events.append(es)
                self._write_jsonl({"type": "emergence", **es.to_dict()})

    def end_run(self, engine: SimulationEngine) -> None:
        """Finalize recording. Called after simulation ends."""
        if self._metadata:
            self._metadata.actual_ticks = engine.state.tick
            self._metadata.final_state = {
                "agents_alive": engine.registry.count_living,
                "agents_dead": engine.registry.count_dead,
                "emergence_events_count": len(self._events),
            }
            # Write final metadata
            self._write_jsonl({"type": "run_complete", **self._metadata.to_dict()})
        if self._jsonl_file:
            self._jsonl_file.close()

        # Dual-write: export to Parquet if pyarrow available
        self._try_parquet_export()
        # Register in run catalog if duckdb available
        self._try_catalog_register()

    def _try_parquet_export(self) -> None:
        """Export to Parquet format if pyarrow is installed."""
        try:
            from src.trajectory.parquet import ParquetExporter

            dataset = self.get_dataset()
            ParquetExporter.export(dataset, str(self.output_dir))
        except ImportError:
            pass  # pyarrow not installed, skip silently
        except Exception:
            pass  # Don't let Parquet export failure break the run

    def _try_catalog_register(self) -> None:
        """Register this run in the DuckDB catalog if available."""
        try:
            from src.trajectory.catalog import RunCatalog

            catalog = RunCatalog()
            catalog.register_run(str(self.output_dir))
            catalog.close()
        except ImportError:
            pass  # duckdb not installed, skip silently
        except Exception:
            pass  # Don't let catalog failure break the run

    def _build_metadata(self, engine: SimulationEngine) -> RunMetadata:
        """Build initial run metadata from engine state."""
        agents_list = []
        for agent in engine.registry.living_agents():
            if agent.profile:
                agents_list.append(
                    {
                        "id": str(agent.agent_id),
                        "name": agent.profile.name,
                        "archetype": agent.profile.archetype,
                        "initial_traits": agent.profile.traits.as_dict(),
                    }
                )

        return RunMetadata(
            run_id=self.run_id,
            timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            seed=engine.config.seed,
            config=self._config_to_dict(engine.config),
            num_agents=len(agents_list),
            max_ticks=engine.config.max_ticks,
            actual_ticks=0,
            agents=agents_list,
            architecture=getattr(engine.config, "default_architecture", None),
        )

    def _build_agent_snapshot(
        self, engine: SimulationEngine, agent_record: AgentTickRecord, tick: int
    ) -> AgentSnapshot:
        """Extract full agent state from AgentTickRecord + engine context."""
        agent = engine.registry.get(agent_record.agent_id)
        loop = engine.registry.get_awareness_loop(agent_record.agent_id)

        # Get SRIE state from awareness loop's cached values
        sensation_summary: dict[str, Any] = {}
        reflection_dict: dict[str, Any] = {}
        intention_dict: dict[str, Any] = {}

        if loop:
            if loop._last_sensation:
                s = loop._last_sensation
                sensation_summary = {
                    "visible_agent_count": len(s.visible_agents),
                    "visible_resource_tiles": sum(1 for t in s.visible_tiles if t.resources),
                    "total_resources": sum(qty for t in s.visible_tiles for _, qty in t.resources),
                    "message_count": len(s.incoming_messages),
                    "time_of_day": s.time_of_day,
                }
            if loop._last_reflection:
                r = loop._last_reflection
                reflection_dict = {
                    "threat_level": r.threat_level,
                    "opportunity_score": r.opportunity_score,
                    "need_trends": dict(r.need_trends),
                    "last_action_succeeded": r.last_action_succeeded,
                    "interaction_count": len(r.recent_interaction_outcomes),
                }
            if loop._last_intention:
                i = loop._last_intention
                intention_dict = {
                    "primary_goal": i.primary_goal,
                    "confidence": i.confidence,
                    "target_position": i.target_position,
                    "target_agent_id": str(i.target_agent_id) if i.target_agent_id else None,
                }

        # Compute trait changes by diffing current traits against previous tick
        trait_changes: list[dict] = []
        current_traits = agent.profile.traits.as_dict() if agent and agent.profile else {}
        agent_id_str = str(agent_record.agent_id)
        if current_traits and agent_id_str in self._prev_traits:
            prev = self._prev_traits[agent_id_str]
            for trait_name, current_val in current_traits.items():
                old_val = prev.get(trait_name, current_val)
                if abs(current_val - old_val) > 1e-9:
                    trait_changes.append(
                        {
                            "trait": trait_name,
                            "old_value": round(old_val, 6),
                            "new_value": round(current_val, 6),
                            "delta": round(current_val - old_val, 6),
                        }
                    )
        # Store current traits for next tick's diff
        if current_traits:
            self._prev_traits[agent_id_str] = dict(current_traits)

        # Cultural transmission data (Phase 2)
        cultural_repertoire: dict = {}
        cultural_learning_style = ""
        transmission_events_this_tick: list[dict] = []
        cultural_group_id = -1

        if hasattr(engine, "cultural_engine") and engine.cultural_engine:
            aid = str(agent_record.agent_id)
            rep = engine.cultural_engine.get_repertoire(aid)
            if rep:
                cultural_repertoire = rep.to_dict()

            weights = engine.cultural_engine.get_transmission_weights(aid)
            if weights:
                cultural_learning_style = weights.dominant_style.value

            # Transmission events for this agent this tick
            for evt in engine.cultural_engine.transmission_events:
                if evt.tick == tick and evt.observer_id == aid:
                    transmission_events_this_tick.append(
                        {
                            "variant_id": evt.variant_id,
                            "actor_id": evt.actor_id,
                            "bias_type": evt.bias_type,
                            "adopted": evt.adopted,
                            "probability": round(evt.adoption_probability, 3),
                        }
                    )

            # Cultural group detection (only on snapshot ticks)
            if (
                hasattr(engine, "cultural_analyzer")
                and engine.cultural_analyzer
                and tick % engine.config.cultural_snapshot_interval == 0
            ):
                groups = engine.cultural_analyzer.detect_cultural_groups(engine.cultural_engine)
                for group_idx, group in enumerate(groups):
                    if aid in group:
                        cultural_group_id = group_idx
                        break

        # Theory of Mind data
        tom_models: dict = {}
        tom_model_count = 0
        if loop:
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
            if mind_state:
                tom_model_count = len(mind_state.models)
                for other_id, model in mind_state.models.items():
                    tom_models[other_id] = {
                        "estimated_disposition": model.estimated_disposition,
                        "prediction_accuracy": model.prediction_accuracy,
                        "last_observed_tick": model.last_observed_tick,
                        "ticks_observed": model.ticks_observed,
                        "times_helped_me": model.times_helped_me,
                        "times_attacked_me": model.times_attacked_me,
                    }

        # Coalition data
        coalition_id_val: str | None = None
        coalition_role = ""
        coalition_goal = ""
        if hasattr(engine, "coalition_manager") and engine.coalition_manager:
            aid = str(agent_record.agent_id)
            coalition = engine.coalition_manager.get_coalition(aid)
            if coalition:
                coalition_id_val = coalition.id
                coalition_role = "leader" if coalition.leader_id == aid else "member"
                coalition_goal = coalition.shared_goal or ""

        # Social relationships (Addition 1)
        social_relationships_data = {}
        memory_tuple = engine.registry.get_memory(agent_record.agent_id)
        if memory_tuple:
            episodic_memory, social_memory = memory_tuple
            if social_memory and hasattr(social_memory, "_relationships"):
                social_relationships_data = {
                    str(other_id): {
                        "trust": round(rel.trust, 3),
                        "interaction_count": rel.interaction_count,
                        "net_resources_given": rel.net_resources_given,
                        "was_attacked_by": rel.was_attacked_by,
                        "was_helped_by": rel.was_helped_by,
                        "last_interaction_tick": rel.last_interaction_tick,
                    }
                    for other_id, rel in social_memory._relationships.items()
                }

        # Metacognition calibration curve and deliberation flag (Additions 4 & 5)
        metacog_calibration_curve = []
        metacog_deliberation_invoked = False
        if hasattr(engine, "metacognition_engine") and engine.metacognition_engine:
            aid = str(agent_record.agent_id)
            mc_state = engine.metacognition_engine.get_agent_state(aid)
            if mc_state and hasattr(mc_state.calibration, "calibration_curve"):
                metacog_calibration_curve = [
                    {"bin_center": bin_center, "accuracy": accuracy, "count": count}
                    for bin_center, accuracy, count in mc_state.calibration.calibration_curve()
                ]
            # Check deliberation flag from intention
            if loop and loop._last_intention:
                metacog_deliberation_invoked = getattr(
                    loop._last_intention, "deliberation_used", False
                )

        # Plan state (Addition 6)
        plan_state_data = {}
        if hasattr(engine.registry, "get_planner"):
            planner = engine.registry.get_planner(agent_record.agent_id)
            if planner and planner._active_goal_id:
                active_plan = planner._plans.get(planner._active_goal_id)
                active_goal = planner._goals.get(planner._active_goal_id)
                if active_plan and active_goal:
                    plan_state_data = {
                        "goal": active_goal.description,
                        "steps": len(active_plan.steps),
                        "current_step": active_plan.current_step_index,
                        "status": active_plan.status,
                        "progress": round(active_plan.progress, 2),
                    }

        # Language symbols (Addition 7)
        language_symbols_data = []
        if hasattr(engine, "language_engine") and engine.language_engine:
            aid = str(agent_record.agent_id)
            lex = engine.language_engine.get_lexicon(aid)
            if lex and hasattr(lex, "_by_symbol"):
                for sym_form, assocs in lex._by_symbol.items():
                    if assocs:
                        best_assoc = max(assocs, key=lambda a: a.strength)
                        if best_assoc.strength >= 0.2:
                            language_symbols_data.append(
                                {
                                    "form": sym_form,
                                    "meaning": (
                                        f"{best_assoc.meaning.meaning_type.value}:"
                                        f"{best_assoc.meaning.referent}"
                                    ),
                                    "success_rate": round(best_assoc.symbol.success_rate, 3),
                                    "times_used": best_assoc.symbol.times_used,
                                    "strength": round(best_assoc.strength, 3),
                                }
                            )

        return AgentSnapshot(
            tick=tick,
            agent_id=str(agent_record.agent_id),
            agent_name=agent.profile.name if agent and agent.profile else "unknown",
            archetype=agent.profile.archetype if agent and agent.profile else "unknown",
            position=agent_record.position,
            alive=agent.alive if agent else False,
            hunger=agent_record.needs_after.get("hunger", 0),
            thirst=agent_record.needs_after.get("thirst", 0),
            energy=agent_record.needs_after.get("energy", 0),
            health=agent_record.needs_after.get("health", 0),
            traits=agent.profile.traits.as_dict() if agent and agent.profile else {},
            sensation_summary=sensation_summary,
            reflection=reflection_dict,
            intention=intention_dict,
            action_type=agent_record.action.type.value if agent_record.action else "none",
            action_target=agent_record.action.target if agent_record.action else None,
            action_target_agent=str(agent_record.action.target_agent_id)
            if agent_record.action
            and hasattr(agent_record.action, "target_agent_id")
            and agent_record.action.target_agent_id
            else None,
            action_succeeded=agent_record.result.success if agent_record.result else False,
            needs_delta=agent_record.result.needs_delta or {} if agent_record.result else {},
            inventory=dict(agent.inventory) if agent and hasattr(agent, "inventory") else {},
            messages_sent=[self._msg_to_dict(m) for m in agent_record.messages_sent],
            messages_received=[self._msg_to_dict(m) for m in agent_record.messages_received],
            internal_monologue=agent_record.internal_monologue or "",
            trait_changes=trait_changes,
            cultural_repertoire=cultural_repertoire,
            cultural_learning_style=cultural_learning_style,
            transmission_events_this_tick=transmission_events_this_tick,
            cultural_group_id=cultural_group_id,
            tom_model_count=tom_model_count,
            tom_models=tom_models,
            coalition_id=coalition_id_val,
            coalition_role=coalition_role,
            coalition_goal=coalition_goal,
            social_relationships=social_relationships_data,
            metacog_calibration_curve=metacog_calibration_curve,
            metacog_deliberation_invoked=metacog_deliberation_invoked,
            plan_state=plan_state_data,
            language_symbols=language_symbols_data,
        )

    def _build_emergence_snapshot(self, event: Any, tick: int) -> EmergenceSnapshot:
        """Convert emergence event to snapshot."""
        # event is likely a string or object with pattern_type, agents, description
        if isinstance(event, str):
            # Parse string format (simple case)
            return EmergenceSnapshot(
                tick=tick,
                pattern_type="unknown",
                agents_involved=[],
                description=event,
                data={},
            )
        else:
            # Assume it has attributes
            return EmergenceSnapshot(
                tick=tick,
                pattern_type=getattr(event, "pattern_type", "unknown"),
                agents_involved=[str(aid) for aid in getattr(event, "agents_involved", [])],
                description=getattr(event, "description", str(event)),
                data=getattr(event, "data", {}),
            )

    def _msg_to_dict(self, msg: Any) -> dict:
        """Convert message object to dictionary."""
        if msg is None:
            return {}

        # Handle message_type which could be enum or string
        message_type = getattr(msg, "message_type", "unknown")
        if hasattr(message_type, "value"):
            type_str: str = message_type.value
        else:
            type_str = str(message_type)

        return {
            "type": type_str,
            "sender": str(getattr(msg, "sender_id", "unknown")),
            "receiver": str(getattr(msg, "receiver_id", "unknown")),
            "content": getattr(msg, "content", ""),
        }

    def _config_to_dict(self, config: Any) -> dict:
        """Convert SimulationConfig to dictionary."""
        # Pydantic Settings has model_dump() method
        if hasattr(config, "model_dump"):
            result = config.model_dump()
            return dict(result)
        elif hasattr(config, "dict"):
            result = config.dict()
            return dict(result)
        else:
            # Fallback: extract public attributes
            return {k: v for k, v in vars(config).items() if not k.startswith("_")}

    def _write_jsonl(self, data: dict) -> None:
        """Write a single line to JSONL file."""
        if self._jsonl_file:
            self._jsonl_file.write(json.dumps(data) + "\n")
            self._jsonl_file.flush()  # Ensure immediate write for crash safety

    def get_dataset(self) -> TrajectoryDataset:
        """Get the current dataset (useful for in-memory analysis)."""
        if not self._metadata:
            raise RuntimeError("Recorder not started — call start_run() first")
        return TrajectoryDataset(
            metadata=self._metadata,
            agent_snapshots=self._snapshots,
            emergence_events=self._events,
        )
