"""Parquet export/import for trajectory data.

Converts between TrajectoryDataset and columnar Parquet format.
Requires: pyarrow (optional dependency, install with pip install cogniarch[data])
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.trajectory.schema import AgentSnapshot, EmergenceSnapshot, TrajectoryDataset


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for non-standard types like tuples."""
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class ParquetExporter:
    """Export/import trajectory data to/from Parquet format."""

    SCHEMA_VERSION = "1.0.0"

    @staticmethod
    def export(dataset: TrajectoryDataset, output_dir: str) -> dict[str, Path]:
        """Export TrajectoryDataset to Parquet files.

        Creates three files:
        - agent_snapshots.parquet: columnar agent state data
        - emergence_events.parquet: emergence events
        - metadata.json: run metadata + schema version

        Args:
            dataset: TrajectoryDataset to export
            output_dir: Directory to write files to (created if needed)

        Returns:
            Dict mapping file type to Path: {"agent_snapshots": Path, ...}

        Raises:
            ImportError: If pyarrow is not installed
        """
        # Import guard
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError(
                "pyarrow is required for Parquet export.\nInstall with: pip install cogniarch[data]"
            ) from None

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        result = {}

        # Export agent snapshots
        rows = [ParquetExporter._snapshot_to_row(snapshot) for snapshot in dataset.agent_snapshots]
        if rows:
            table = pa.Table.from_pylist(rows, schema=ParquetExporter._build_agent_schema())
        else:
            # Create empty table with schema
            schema = ParquetExporter._build_agent_schema()
            arrays = [pa.array([], type=field.type) for field in schema]
            table = pa.Table.from_arrays(arrays, schema=schema)

        agent_path = output_path / "agent_snapshots.parquet"
        pq.write_table(table, agent_path)
        result["agent_snapshots"] = agent_path

        # Export emergence events
        event_rows = [
            ParquetExporter._emergence_to_row(event) for event in dataset.emergence_events
        ]
        if event_rows:
            event_table = pa.Table.from_pylist(
                event_rows, schema=ParquetExporter._build_emergence_schema()
            )
        else:
            # Create empty table with schema
            schema = ParquetExporter._build_emergence_schema()
            arrays = [pa.array([], type=field.type) for field in schema]
            event_table = pa.Table.from_arrays(arrays, schema=schema)

        event_path = output_path / "emergence_events.parquet"
        pq.write_table(event_table, event_path)
        result["emergence_events"] = event_path

        # Export metadata
        metadata_dict = dataset.metadata.to_dict()
        metadata_dict["schema_version"] = ParquetExporter.SCHEMA_VERSION
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata_dict, f, indent=2)
        result["metadata"] = metadata_path

        return result

    @staticmethod
    def load(input_dir: str) -> TrajectoryDataset:
        """Load TrajectoryDataset from Parquet files.

        Expects directory containing:
        - agent_snapshots.parquet
        - emergence_events.parquet
        - metadata.json

        Args:
            input_dir: Directory containing Parquet files

        Returns:
            Loaded TrajectoryDataset

        Raises:
            FileNotFoundError: If required files are missing
            ImportError: If pyarrow is not installed
        """
        # Import guard
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError(
                "pyarrow is required for Parquet import.\nInstall with: pip install cogniarch[data]"
            ) from None

        from src.trajectory.schema import (
            RunMetadata,
            TrajectoryDataset,
        )

        input_path = Path(input_dir)

        # Check files exist
        agent_file = input_path / "agent_snapshots.parquet"
        event_file = input_path / "emergence_events.parquet"
        metadata_file = input_path / "metadata.json"

        if not agent_file.exists():
            raise FileNotFoundError(f"Missing agent_snapshots.parquet in {input_dir}")
        if not event_file.exists():
            raise FileNotFoundError(f"Missing emergence_events.parquet in {input_dir}")
        if not metadata_file.exists():
            raise FileNotFoundError(f"Missing metadata.json in {input_dir}")

        # Load metadata
        with open(metadata_file) as f:
            metadata_dict = json.load(f)
        # Remove schema_version before passing to RunMetadata
        metadata_dict.pop("schema_version", None)
        metadata = RunMetadata(**metadata_dict)

        # Load agent snapshots
        agent_table = pq.read_table(agent_file)
        agent_snapshots = [ParquetExporter._row_to_snapshot(row) for row in agent_table.to_pylist()]

        # Load emergence events
        event_table = pq.read_table(event_file)
        emergence_events = [
            ParquetExporter._row_to_emergence(row) for row in event_table.to_pylist()
        ]

        return TrajectoryDataset(
            metadata=metadata,
            agent_snapshots=agent_snapshots,
            emergence_events=emergence_events,
        )

    @staticmethod
    def _snapshot_to_row(snapshot: AgentSnapshot) -> dict:
        """Flatten AgentSnapshot to dict for Parquet.

        Scalar columns use native Parquet types.
        Complex nested structures stored as JSON strings.

        Args:
            snapshot: AgentSnapshot to flatten

        Returns:
            Dictionary with scalar + JSON-string columns
        """
        # Extract traits with defaults
        traits = snapshot.traits
        cooperation = traits.get("cooperation_tendency", 0.5)
        curiosity = traits.get("curiosity", 0.5)
        risk_tolerance = traits.get("risk_tolerance", 0.5)
        resource_sharing = traits.get("resource_sharing", 0.5)
        aggression = traits.get("aggression", 0.5)
        sociability = traits.get("sociability", 0.5)

        # Extract reflection fields
        reflection = snapshot.reflection
        threat_level = reflection.get("threat_level", 0.0)
        opportunity_score = reflection.get("opportunity_score", 0.0)

        # Extract intention fields
        intention = snapshot.intention
        primary_goal = intention.get("primary_goal", "")
        confidence = intention.get("confidence", 0.0)

        return {
            # Scalar columns
            "tick": snapshot.tick,
            "agent_id": snapshot.agent_id,
            "agent_name": snapshot.agent_name,
            "archetype": snapshot.archetype,
            "pos_x": snapshot.position[0],
            "pos_y": snapshot.position[1],
            "alive": snapshot.alive,
            "hunger": snapshot.hunger,
            "thirst": snapshot.thirst,
            "energy": snapshot.energy,
            "health": snapshot.health,
            "action_type": snapshot.action_type,
            "action_target": snapshot.action_target,
            "action_target_agent": snapshot.action_target_agent,
            "action_succeeded": snapshot.action_succeeded,
            "cooperation_tendency": cooperation,
            "curiosity": curiosity,
            "risk_tolerance": risk_tolerance,
            "resource_sharing": resource_sharing,
            "aggression": aggression,
            "sociability": sociability,
            "threat_level": threat_level,
            "opportunity_score": opportunity_score,
            "primary_goal": primary_goal,
            "confidence": confidence,
            "messages_sent_count": len(snapshot.messages_sent),
            "messages_received_count": len(snapshot.messages_received),
            "internal_monologue": snapshot.internal_monologue,
            "tom_model_count": snapshot.tom_model_count,
            "coalition_id": snapshot.coalition_id,
            "coalition_role": snapshot.coalition_role,
            "metacog_deliberation_invoked": snapshot.metacog_deliberation_invoked,
            "cultural_learning_style": snapshot.cultural_learning_style,
            "cultural_group_id": snapshot.cultural_group_id,
            # JSON string columns (complex nested data)
            "needs_delta": json.dumps(snapshot.needs_delta),
            "inventory": json.dumps(snapshot.inventory),
            "trait_changes": json.dumps(snapshot.trait_changes),
            "messages_sent": json.dumps(snapshot.messages_sent),
            "messages_received": json.dumps(snapshot.messages_received),
            "sensation_summary": json.dumps(snapshot.sensation_summary),
            "reflection": json.dumps(snapshot.reflection),
            "intention": json.dumps(snapshot.intention, default=_json_serializer),
            "tom_models": json.dumps(snapshot.tom_models),
            "social_relationships": json.dumps(snapshot.social_relationships),
            "cultural_repertoire": json.dumps(snapshot.cultural_repertoire),
            "transmission_events": json.dumps(snapshot.transmission_events_this_tick),
            "plan_state": json.dumps(snapshot.plan_state),
            "language_symbols": json.dumps(snapshot.language_symbols),
            "metacog_calibration_curve": json.dumps(snapshot.metacog_calibration_curve),
        }

    @staticmethod
    def _row_to_snapshot(row: dict) -> AgentSnapshot:
        """Reconstruct AgentSnapshot from Parquet row.

        Args:
            row: Dictionary from Parquet table

        Returns:
            AgentSnapshot instance
        """
        from src.trajectory.schema import AgentSnapshot

        # Reconstruct position tuple
        position = (row["pos_x"], row["pos_y"])

        # Reconstruct traits dict
        traits = {
            "cooperation_tendency": row["cooperation_tendency"],
            "curiosity": row["curiosity"],
            "risk_tolerance": row["risk_tolerance"],
            "resource_sharing": row["resource_sharing"],
            "aggression": row["aggression"],
            "sociability": row["sociability"],
        }

        # Parse JSON strings back to dicts/lists
        needs_delta = json.loads(row["needs_delta"])
        inventory = json.loads(row["inventory"])
        trait_changes = json.loads(row["trait_changes"])
        messages_sent = json.loads(row["messages_sent"])
        messages_received = json.loads(row["messages_received"])
        sensation_summary = json.loads(row["sensation_summary"])
        reflection = json.loads(row["reflection"])
        intention = json.loads(row["intention"])
        tom_models = json.loads(row["tom_models"])
        social_relationships = json.loads(row["social_relationships"])
        cultural_repertoire = json.loads(row["cultural_repertoire"])
        transmission_events = json.loads(row["transmission_events"])
        plan_state = json.loads(row["plan_state"])
        language_symbols = json.loads(row["language_symbols"])
        metacog_calibration_curve = json.loads(row["metacog_calibration_curve"])

        return AgentSnapshot(
            tick=row["tick"],
            agent_id=row["agent_id"],
            agent_name=row["agent_name"],
            archetype=row["archetype"],
            position=position,
            alive=row["alive"],
            hunger=row["hunger"],
            thirst=row["thirst"],
            energy=row["energy"],
            health=row["health"],
            traits=traits,
            sensation_summary=sensation_summary,
            reflection=reflection,
            intention=intention,
            action_type=row["action_type"],
            action_target=row["action_target"],
            action_target_agent=row["action_target_agent"],
            action_succeeded=row["action_succeeded"],
            needs_delta=needs_delta,
            inventory=inventory,
            messages_sent=messages_sent,
            messages_received=messages_received,
            internal_monologue=row["internal_monologue"],
            trait_changes=trait_changes,
            cultural_repertoire=cultural_repertoire,
            cultural_learning_style=row["cultural_learning_style"],
            transmission_events_this_tick=transmission_events,
            cultural_group_id=row["cultural_group_id"],
            tom_model_count=row["tom_model_count"],
            tom_models=tom_models,
            coalition_id=row["coalition_id"],
            coalition_role=row["coalition_role"],
            coalition_goal="",  # Not stored separately, embedded in plan_state if needed
            social_relationships=social_relationships,
            metacog_calibration_curve=metacog_calibration_curve,
            metacog_deliberation_invoked=row["metacog_deliberation_invoked"],
            plan_state=plan_state,
            language_symbols=language_symbols,
        )

    @staticmethod
    def _emergence_to_row(event: EmergenceSnapshot) -> dict:
        """Flatten EmergenceSnapshot to dict for Parquet.

        Args:
            event: EmergenceSnapshot to flatten

        Returns:
            Dictionary with scalar + JSON columns
        """
        return {
            "tick": event.tick,
            "pattern_type": event.pattern_type,
            "agents_involved": json.dumps(event.agents_involved),
            "description": event.description,
            "data": json.dumps(event.data),
        }

    @staticmethod
    def _row_to_emergence(row: dict) -> EmergenceSnapshot:
        """Reconstruct EmergenceSnapshot from Parquet row.

        Args:
            row: Dictionary from Parquet table

        Returns:
            EmergenceSnapshot instance
        """
        from src.trajectory.schema import EmergenceSnapshot

        agents_involved = json.loads(row["agents_involved"])
        data = json.loads(row["data"])

        return EmergenceSnapshot(
            tick=row["tick"],
            pattern_type=row["pattern_type"],
            agents_involved=agents_involved,
            description=row["description"],
            data=data,
        )

    @staticmethod
    def _build_agent_schema():
        """Build PyArrow schema for agent snapshots.

        Returns:
            PyArrow Schema with proper types for all columns
        """
        try:
            import pyarrow as pa
        except ImportError:
            raise ImportError(
                "pyarrow is required for Parquet export.\nInstall with: pip install cogniarch[data]"
            ) from None

        return pa.schema(
            [
                # Scalar columns
                ("tick", pa.int32()),
                ("agent_id", pa.string()),
                ("agent_name", pa.string()),
                ("archetype", pa.string()),
                ("pos_x", pa.int16()),
                ("pos_y", pa.int16()),
                ("alive", pa.bool_()),
                ("hunger", pa.float32()),
                ("thirst", pa.float32()),
                ("energy", pa.float32()),
                ("health", pa.float32()),
                ("action_type", pa.string()),
                ("action_target", pa.string()),
                ("action_target_agent", pa.string()),
                ("action_succeeded", pa.bool_()),
                ("cooperation_tendency", pa.float32()),
                ("curiosity", pa.float32()),
                ("risk_tolerance", pa.float32()),
                ("resource_sharing", pa.float32()),
                ("aggression", pa.float32()),
                ("sociability", pa.float32()),
                ("threat_level", pa.float32()),
                ("opportunity_score", pa.float32()),
                ("primary_goal", pa.string()),
                ("confidence", pa.float32()),
                ("messages_sent_count", pa.int16()),
                ("messages_received_count", pa.int16()),
                ("internal_monologue", pa.string()),
                ("tom_model_count", pa.int16()),
                ("coalition_id", pa.string()),
                ("coalition_role", pa.string()),
                ("metacog_deliberation_invoked", pa.bool_()),
                ("cultural_learning_style", pa.string()),
                ("cultural_group_id", pa.int16()),
                # JSON string columns
                ("needs_delta", pa.string()),
                ("inventory", pa.string()),
                ("trait_changes", pa.string()),
                ("messages_sent", pa.string()),
                ("messages_received", pa.string()),
                ("sensation_summary", pa.string()),
                ("reflection", pa.string()),
                ("intention", pa.string()),
                ("tom_models", pa.string()),
                ("social_relationships", pa.string()),
                ("cultural_repertoire", pa.string()),
                ("transmission_events", pa.string()),
                ("plan_state", pa.string()),
                ("language_symbols", pa.string()),
                ("metacog_calibration_curve", pa.string()),
            ]
        )

    @staticmethod
    def _build_emergence_schema():
        """Build PyArrow schema for emergence events.

        Returns:
            PyArrow Schema with proper types for emergence columns
        """
        try:
            import pyarrow as pa
        except ImportError:
            raise ImportError(
                "pyarrow is required for Parquet export.\nInstall with: pip install cogniarch[data]"
            ) from None

        return pa.schema(
            [
                ("tick", pa.int32()),
                ("pattern_type", pa.string()),
                ("agents_involved", pa.string()),  # JSON array
                ("description", pa.string()),
                ("data", pa.string()),  # JSON object
            ]
        )
