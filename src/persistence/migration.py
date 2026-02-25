"""Schema migration: handle version changes in checkpoint format.

Currently only version 1 exists. This module provides the structure
for future migrations when the checkpoint format evolves.
"""

from __future__ import annotations


class SchemaMigration:
    """Handle schema version changes in checkpoint format."""

    @staticmethod
    def migrate(data: dict) -> dict:
        """Migrate data to current schema version.

        Args:
            data: Checkpoint data to migrate

        Returns:
            Migrated data at current schema version
        """
        version = SchemaMigration.get_version(data)

        # Migration chain
        if version == 1:
            data = SchemaMigration._migrate_1_to_2(data)
        # Future migrations:
        # if version == 2:
        #     data = SchemaMigration._migrate_2_to_3(data)
        # etc.

        return data

    @staticmethod
    def get_version(data: dict) -> int:
        """Extract schema version from checkpoint data.

        Args:
            data: Checkpoint data

        Returns:
            Schema version number (default 1)
        """
        version = data.get("schema_version", 1)
        return int(version)

    @staticmethod
    def _migrate_1_to_2(data: dict) -> dict:
        """Migrate from schema v1 to v2.

        Adds new top-level sections for:
        - mailboxes: Agent message passing system
        - awareness_loops: Metacognitive tracking
        - trait_evolution: Personality trait history
        - lineage: Agent lineage tracking
        - effectiveness: Performance metrics and scoring

        Args:
            data: Schema v1 checkpoint data

        Returns:
            Schema v2 checkpoint data
        """
        data["schema_version"] = 2

        # Add new top-level sections with empty defaults
        data.setdefault("mailboxes", {})
        data.setdefault("awareness_loops", {})
        data.setdefault("trait_evolution", {"history": []})
        data.setdefault("lineage", {"roots": [], "nodes": {}})
        data.setdefault(
            "effectiveness",
            {
                "nudge_scores": [],
                "router_records": [],
                "classifier_records": [],
                "quality_scores": {},
                "classifier_accuracy": {},
            },
        )

        # Preserve existing emergence data
        data.setdefault("emergence", data.get("emergence", {"event_count": 0}))

        return data
