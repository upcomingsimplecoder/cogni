"""Experiment registry: track all experiments run on this machine."""

from __future__ import annotations

import json
from pathlib import Path

from src.experiments.provenance import ExperimentProvenance


class ExperimentRegistry:
    """Track all experiments run on this machine. JSONL-backed.

    Each line in the registry file is a JSON object with:
    - provenance: Full ExperimentProvenance dict
    - results_dir: Path to experiment output directory
    """

    REGISTRY_PATH = "data/experiments/_registry.jsonl"

    def __init__(self, registry_path: str | None = None):
        """Initialize registry.

        Args:
            registry_path: Path to registry file (defaults to REGISTRY_PATH)
        """
        self.registry_path = Path(registry_path or self.REGISTRY_PATH)

    def register(self, provenance: ExperimentProvenance, results_dir: str) -> None:
        """Append experiment to registry.

        Args:
            provenance: Experiment provenance metadata
            results_dir: Path to directory containing experiment results
        """
        # Ensure parent directory exists
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        # Append entry
        entry = {
            "provenance": provenance.to_dict(),
            "results_dir": results_dir,
        }

        with open(self.registry_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def list_experiments(self, filter_by: dict | None = None) -> list[dict]:
        """List past experiments with optional filtering.

        Args:
            filter_by: Optional dict of field → value to filter by
                      Supports nested fields like {"provenance.git_commit": "abc123"}

        Returns:
            List of registry entries (each with provenance + results_dir)
        """
        if not self.registry_path.exists():
            return []

        experiments = []
        with open(self.registry_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if self._matches_filter(entry, filter_by):
                        experiments.append(entry)
                except json.JSONDecodeError:
                    # Skip corrupted lines
                    continue

        return experiments

    def find_by_config(self, yaml_hash: str) -> list[dict]:
        """Find experiments that used the same config.

        Args:
            yaml_hash: SHA256 hash of YAML config file

        Returns:
            List of registry entries with matching config hash
        """
        return self.list_experiments(filter_by={"provenance.config_yaml_hash": yaml_hash})

    def compare(self, exp_id_a: str, exp_id_b: str) -> dict:
        """Compare two experiments: config diff + results diff.

        Args:
            exp_id_a: First experiment ID
            exp_id_b: Second experiment ID

        Returns:
            Dict with:
            - exp_a: First experiment entry (or None)
            - exp_b: Second experiment entry (or None)
            - config_diff: Dict of config differences
            - metadata_diff: Dict of metadata differences

        Raises:
            ValueError: If either experiment is not found
        """
        experiments = self.list_experiments()

        exp_a = None
        exp_b = None

        for exp in experiments:
            if exp["provenance"]["experiment_id"] == exp_id_a:
                exp_a = exp
            if exp["provenance"]["experiment_id"] == exp_id_b:
                exp_b = exp

        if exp_a is None:
            raise ValueError(f"Experiment {exp_id_a} not found in registry")
        if exp_b is None:
            raise ValueError(f"Experiment {exp_id_b} not found in registry")

        # Compare configs
        config_a = exp_a["provenance"]["config_resolved"]
        config_b = exp_b["provenance"]["config_resolved"]
        config_diff = self._dict_diff(config_a, config_b)

        # Compare metadata
        metadata_a = {
            k: v
            for k, v in exp_a["provenance"].items()
            if k not in ["config_resolved", "experiment_id"]
        }
        metadata_b = {
            k: v
            for k, v in exp_b["provenance"].items()
            if k not in ["config_resolved", "experiment_id"]
        }
        metadata_diff = self._dict_diff(metadata_a, metadata_b)

        return {
            "exp_a": exp_a,
            "exp_b": exp_b,
            "config_diff": config_diff,
            "metadata_diff": metadata_diff,
        }

    def _matches_filter(self, entry: dict, filter_by: dict | None) -> bool:
        """Check if entry matches filter criteria.

        Args:
            entry: Registry entry to check
            filter_by: Filter criteria (supports nested keys with dots)

        Returns:
            True if entry matches all filter criteria
        """
        if filter_by is None:
            return True

        for key, value in filter_by.items():
            # Support nested keys like "provenance.git_commit"
            parts = key.split(".")
            current = entry
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return False

            if current != value:
                return False

        return True

    def _dict_diff(self, dict_a: dict, dict_b: dict) -> dict:
        """Find differences between two dictionaries.

        Args:
            dict_a: First dictionary
            dict_b: Second dictionary

        Returns:
            Dict with:
            - only_in_a: Keys only in dict_a
            - only_in_b: Keys only in dict_b
            - different_values: Keys with different values
        """
        all_keys = set(dict_a.keys()) | set(dict_b.keys())

        only_in_a = {}
        only_in_b = {}
        different_values = {}

        for key in all_keys:
            if key not in dict_b:
                only_in_a[key] = dict_a[key]
            elif key not in dict_a:
                only_in_b[key] = dict_b[key]
            elif dict_a[key] != dict_b[key]:
                different_values[key] = {
                    "a": dict_a[key],
                    "b": dict_b[key],
                }

        return {
            "only_in_a": only_in_a,
            "only_in_b": only_in_b,
            "different_values": different_values,
        }
