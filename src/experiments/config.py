"""Experiment configuration with YAML support."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TickHook:
    """A hook that fires at a specific simulation tick."""

    at_tick: int
    action: str  # "corrupt_traits" | "remove_agent"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentCondition:
    """A single experimental condition with config overrides."""

    name: str
    overrides: dict[str, Any]
    tick_hooks: list[TickHook] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    """Configuration for a multi-condition experiment.

    Supports loading from YAML or dict, with automatic replicate expansion.
    """

    name: str
    description: str
    base: dict[str, Any]  # Base SimulationConfig overrides
    conditions: list[ExperimentCondition]
    replicates: int = 5
    seed_start: int = 42
    metrics: list[str] = field(
        default_factory=lambda: [
            "agents_alive_at_end",
            "avg_survival_ticks",
            "total_cooperation_events",
        ]
    )
    output_dir: str = "data/experiments"
    formats: list[str] = field(default_factory=lambda: ["csv", "markdown"])
    record_trajectories: bool = False
    trajectory_sample_count: int = (
        0  # Record trajectories for first N replicates only (0 = use record_trajectories bool)
    )
    tick_hooks: list[TickHook] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str) -> ExperimentConfig:
        """Load experiment config from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            ExperimentConfig instance

        Raises:
            ImportError: If pyyaml is not installed
            FileNotFoundError: If file doesn't exist
        """
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as err:
            raise ImportError(
                "pyyaml is required for YAML loading. Install with: pip install pyyaml"
            ) from err

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> ExperimentConfig:
        """Build ExperimentConfig from dictionary.

        Args:
            data: Dict with experiment configuration

        Returns:
            ExperimentConfig instance
        """
        # Parse conditions
        conditions = []
        for cond_data in data.get("conditions", []):
            # Parse tick_hooks for this condition
            cond_hooks = []
            for hook_data in cond_data.get("tick_hooks", []):
                cond_hooks.append(
                    TickHook(
                        at_tick=hook_data["at_tick"],
                        action=hook_data["action"],
                        params=hook_data.get("params", {}),
                    )
                )

            conditions.append(
                ExperimentCondition(
                    name=cond_data["name"],
                    overrides=cond_data.get("overrides", {}),
                    tick_hooks=cond_hooks,
                )
            )

        # Parse top-level tick_hooks
        top_level_hooks = []
        for hook_data in data.get("tick_hooks", []):
            top_level_hooks.append(
                TickHook(
                    at_tick=hook_data["at_tick"],
                    action=hook_data["action"],
                    params=hook_data.get("params", {}),
                )
            )

        return cls(
            name=data["name"],
            description=data.get("description", ""),
            base=data.get("base", {}),
            conditions=conditions,
            replicates=data.get("replicates", 5),
            seed_start=data.get("seed_start", 42),
            metrics=data.get(
                "metrics",
                [
                    "agents_alive_at_end",
                    "avg_survival_ticks",
                    "total_cooperation_events",
                ],
            ),
            output_dir=data.get("output_dir", "data/experiments"),
            formats=data.get("formats", ["csv", "markdown"]),
            record_trajectories=data.get("record_trajectories", False),
            trajectory_sample_count=data.get("trajectory_sample_count", 0),
            tick_hooks=top_level_hooks,
        )

    def expand_conditions(self) -> list[ExperimentCondition]:
        """Expand grid sweeps into individual conditions.

        Currently returns conditions as-is. Future versions could support
        grid syntax like {"num_agents": [3, 5, 10]} expanding to 3 conditions.

        Returns:
            List of conditions (currently unchanged)
        """
        # For now, just return conditions as-is
        # Future: implement grid expansion for parameter sweeps
        return self.conditions
