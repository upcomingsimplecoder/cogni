"""Experiment provenance tracking for reproducible research."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class ExperimentProvenance:
    """Everything needed to reproduce this experiment."""

    experiment_id: str  # UUID
    timestamp: str  # ISO 8601
    git_commit: str | None  # HEAD SHA (if in a git repo)
    git_dirty: bool  # True if uncommitted changes
    python_version: str  # e.g., "3.12.1"
    platform_info: str  # e.g., "Linux-6.1-x86_64"
    autocog_version: str  # From pyproject.toml or fallback
    config_yaml_path: str  # Path to experiment YAML
    config_yaml_hash: str  # SHA256 of YAML content
    config_resolved: dict  # Full resolved config (after defaults)
    seed_range: list[int]  # Seeds used
    duration_seconds: float  # Total wall time
    dependencies: dict  # Pinned versions of key deps

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ExperimentProvenance:
        """Create from dictionary."""
        return cls(**data)


def capture_provenance(
    experiment_id: str,
    config_yaml_path: str,
    config_resolved: dict,
    seed_range: list[int],
    duration_seconds: float = 0.0,
) -> ExperimentProvenance:
    """Capture full provenance for current experiment.

    Args:
        experiment_id: Unique identifier for this experiment
        config_yaml_path: Path to the YAML config file
        config_resolved: Full resolved configuration dictionary
        seed_range: List of seeds used in this experiment
        duration_seconds: Total wall time for experiment

    Returns:
        ExperimentProvenance object with all captured metadata
    """
    return ExperimentProvenance(
        experiment_id=experiment_id,
        timestamp=datetime.now(UTC).isoformat(),
        git_commit=_get_git_commit(),
        git_dirty=_is_git_dirty(),
        python_version=platform.python_version(),
        platform_info=platform.platform(),
        autocog_version=_get_autocog_version(),
        config_yaml_path=str(config_yaml_path),
        config_yaml_hash=_hash_file(config_yaml_path),
        config_resolved=config_resolved,
        seed_range=seed_range,
        duration_seconds=duration_seconds,
        dependencies=_get_dependency_versions(),
    )


def _get_git_commit() -> str | None:
    """Get current git HEAD SHA, or None if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _is_git_dirty() -> bool:
    """Check if working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False


def _get_autocog_version() -> str:
    """Get AUTOCOG version from pyproject.toml or fallback."""
    try:
        toml_path = Path(__file__).parent.parent.parent / "pyproject.toml"
        if toml_path.exists():
            content = toml_path.read_text()
            for line in content.splitlines():
                if line.strip().startswith("version"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return "0.1.0"


def _hash_file(path: str) -> str:
    """SHA256 hash of file contents.

    Args:
        path: Path to file to hash

    Returns:
        Hex digest of SHA256 hash, or "file_not_found" if file doesn't exist
    """
    try:
        content = Path(path).read_bytes()
        return hashlib.sha256(content).hexdigest()
    except (FileNotFoundError, OSError):
        return "file_not_found"


def _get_dependency_versions() -> dict:
    """Get pinned versions of key dependencies.

    Returns:
        Dictionary mapping package name to version string
    """
    deps = {}
    for pkg in ["pydantic", "pyyaml", "rich", "openai"]:
        try:
            mod = __import__(pkg)
            deps[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            deps[pkg] = "not_installed"
    return deps
