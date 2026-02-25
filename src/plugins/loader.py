"""Plugin loader for discovering and loading AUTOCOG plugins.

Scans directories for Python modules and registers their components
via the PluginRegistry.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

from src.plugins.registry import PluginRegistry

logger = logging.getLogger(__name__)


class PluginLoader:
    """Discover and load plugins from directories."""

    @classmethod
    def load_all(cls, plugin_dirs: list[str] | None = None) -> dict[str, Any]:
        """Load all plugins from specified directories.

        Args:
            plugin_dirs: List of directory paths to scan. If None, uses default
                        'plugins/' directory at project root.

        Returns:
            Dict with loaded plugin counts by category
        """
        if plugin_dirs is None:
            # Default to 'plugins/' directory at project root
            plugin_dirs = ["plugins"]

        stats = {
            "strategies": 0,
            "actions": 0,
            "archetypes": 0,
            "evaluations": 0,
            "metrics": 0,
            "errors": [],
        }

        for dir_path in plugin_dirs:
            path = Path(dir_path)
            if not path.exists():
                logger.warning(f"Plugin directory not found: {dir_path}")
                continue

            if not path.is_dir():
                logger.warning(f"Plugin path is not a directory: {dir_path}")
                continue

            loaded = cls.load_from_directory(str(path))
            stats["strategies"] = stats["strategies"] + loaded["strategies"]
            stats["actions"] = stats["actions"] + loaded["actions"]
            stats["archetypes"] = stats["archetypes"] + loaded["archetypes"]
            stats["evaluations"] = stats["evaluations"] + loaded["evaluations"]
            stats["metrics"] = stats["metrics"] + loaded["metrics"]
            if isinstance(stats["errors"], list) and isinstance(loaded["errors"], list):
                stats["errors"].extend(loaded["errors"])

        logger.info(
            f"Loaded {stats['strategies']} strategies, "
            f"{stats['actions']} actions, "
            f"{stats['archetypes']} archetypes, "
            f"{stats['evaluations']} evaluations, "
            f"{stats['metrics']} metrics"
        )

        return stats

    @classmethod
    def load_from_directory(cls, directory: str) -> dict[str, Any]:
        """Load all Python files from a directory.

        Args:
            directory: Path to directory containing plugin files

        Returns:
            Dict with loaded plugin counts by category
        """
        dir_path = Path(directory)
        stats = {
            "strategies": 0,
            "actions": 0,
            "archetypes": 0,
            "evaluations": 0,
            "metrics": 0,
            "errors": [],
        }

        # Find all .py files (excluding __init__.py and files starting with _)
        py_files = [
            f
            for f in dir_path.glob("*.py")
            if f.name != "__init__.py" and not f.name.startswith("_")
        ]

        logger.debug(f"Found {len(py_files)} plugin files in {directory}")

        for py_file in py_files:
            try:
                # Capture counts before loading
                before = {
                    "strategies": len(PluginRegistry.all_strategies()),
                    "actions": len(PluginRegistry._actions),
                    "archetypes": len(PluginRegistry._archetypes),
                    "evaluations": len(PluginRegistry._evaluations),
                    "metrics": len(PluginRegistry._metrics),
                }

                cls.load_plugin_file(str(py_file))

                # Calculate what was added
                after = {
                    "strategies": len(PluginRegistry.all_strategies()),
                    "actions": len(PluginRegistry._actions),
                    "archetypes": len(PluginRegistry._archetypes),
                    "evaluations": len(PluginRegistry._evaluations),
                    "metrics": len(PluginRegistry._metrics),
                }

                # Cast stats values to int for safe arithmetic
                stats["strategies"] = (
                    int(stats["strategies"]) if isinstance(stats["strategies"], int) else 0
                ) + (after["strategies"] - before["strategies"])
                stats["actions"] = (
                    int(stats["actions"]) if isinstance(stats["actions"], int) else 0
                ) + (after["actions"] - before["actions"])
                stats["archetypes"] = (
                    int(stats["archetypes"]) if isinstance(stats["archetypes"], int) else 0
                ) + (after["archetypes"] - before["archetypes"])
                stats["evaluations"] = (
                    int(stats["evaluations"]) if isinstance(stats["evaluations"], int) else 0
                ) + (after["evaluations"] - before["evaluations"])
                stats["metrics"] = (
                    int(stats["metrics"]) if isinstance(stats["metrics"], int) else 0
                ) + (after["metrics"] - before["metrics"])

                logger.debug(f"Loaded plugin: {py_file.name}")

            except Exception as e:
                error_msg = f"Failed to load {py_file.name}: {e}"
                logger.error(error_msg)
                if isinstance(stats["errors"], list):
                    stats["errors"].append(error_msg)

        return stats

    @classmethod
    def load_plugin_file(cls, file_path: str) -> None:
        """Load a single plugin file.

        Args:
            file_path: Path to Python file to load

        Raises:
            ImportError: If file cannot be loaded
            FileNotFoundError: If file does not exist
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Plugin file not found: {file_path}")

        if not path.is_file() or path.suffix != ".py":
            raise ValueError(f"Not a Python file: {file_path}")

        # Generate module name from file path
        module_name = f"autocog_plugin_{path.stem}"

        # Load module dynamically
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load spec for {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        logger.debug(f"Executed plugin module: {module_name}")

    @classmethod
    def load_plugin(cls, module_name: str) -> None:
        """Load a plugin by module name (for installed packages).

        Args:
            module_name: Python module name (e.g., 'autocog_plugin_xyz')

        Raises:
            ImportError: If module cannot be imported
        """
        try:
            importlib.import_module(module_name)
            logger.debug(f"Loaded plugin module: {module_name}")
        except ImportError as e:
            raise ImportError(f"Failed to import plugin {module_name}: {e}") from e
