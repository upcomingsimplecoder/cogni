"""Plugin system for AUTOCOG.

Provides a registry-based plugin architecture for extending:
- Decision strategies
- Actions
- Agent archetypes
- Evaluation strategies
- Metrics
"""

from __future__ import annotations

from src.plugins.loader import PluginLoader
from src.plugins.registry import PluginRegistry

__all__ = ["PluginRegistry", "PluginLoader"]
