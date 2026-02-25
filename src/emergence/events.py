"""Emergent behavior events detected during simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EmergentEvent:
    """A detected emergent behavior pattern."""

    tick: int
    pattern_type: str  # "cluster", "sharing_network", "territory", "specialization"
    agents_involved: list[object] = field(default_factory=list)  # AgentIDs
    description: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[Tick {self.tick}] {self.pattern_type}: {self.description}"
