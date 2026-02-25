"""Dynamic team formation and coalition management.

This package implements multi-agent coalition mechanics:
- Coalition: Groups with shared goals and role assignments
- CoalitionManager: Creation, membership, and lifecycle tracking
- CoalitionFormation: Decision logic for proposing and joining coalitions
- CoalitionCoordinator: Role assignment and coordinated action suggestions
- DissolutionDetector: Monitors cohesion and triggers coalition breakup
"""

from __future__ import annotations

from src.social.coalition import Coalition, CoalitionManager
from src.social.coordination import CoalitionCoordinator
from src.social.dissolution import DissolutionDetector
from src.social.formation import CoalitionFormation

__all__ = [
    "Coalition",
    "CoalitionManager",
    "CoalitionFormation",
    "CoalitionCoordinator",
    "DissolutionDetector",
]
