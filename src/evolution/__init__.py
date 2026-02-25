"""Genetic and cultural evolution systems.

This package implements agent evolution mechanics:
- GeneticSystem: Trait inheritance, crossover, and mutation
- ReproductionSystem: Reproduction conditions and offspring creation
- CulturalTransmission: Social learning and norm propagation
- CulturalNorm: Shared behavioral patterns
- NormDetector: Emergent norm detection from behavior patterns
- PopulationManager: Birth, death, and population control
- LineageTracker: Family tree and trait drift tracking
"""

from __future__ import annotations

from src.evolution.culture import (
    CulturalNorm,
    CulturalTransmission,
    NormDetector,
)
from src.evolution.genetics import GeneticSystem
from src.evolution.lineage import LineageNode, LineageTracker
from src.evolution.population import PopulationManager
from src.evolution.reproduction import ReproductionSystem

__all__ = [
    "GeneticSystem",
    "ReproductionSystem",
    "CulturalTransmission",
    "CulturalNorm",
    "NormDetector",
    "PopulationManager",
    "LineageTracker",
    "LineageNode",
]
