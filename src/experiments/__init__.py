"""Experiment runner for systematic simulation studies.

Provides configuration, execution, analysis, and reporting tools for
running and analyzing multiple simulation conditions with replicates.
"""

from __future__ import annotations

from src.experiments.analysis import ConditionSummary, ResultAnalyzer
from src.experiments.config import ExperimentCondition, ExperimentConfig
from src.experiments.provenance import ExperimentProvenance, capture_provenance
from src.experiments.registry import ExperimentRegistry
from src.experiments.report import ReportGenerator
from src.experiments.runner import ExperimentRunner, RunResult

__all__ = [
    "ExperimentConfig",
    "ExperimentCondition",
    "ExperimentRunner",
    "RunResult",
    "ResultAnalyzer",
    "ConditionSummary",
    "ReportGenerator",
    "ExperimentProvenance",
    "capture_provenance",
    "ExperimentRegistry",
]
