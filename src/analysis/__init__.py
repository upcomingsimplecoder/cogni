"""Data analysis toolkit for AUTOCOG trajectories.

This package provides tools to analyze trajectory data and generate research insights:
- TraitBehaviorAnalyzer: Correlate personality traits with behavioral outcomes
- SurvivalAnalyzer: Survival statistics and predictors
- BehavioralAnalyzer: Behavioral fingerprints and pattern detection
- DatasetAggregator: Combine multiple runs for cross-run analysis
- AnalysisReportGenerator: Auto-generate comprehensive reports
"""

from src.analysis.aggregator import AggregateDataset, DatasetAggregator
from src.analysis.behavioral import BehavioralAnalyzer
from src.analysis.correlations import TraitBehaviorAnalyzer
from src.analysis.reports import AnalysisReportGenerator
from src.analysis.survival import SurvivalAnalyzer

__all__ = [
    "TraitBehaviorAnalyzer",
    "SurvivalAnalyzer",
    "BehavioralAnalyzer",
    "DatasetAggregator",
    "AggregateDataset",
    "AnalysisReportGenerator",
]
