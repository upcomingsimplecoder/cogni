"""Analysis report generation.

Auto-generate comprehensive markdown reports from trajectory data.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.analysis.aggregator import DatasetAggregator
from src.analysis.behavioral import BehavioralAnalyzer
from src.analysis.correlations import TraitBehaviorAnalyzer
from src.analysis.survival import SurvivalAnalyzer

if TYPE_CHECKING:
    from src.trajectory.schema import TrajectoryDataset


class AnalysisReportGenerator:
    """Generate comprehensive analysis report from trajectory data."""

    def __init__(self):
        self.trait_analyzer = TraitBehaviorAnalyzer()
        self.survival_analyzer = SurvivalAnalyzer()
        self.behavioral_analyzer = BehavioralAnalyzer()
        self.aggregator = DatasetAggregator()

    def generate_run_report(self, dataset: TrajectoryDataset, output_path: str) -> None:
        """Generate markdown report for a single run."""
        lines = []

        # Header
        lines.append("# AUTOCOG Trajectory Analysis Report")
        lines.append("")
        lines.append(f"**Run ID:** {dataset.metadata.run_id}")
        lines.append(f"**Timestamp:** {dataset.metadata.timestamp}")
        lines.append(f"**Seed:** {dataset.metadata.seed}")
        lines.append(f"**Architecture:** {dataset.metadata.architecture or 'Unknown'}")
        lines.append("")

        # Run summary
        lines.append("## Run Summary")
        lines.append("")
        lines.append(f"- **Agents:** {dataset.metadata.num_agents}")
        lines.append(f"- **Max Ticks:** {dataset.metadata.max_ticks}")
        lines.append(f"- **Actual Ticks:** {dataset.metadata.actual_ticks}")
        lines.append(f"- **Agents Alive:** {dataset.metadata.final_state.get('agents_alive', 0)}")
        lines.append(f"- **Agents Dead:** {dataset.metadata.final_state.get('agents_dead', 0)}")
        lines.append(f"- **Emergence Events:** {len(dataset.emergence_events)}")
        lines.append("")

        # Trait-behavior correlations
        lines.append("## Trait-Behavior Correlations")
        lines.append("")
        trait_action_corr = self.trait_analyzer.trait_action_correlation(dataset)

        for trait, action_corrs in trait_action_corr.items():
            lines.append(f"### {trait}")
            lines.append("")
            # Sort by absolute correlation value
            sorted_actions = sorted(action_corrs.items(), key=lambda x: abs(x[1]), reverse=True)
            for action, corr in sorted_actions[:5]:  # Top 5
                lines.append(f"- **{action}:** {corr:.3f}")
            lines.append("")

        # Survival analysis
        lines.append("## Survival Analysis")
        lines.append("")

        trait_survival_corr = self.trait_analyzer.trait_survival_correlation(dataset)
        lines.append("### Trait-Survival Correlations")
        lines.append("")
        sorted_traits = sorted(trait_survival_corr.items(), key=lambda x: abs(x[1]), reverse=True)
        for trait, corr in sorted_traits:
            lines.append(f"- **{trait}:** {corr:.3f}")
        lines.append("")

        death_causes = self.survival_analyzer.death_cause_distribution(dataset)
        lines.append("### Death Causes")
        lines.append("")
        for cause, count in death_causes.items():
            lines.append(f"- **{cause}:** {count}")
        lines.append("")

        # Survival curve
        survival_curve = self.survival_analyzer.survival_curve(dataset)
        lines.append("### Survival Curve")
        lines.append("")
        lines.append("| Tick | Fraction Alive |")
        lines.append("|------|----------------|")
        # Sample every 10th point
        for i, (tick, fraction) in enumerate(survival_curve):
            if i % 10 == 0 or i == len(survival_curve) - 1:
                lines.append(f"| {tick} | {fraction:.2%} |")
        lines.append("")

        # Behavioral analysis
        lines.append("## Behavioral Analysis")
        lines.append("")

        archetype_fidelity = self.behavioral_analyzer.archetype_fidelity(dataset)
        lines.append("### Archetype Fidelity")
        lines.append("")
        lines.append("How closely each agent's behavior matches their archetype template:")
        lines.append("")

        # Group by archetype
        by_archetype: dict[str, list[float]] = {}
        for agent_id, fidelity in archetype_fidelity.items():
            agent_snapshots = [s for s in dataset.agent_snapshots if s.agent_id == agent_id]
            if agent_snapshots:
                archetype = agent_snapshots[0].archetype
                if archetype not in by_archetype:
                    by_archetype[archetype] = []
                by_archetype[archetype].append(fidelity)

        for archetype, fidelities in by_archetype.items():
            avg_fidelity = sum(fidelities) / len(fidelities)
            lines.append(
                f"- **{archetype}:** {avg_fidelity:.2%} (avg across {len(fidelities)} agents)"
            )
        lines.append("")

        # Agent profiles
        agent_ids = list(set(s.agent_id for s in dataset.agent_snapshots))[:5]  # First 5 agents
        if agent_ids:
            lines.append("### Sample Agent Behavioral Fingerprints")
            lines.append("")

            for agent_id in agent_ids:
                fingerprint = self.behavioral_analyzer.behavioral_fingerprint(dataset, agent_id)
                agent_name = next(
                    (s.agent_name for s in dataset.agent_snapshots if s.agent_id == agent_id),
                    agent_id,
                )

                lines.append(f"#### {agent_name} ({agent_id})")
                lines.append("")
                lines.append(f"- **Social Ratio:** {fingerprint['social_ratio']:.2%}")
                lines.append(f"- **Exploration Ratio:** {fingerprint['exploration_ratio']:.2%}")
                lines.append(f"- **Risk Index:** {fingerprint['risk_index']:.2%}")
                lines.append(f"- **Consistency:** {fingerprint['consistency']:.2%}")
                lines.append("")

                # Top actions
                top_actions = sorted(
                    fingerprint["action_distribution"].items(), key=lambda x: x[1], reverse=True
                )[:3]
                lines.append("**Top Actions:**")
                for action, freq in top_actions:
                    lines.append(f"- {action}: {freq:.2%}")
                lines.append("")

        # Write to file
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines))

    def generate_comparison_report(
        self, datasets: list[TrajectoryDataset], output_path: str
    ) -> None:
        """Compare multiple runs."""
        lines = []

        # Header
        lines.append("# AUTOCOG Multi-Run Comparison Report")
        lines.append("")
        lines.append(f"**Runs Analyzed:** {len(datasets)}")
        lines.append("")

        # Run summaries
        lines.append("## Run Summaries")
        lines.append("")
        lines.append("| Run ID | Architecture | Agents | Ticks | Alive | Dead |")
        lines.append("|--------|-------------|--------|-------|-------|------|")

        for dataset in datasets:
            run_id_short = dataset.metadata.run_id[:8]
            arch = dataset.metadata.architecture or "Unknown"
            agents = dataset.metadata.num_agents
            ticks = dataset.metadata.actual_ticks
            alive = dataset.metadata.final_state.get("agents_alive", 0)
            dead = dataset.metadata.final_state.get("agents_dead", 0)

            lines.append(f"| {run_id_short} | {arch} | {agents} | {ticks} | {alive} | {dead} |")

        lines.append("")

        # Aggregate dataset
        agg = self.aggregator.combine(datasets)

        # Architecture comparison
        lines.append("## Architecture Comparison")
        lines.append("")

        arch_comparison = self.aggregator.architecture_comparison(agg)
        lines.append("| Architecture | Runs | Avg Survival | Avg Alive | Total Emergence |")
        lines.append("|-------------|------|--------------|-----------|-----------------|")

        for arch, metrics in arch_comparison.items():
            lines.append(
                f"| {arch} | {metrics['num_runs']} | "
                f"{metrics['avg_survival_ticks']:.1f} | "
                f"{metrics['avg_agents_alive']:.1f} | "
                f"{metrics['total_emergence_events']} |"
            )

        lines.append("")

        # Optimal trait profile
        lines.append("## Optimal Trait Profile")
        lines.append("")
        lines.append("Trait values of top 25% survivors across all runs:")
        lines.append("")

        optimal = self.trait_analyzer.optimal_trait_profile(datasets)
        for trait, value in optimal.items():
            lines.append(f"- **{trait}:** {value:.3f}")

        lines.append("")

        # Survival predictors
        lines.append("## Survival Predictors")
        lines.append("")
        lines.append("Information gain for each trait predicting survival:")
        lines.append("")

        predictors = self.survival_analyzer.survival_predictors(datasets)
        sorted_predictors = sorted(predictors.items(), key=lambda x: x[1], reverse=True)

        for trait, gain in sorted_predictors:
            lines.append(f"- **{trait}:** {gain:.3f}")

        lines.append("")

        # Personality-behavior matrix
        lines.append("## Personality-Behavior Matrix")
        lines.append("")

        matrix = self.aggregator.personality_behavior_matrix(agg)

        for trait, buckets in list(matrix.items())[:3]:  # Show first 3 traits
            lines.append(f"### {trait}")
            lines.append("")

            # Get all action types across all buckets
            all_actions = set()
            for bucket_actions in buckets.values():
                all_actions.update(bucket_actions.keys())

            # Show top 5 actions
            action_totals = {}
            for action in all_actions:
                total = sum(buckets[b].get(action, 0) for b in ["low", "med", "high"])
                action_totals[action] = total

            top_actions = sorted(action_totals.items(), key=lambda x: x[1], reverse=True)[:5]

            lines.append("| Action | Low | Med | High |")
            lines.append("|--------|-----|-----|------|")

            for action, _ in top_actions:
                low = buckets.get("low", {}).get(action, 0)
                med = buckets.get("med", {}).get(action, 0)
                high = buckets.get("high", {}).get(action, 0)
                lines.append(f"| {action} | {low:.2%} | {med:.2%} | {high:.2%} |")

            lines.append("")

        # Write to file
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines))
