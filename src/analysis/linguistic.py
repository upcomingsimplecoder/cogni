"""Linguistic evolution analysis and emergent language metrics.

Analyze language dynamics from LanguageEngine snapshots and produce
research-grade metrics about vocabulary growth, convention formation,
communication efficiency, dialect emergence, and innovation patterns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.communication.language_engine import LanguageEngine


class LinguisticEvolutionAnalyzer:
    """Analyze emergent language dynamics and evolutionary patterns.

    Tracks linguistic phenomena over time:
    - Vocabulary growth and diversification
    - Convention formation and stabilization
    - Communication success trends
    - Dialect emergence and divergence
    - Innovation vs. adoption dynamics
    """

    def __init__(self) -> None:
        """Initialize with empty history."""
        self._history: list[dict[str, Any]] = []

    def record_tick(self, language_engine: LanguageEngine, tick: int) -> None:
        """Record a snapshot from the language engine.

        Captures the current state of the linguistic system for later analysis.
        Should be called once per tick after LanguageEngine.tick().

        Args:
            language_engine: The LanguageEngine instance to snapshot
            tick: Current simulation tick
        """
        snapshot = {
            "tick": tick,
            "snapshots": list(language_engine.snapshots),
            "established_conventions": list(language_engine.get_established_conventions()),
            "all_conventions": list(language_engine.get_all_conventions()),
            "language_stats": language_engine.get_language_stats(),
        }
        self._history.append(snapshot)

    def vocabulary_growth_curve(self) -> list[tuple[int, int]]:
        """Compute vocabulary growth over time.

        Returns list of (tick, total_unique_symbols) showing how many
        distinct symbols exist across all agents at each point in time.

        Returns:
            List of (tick, unique_symbol_count) tuples
        """
        if not self._history:
            return []

        curve: list[tuple[int, int]] = []

        for record in self._history:
            tick = record["tick"]
            # Get the most recent snapshot for this tick
            snapshots = record["snapshots"]
            if snapshots:
                unique_symbols = snapshots[-1].unique_symbols
                curve.append((tick, unique_symbols))

        return curve

    def convention_formation_timeline(self) -> list[dict[str, Any]]:
        """Track when and how conventions first appeared.

        Returns chronological list of convention formation events,
        showing when each shared meaning-symbol mapping emerged.

        Returns:
            List of dicts with keys:
            - tick: When convention first appeared
            - symbol: Symbol form
            - meaning: Semantic referent
            - adopters: Number of agents sharing this convention
        """
        if not self._history:
            return []

        timeline: list[dict[str, Any]] = []
        seen_conventions: set[str] = set()  # Track "(symbol, meaning)" pairs

        for record in self._history:
            conventions = record["established_conventions"]

            for conv in conventions:
                # Create unique key for this convention
                conv_key = (
                    f"{conv.symbol_form}:{conv.meaning.meaning_type.value}:{conv.meaning.referent}"
                )

                # Only record first appearance
                if conv_key not in seen_conventions:
                    seen_conventions.add(conv_key)
                    timeline.append(
                        {
                            "tick": conv.first_observed_tick,
                            "symbol": conv.symbol_form,
                            "meaning": f"{conv.meaning.meaning_type.value}:{conv.meaning.referent}",
                            "adopters": conv.adoption_count,
                        }
                    )

        # Sort by tick
        timeline.sort(key=lambda x: x["tick"])
        return timeline

    def communication_success_over_time(self) -> list[tuple[int, float]]:
        """Compute communication success rate over time.

        Returns the proportion of successful communications at each tick.
        Success = listener correctly interprets at least one symbol.

        Returns:
            List of (tick, success_rate) tuples where success_rate is 0.0-1.0
        """
        if not self._history:
            return []

        success_curve: list[tuple[int, float]] = []

        for record in self._history:
            tick = record["tick"]
            snapshots = record["snapshots"]

            if snapshots:
                success_rate = snapshots[-1].communication_success_rate
                success_curve.append((tick, success_rate))

        return success_curve

    def dialect_divergence_over_time(self) -> list[tuple[int, int]]:
        """Track dialect group formation over time.

        Dialects form when subpopulations develop distinct vocabularies
        with limited overlap. More dialect groups = more fragmentation.

        Returns:
            List of (tick, dialect_count) tuples
        """
        if not self._history:
            return []

        dialect_curve: list[tuple[int, int]] = []

        for record in self._history:
            tick = record["tick"]
            snapshots = record["snapshots"]

            if snapshots:
                dialect_count = snapshots[-1].dialect_count
                dialect_curve.append((tick, dialect_count))

        return dialect_curve

    def innovation_rate_over_time(self) -> list[tuple[int, int]]:
        """Track new symbol creation over time.

        Innovations are new symbols created by agents for meanings
        they lack words for. High innovation suggests active exploration
        or communication pressure.

        Returns:
            List of (tick, innovation_count) tuples
        """
        if not self._history:
            return []

        innovation_curve: list[tuple[int, int]] = []

        for record in self._history:
            tick = record["tick"]
            snapshots = record["snapshots"]

            if snapshots:
                innovations = snapshots[-1].innovations_this_tick
                innovation_curve.append((tick, innovations))

        return innovation_curve

    def generate_report(self) -> str:
        """Generate a markdown research report on linguistic evolution.

        Synthesizes all tracked metrics into a human-readable summary
        suitable for research documentation or trajectory analysis.

        Returns:
            Markdown-formatted report string
        """
        if not self._history:
            return "# Linguistic Evolution Report\n\nNo data recorded.\n"

        # Gather all metrics
        vocab_curve = self.vocabulary_growth_curve()
        conventions = self.convention_formation_timeline()
        success_curve = self.communication_success_over_time()
        dialect_curve = self.dialect_divergence_over_time()
        innovation_curve = self.innovation_rate_over_time()

        # Build report sections
        lines = ["# Linguistic Evolution Report\n"]

        # Vocabulary growth summary
        lines.append("## Vocabulary Growth\n")
        if vocab_curve:
            initial_vocab = vocab_curve[0][1]
            final_vocab = vocab_curve[-1][1]
            growth = final_vocab - initial_vocab
            growth_rate = (growth / initial_vocab * 100) if initial_vocab > 0 else 0.0

            lines.append(f"- **Initial vocabulary**: {initial_vocab} unique symbols")
            lines.append(f"- **Final vocabulary**: {final_vocab} unique symbols")
            lines.append(f"- **Total growth**: {growth} symbols ({growth_rate:.1f}% increase)")
            lines.append(f"- **Peak vocabulary**: {max(v for _, v in vocab_curve)} symbols\n")
        else:
            lines.append("No vocabulary data recorded.\n")

        # Convention formation
        lines.append("## Convention Formation\n")
        if conventions:
            lines.append(f"- **Total conventions established**: {len(conventions)}")

            # Find earliest convention
            earliest = conventions[0]
            lines.append(
                f"- **First convention**: '{earliest['symbol']}' "
                f"for {earliest['meaning']} (tick {earliest['tick']})"
            )

            # Find most adopted convention
            most_adopted = max(conventions, key=lambda c: c["adopters"])
            lines.append(
                f"- **Most widely adopted**: '{most_adopted['symbol']}' "
                f"with {most_adopted['adopters']} adopters\n"
            )

            # Recent conventions (last 5)
            lines.append("### Recent Conventions\n")
            recent = conventions[-5:] if len(conventions) > 5 else conventions
            for conv in recent:
                lines.append(
                    f"- Tick {conv['tick']}: '{conv['symbol']}' → "
                    f"{conv['meaning']} ({conv['adopters']} adopters)"
                )
            lines.append("")
        else:
            lines.append("No conventions formed.\n")

        # Communication efficiency
        lines.append("## Communication Efficiency\n")
        if success_curve:
            avg_success = sum(rate for _, rate in success_curve) / len(success_curve)
            initial_success = success_curve[0][1]
            final_success = success_curve[-1][1]

            lines.append(f"- **Average success rate**: {avg_success:.1%}")
            lines.append(f"- **Initial success rate**: {initial_success:.1%}")
            lines.append(f"- **Final success rate**: {final_success:.1%}")

            improvement = final_success - initial_success
            if improvement > 0.05:
                lines.append(f"- **Trend**: Improving (+{improvement:.1%})")
            elif improvement < -0.05:
                lines.append(f"- **Trend**: Declining ({improvement:.1%})")
            else:
                lines.append("- **Trend**: Stable")
            lines.append("")
        else:
            lines.append("No communication data recorded.\n")

        # Dialect analysis
        lines.append("## Dialect Formation\n")
        if dialect_curve:
            initial_dialects = dialect_curve[0][1]
            final_dialects = dialect_curve[-1][1]
            max_dialects = max(count for _, count in dialect_curve)

            lines.append(f"- **Initial dialect groups**: {initial_dialects}")
            lines.append(f"- **Final dialect groups**: {final_dialects}")
            lines.append(f"- **Peak fragmentation**: {max_dialects} dialect groups")

            if final_dialects > initial_dialects:
                lines.append("- **Pattern**: Divergence (increasing fragmentation)")
            elif final_dialects < initial_dialects:
                lines.append("- **Pattern**: Convergence (dialect merging)")
            else:
                lines.append("- **Pattern**: Stable dialect structure")
            lines.append("")
        else:
            lines.append("No dialect data recorded.\n")

        # Innovation dynamics
        lines.append("## Innovation vs. Adoption\n")
        if innovation_curve:
            total_innovations = sum(count for _, count in innovation_curve)
            avg_innovations = total_innovations / len(innovation_curve)
            peak_innovation = max(count for _, count in innovation_curve)

            lines.append(f"- **Total innovations**: {total_innovations} new symbols created")
            lines.append(f"- **Average per tick**: {avg_innovations:.2f} innovations")
            lines.append(f"- **Peak innovation**: {peak_innovation} symbols in a single tick")

            # Innovation trend (early vs late)
            if len(innovation_curve) >= 10:
                early_avg = sum(c for _, c in innovation_curve[:5]) / 5
                late_avg = sum(c for _, c in innovation_curve[-5:]) / 5

                if late_avg > early_avg * 1.2:
                    lines.append("- **Trend**: Accelerating innovation")
                elif late_avg < early_avg * 0.8:
                    lines.append("- **Trend**: Declining innovation (stabilization)")
                else:
                    lines.append("- **Trend**: Steady innovation rate")
            lines.append("")
        else:
            lines.append("No innovation data recorded.\n")

        # Summary
        lines.append("## Summary\n")
        if vocab_curve and success_curve and conventions:
            # Determine overall linguistic health
            final_success = success_curve[-1][1] if success_curve else 0.0
            convention_count = len(conventions)

            if final_success > 0.6 and convention_count > 5:
                health = "healthy and robust"
            elif final_success > 0.4 and convention_count > 2:
                health = "developing"
            else:
                health = "nascent"

            lines.append(
                f"The emergent language system is **{health}** with "
                f"{len(conventions)} established conventions and a "
                f"{final_success:.1%} communication success rate. "
            )

            if vocab_curve[-1][1] > vocab_curve[0][1] * 2:
                lines.append(
                    "Vocabulary growth has been **rapid**, suggesting "
                    "active linguistic innovation and population pressure."
                )
            elif vocab_curve[-1][1] > vocab_curve[0][1] * 1.2:
                lines.append(
                    "Vocabulary growth has been **steady**, indicating "
                    "balanced innovation and stabilization."
                )
            else:
                lines.append(
                    "Vocabulary has remained **relatively stable**, "
                    "suggesting established conventions dominate."
                )
        else:
            lines.append("Insufficient data for comprehensive assessment.")

        lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize all linguistic evolution data for trajectory export.

        Returns:
            Dictionary containing all tracked metrics and raw history
        """
        return {
            "vocabulary_growth": self.vocabulary_growth_curve(),
            "convention_timeline": self.convention_formation_timeline(),
            "communication_success": self.communication_success_over_time(),
            "dialect_divergence": self.dialect_divergence_over_time(),
            "innovation_rate": self.innovation_rate_over_time(),
            "raw_history": self._history,
        }
