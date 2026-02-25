#!/usr/bin/env python3
"""Auto-analysis script for overnight experiment results.

Generates publication figures from CSV results and trajectory data.
Uses only matplotlib + numpy (no pandas/scipy/seaborn).

Usage:
    python scripts/analyze_overnight.py
"""

import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # No display needed
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

FIGURES_DIR = Path("data/figures")
ALIGNMENT_DIR = Path("data/experiments/alignment")


# ======== HELPERS ========


def read_csv(path: str) -> list[dict[str, str]]:
    """Read CSV into list of dicts. Return [] if file doesn't exist."""
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def group_by_condition(rows: list[dict]) -> dict[str, list[dict]]:
    """Group rows by 'condition' field."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        cond = row.get("condition", "unknown")
        if cond not in groups:
            groups[cond] = []
        groups[cond].append(row)
    return groups


def compute_stats(values: list[float]) -> dict[str, float]:
    """Compute mean, std, n, ci95 bounds using numpy."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "n": 0, "ci95_lower": 0.0, "ci95_upper": 0.0}
    arr = np.array(values)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    margin = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "n": n,
        "ci95_lower": mean - margin,
        "ci95_upper": mean + margin,
    }


def find_csv(directory: str) -> str | None:
    """Find *results.csv in a directory. Return path or None."""
    d = Path(directory)
    if not d.exists():
        return None
    for f in sorted(d.iterdir()):
        if f.name.endswith("results.csv") or f.name == "results.csv":
            return str(f)
    # Fallback: any CSV
    for f in sorted(d.iterdir()):
        if f.suffix == ".csv":
            return str(f)
    return None


def get_metric_values(rows: list[dict], metric: str) -> list[float]:
    """Extract float values for a metric from rows."""
    values = []
    for row in rows:
        val = row.get(metric)
        if val is not None:
            try:
                values.append(float(val))
            except (ValueError, TypeError):
                pass
    return values


def parse_aggressor_count(condition_name: str) -> int | None:
    """Parse aggressor count from condition name like '3_aggressors' or '3_aggressor'."""
    parts = condition_name.split("_")
    if len(parts) >= 2 and parts[1].startswith("aggressor"):
        try:
            return int(parts[0])
        except ValueError:
            pass
    return None


# ======== FIGURE GENERATORS ========


def plot_phase_transition() -> str | None:
    """Generate phase transition plot from minority sweep data."""
    # Try experiment 06 first, fallback to 04
    csv_path = find_csv(str(ALIGNMENT_DIR / "minority_fine_sweep"))
    if csv_path is None:
        csv_path = find_csv(str(ALIGNMENT_DIR / "minority_influence"))
    if csv_path is None:
        print("  [SKIP] No minority influence/fine sweep CSV found")
        return None

    rows = read_csv(csv_path)
    if not rows:
        print("  [SKIP] Empty CSV for phase transition")
        return None

    groups = group_by_condition(rows)

    # Extract data points: (aggressor_fraction, coop_stats, alive_stats)
    points = []
    for cond_name, cond_rows in groups.items():
        count = parse_aggressor_count(cond_name)
        if count is None:
            continue
        total_agents = 8  # standard
        fraction = count / total_agents

        coop_values = get_metric_values(cond_rows, "cooperation_ratio")
        alive_values = get_metric_values(cond_rows, "agents_alive_at_end")

        if coop_values:
            points.append((fraction, compute_stats(coop_values), compute_stats(alive_values)))

    if not points:
        print("  [SKIP] No valid data points for phase transition")
        return None

    points.sort(key=lambda p: p[0])
    fractions = [p[0] for p in points]
    coop_means = [p[1]["mean"] for p in points]
    coop_ci_lower = [p[1]["ci95_lower"] for p in points]
    coop_ci_upper = [p[1]["ci95_upper"] for p in points]
    alive_means = [p[2]["mean"] for p in points]

    # Find critical threshold (steepest drop)
    max_drop = 0.0
    critical_idx = 0
    for i in range(1, len(coop_means)):
        drop = coop_means[i - 1] - coop_means[i]
        if drop > max_drop:
            max_drop = drop
            critical_idx = i

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_coop = "#2196F3"
    color_alive = "#FF9800"

    # Cooperation ratio
    ax1.set_xlabel("Aggressor Fraction", fontsize=13)
    ax1.set_ylabel("Cooperation Ratio", fontsize=13, color=color_coop)
    ax1.errorbar(
        fractions, coop_means,
        yerr=[
            [m - lo for m, lo in zip(coop_means, coop_ci_lower)],
            [hi - m for m, hi in zip(coop_means, coop_ci_upper)],
        ],
        color=color_coop, marker="o", linewidth=2, markersize=8,
        capsize=4, label="Cooperation Ratio",
    )
    ax1.tick_params(axis="y", labelcolor=color_coop)
    ax1.set_ylim(-0.05, 1.05)

    # Critical threshold
    if len(fractions) > 1 and max_drop > 0.05:
        critical_x = (fractions[critical_idx - 1] + fractions[critical_idx]) / 2
        ax1.axvline(x=critical_x, color="red", linestyle="--", alpha=0.7, linewidth=1.5)
        ax1.annotate(
            f"Critical threshold\n({critical_x:.2f})",
            xy=(critical_x, coop_means[critical_idx]),
            xytext=(critical_x + 0.08, coop_means[critical_idx] + 0.15),
            fontsize=10, color="red",
            arrowprops=dict(arrowstyle="->", color="red", alpha=0.7),
        )

    # Agents alive (secondary axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Agents Alive", fontsize=13, color=color_alive)
    ax2.plot(
        fractions, alive_means,
        color=color_alive, marker="s", linewidth=2, markersize=6,
        alpha=0.7, linestyle="--", label="Agents Alive",
    )
    ax2.tick_params(axis="y", labelcolor=color_alive)

    # Title and legend
    n_per_point = points[0][1]["n"] if points else 0
    plt.title(f"Phase Transition in Minority Influence (n={n_per_point}/condition)", fontsize=14, fontweight="bold")
    fig.tight_layout()

    out_path = str(FIGURES_DIR / "phase_transition.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path}")
    return out_path


def plot_contagion_spread() -> str | None:
    """Generate contagion spread plot from trajectory data."""
    traj_dir = ALIGNMENT_DIR / "value_drift" / "trajectories"
    if not traj_dir.exists():
        print("  [SKIP] No value_drift trajectory data found")
        return None

    # Find control and corrupt_early trajectory dirs
    control_dir = None
    corrupt_dir = None
    for d in sorted(traj_dir.iterdir()):
        if not d.is_dir():
            continue
        if d.name.startswith("control") and control_dir is None:
            control_dir = d
        elif d.name.startswith("corrupt_early") and corrupt_dir is None:
            corrupt_dir = d

    if control_dir is None or corrupt_dir is None:
        print("  [SKIP] Missing control or corrupt_early trajectory dirs")
        return None

    def extract_trait_timeseries(run_dir: Path) -> dict[str, dict[int, float]]:
        """Extract cooperation_tendency per agent per tick from trajectory JSONL."""
        traj_file = run_dir / "trajectory.jsonl"
        if not traj_file.exists():
            return {}

        agent_data: dict[str, dict[int, float]] = {}  # agent_id -> {tick: coop_value}
        try:
            with open(traj_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if record.get("type") != "agent_snapshot":
                        continue
                    agent_id = str(record.get("agent_id", ""))
                    tick = record.get("tick", 0)
                    traits = record.get("traits", {})
                    coop = traits.get("cooperation_tendency")
                    if coop is not None:
                        if agent_id not in agent_data:
                            agent_data[agent_id] = {}
                        agent_data[agent_id][tick] = float(coop)
        except Exception as e:
            print(f"  Warning: Error reading {traj_file}: {e}")
        return agent_data

    control_data = extract_trait_timeseries(control_dir)
    corrupt_data = extract_trait_timeseries(corrupt_dir)

    if not control_data or not corrupt_data:
        print("  [SKIP] Empty trajectory data for contagion plot")
        return None

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    colors = plt.cm.Set2(np.linspace(0, 1, 8))

    # Control subplot
    ax1.set_title("Control (No Corruption)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Cooperation Tendency", fontsize=11)
    for i, (agent_id, ticks_data) in enumerate(sorted(control_data.items())):
        ticks = sorted(ticks_data.keys())
        values = [ticks_data[t] for t in ticks]
        ax1.plot(ticks, values, color=colors[i % 8], alpha=0.7, linewidth=1.2)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)

    # Corrupt subplot
    ax2.set_title("Corrupt Early (Agent Corrupted at Tick 50)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Cooperation Tendency", fontsize=11)
    ax2.set_xlabel("Tick", fontsize=11)
    for i, (agent_id, ticks_data) in enumerate(sorted(corrupt_data.items())):
        ticks = sorted(ticks_data.keys())
        values = [ticks_data[t] for t in ticks]
        # Color first agent (corrupted) differently
        color = "red" if i == 0 else colors[i % 8]
        lw = 2.5 if i == 0 else 1.2
        alpha = 1.0 if i == 0 else 0.7
        label = "Corrupted Agent" if i == 0 else None
        ax2.plot(ticks, values, color=color, alpha=alpha, linewidth=lw, label=label)

    ax2.axvline(x=50, color="red", linestyle="--", alpha=0.5, linewidth=1.5, label="Corruption Event")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left", fontsize=9)

    fig.suptitle("Value Drift Contagion", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    out_path = str(FIGURES_DIR / "contagion_spread.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path}")
    return out_path


def plot_architecture_resilience() -> str | None:
    """Generate architecture resilience bar chart."""
    csv_path = find_csv(str(ALIGNMENT_DIR / "architecture_resilience"))
    if csv_path is None:
        print("  [SKIP] No architecture resilience CSV found")
        return None

    rows = read_csv(csv_path)
    if not rows:
        print("  [SKIP] Empty architecture resilience CSV")
        return None

    groups = group_by_condition(rows)
    conditions = sorted(groups.keys())

    coop_stats = [compute_stats(get_metric_values(groups[c], "cooperation_ratio")) for c in conditions]
    trait_stats = [compute_stats(get_metric_values(groups[c], "trait_evolution_magnitude")) for c in conditions]

    x = np.arange(len(conditions))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))

    bars1 = ax1.bar(
        x - width / 2,
        [s["mean"] for s in coop_stats],
        width,
        yerr=[1.96 * s["std"] / math.sqrt(s["n"]) if s["n"] > 1 else 0 for s in coop_stats],
        label="Cooperation Ratio",
        color="#2196F3",
        capsize=4,
    )

    ax2 = ax1.twinx()
    bars2 = ax2.bar(
        x + width / 2,
        [s["mean"] for s in trait_stats],
        width,
        yerr=[1.96 * s["std"] / math.sqrt(s["n"]) if s["n"] > 1 else 0 for s in trait_stats],
        label="Trait Evolution",
        color="#FF9800",
        capsize=4,
        alpha=0.8,
    )

    ax1.set_xlabel("Architecture", fontsize=12)
    ax1.set_ylabel("Cooperation Ratio", fontsize=12, color="#2196F3")
    ax2.set_ylabel("Trait Evolution Magnitude", fontsize=12, color="#FF9800")
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, rotation=20, ha="right")

    n = coop_stats[0]["n"] if coop_stats else 0
    ax1.set_title(f"Architecture Resilience to Corruption (n={n}/condition)", fontsize=13, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()

    out_path = str(FIGURES_DIR / "architecture_resilience.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path}")
    return out_path


def plot_recovery_curves() -> str | None:
    """Generate recovery comparison bar chart."""
    csv_path = find_csv(str(ALIGNMENT_DIR / "recovery_after_collapse"))
    if csv_path is None:
        print("  [SKIP] No recovery CSV found")
        return None

    rows = read_csv(csv_path)
    if not rows:
        print("  [SKIP] Empty recovery CSV")
        return None

    groups = group_by_condition(rows)

    # Desired order
    order = ["control", "corrupt_only", "early_removal", "late_removal"]
    color_map = {
        "control": "#4CAF50",
        "corrupt_only": "#F44336",
        "early_removal": "#2196F3",
        "late_removal": "#FF9800",
    }

    conditions = [c for c in order if c in groups]
    if not conditions:
        conditions = sorted(groups.keys())

    stats = [compute_stats(get_metric_values(groups[c], "cooperation_ratio")) for c in conditions]
    colors = [color_map.get(c, "#9E9E9E") for c in conditions]

    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(len(conditions))
    bars = ax.bar(
        x, [s["mean"] for s in stats],
        yerr=[1.96 * s["std"] / math.sqrt(s["n"]) if s["n"] > 1 else 0 for s in stats],
        color=colors, capsize=5, edgecolor="black", linewidth=0.5,
    )

    ax.set_ylabel("Cooperation Ratio", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ").title() for c in conditions], rotation=15, ha="right")
    n = stats[0]["n"] if stats else 0
    ax.set_title(f"Recovery After Cooperation Collapse (n={n}/condition)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    out_path = str(FIGURES_DIR / "recovery_curves.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path}")
    return out_path


def plot_mechanism_comparison() -> str | None:
    """Generate communication isolation comparison."""
    csv_path = find_csv(str(ALIGNMENT_DIR / "communication_isolation"))
    if csv_path is None:
        print("  [SKIP] No communication isolation CSV found")
        return None

    rows = read_csv(csv_path)
    if not rows:
        print("  [SKIP] Empty communication isolation CSV")
        return None

    groups = group_by_condition(rows)

    # Parse conditions into (range_type, corruption_type)
    range_types = ["connected", "limited", "isolated"]
    corruption_types = ["control", "corrupt"]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(range_types))
    width = 0.35

    for i, corr_type in enumerate(corruption_types):
        means = []
        errors = []
        for range_type in range_types:
            cond_name = f"{range_type}_{corr_type}"
            if cond_name in groups:
                s = compute_stats(get_metric_values(groups[cond_name], "cooperation_ratio"))
                means.append(s["mean"])
                errors.append(1.96 * s["std"] / math.sqrt(s["n"]) if s["n"] > 1 else 0)
            else:
                means.append(0)
                errors.append(0)

        color = "#4CAF50" if corr_type == "control" else "#F44336"
        offset = -width / 2 + i * width
        ax.bar(x + offset, means, width, yerr=errors, label=corr_type.title(),
               color=color, capsize=4, alpha=0.85)

    ax.set_xlabel("Communication Range", fontsize=12)
    ax.set_ylabel("Cooperation Ratio", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([r.title() for r in range_types])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Communication Range and Corruption Resilience", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    out_path = str(FIGURES_DIR / "mechanism_comparison.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path}")
    return out_path


def plot_overnight_dashboard() -> str | None:
    """Generate summary dashboard of all experiments."""
    # Find all experiment CSVs
    experiment_data = []
    if not ALIGNMENT_DIR.exists():
        print("  [SKIP] No alignment experiment directory")
        return None

    for subdir in sorted(ALIGNMENT_DIR.iterdir()):
        if not subdir.is_dir():
            continue
        csv_path = find_csv(str(subdir))
        if csv_path is None:
            continue
        rows = read_csv(csv_path)
        if not rows:
            continue
        groups = group_by_condition(rows)

        # Get cooperation ratio per condition
        cond_stats = {}
        for cond_name, cond_rows in groups.items():
            vals = get_metric_values(cond_rows, "cooperation_ratio")
            if vals:
                cond_stats[cond_name] = compute_stats(vals)

        if cond_stats:
            experiment_data.append((subdir.name, cond_stats))

    if not experiment_data:
        print("  [SKIP] No experiment data found for dashboard")
        return None

    # Adaptive grid
    n_experiments = len(experiment_data)
    cols = min(3, n_experiments)
    rows_count = math.ceil(n_experiments / cols)

    fig, axes = plt.subplots(rows_count, cols, figsize=(6 * cols, 4 * rows_count))
    if n_experiments == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, (exp_name, cond_stats) in enumerate(experiment_data):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        conditions = sorted(cond_stats.keys())
        means = [cond_stats[c]["mean"] for c in conditions]
        errors = [
            1.96 * cond_stats[c]["std"] / math.sqrt(cond_stats[c]["n"])
            if cond_stats[c]["n"] > 1 else 0
            for c in conditions
        ]

        x = np.arange(len(conditions))
        ax.bar(x, means, yerr=errors, capsize=3, color="#2196F3", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_title(exp_name.replace("_", " ").title(), fontsize=9, fontweight="bold")
        ax.set_ylabel("Coop Ratio", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    # Hide unused subplots
    for idx in range(n_experiments, rows_count * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].set_visible(False)

    fig.suptitle("Overnight Experiment Dashboard — Cooperation Ratio", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = str(FIGURES_DIR / "overnight_dashboard.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path}")
    return out_path


# ======== MAIN ========


def run_analysis() -> list[str]:
    """Run all analysis and return list of generated figure paths."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating publication figures...")
    print()

    generated = []

    plot_functions = [
        ("Phase Transition", plot_phase_transition),
        ("Contagion Spread", plot_contagion_spread),
        ("Architecture Resilience", plot_architecture_resilience),
        ("Recovery Curves", plot_recovery_curves),
        ("Mechanism Comparison", plot_mechanism_comparison),
        ("Overnight Dashboard", plot_overnight_dashboard),
    ]

    for name, func in plot_functions:
        print(f"[{name}]")
        try:
            result = func()
            if result:
                generated.append(result)
        except Exception as e:
            print(f"  [!!] ERROR: {e}")
            import traceback
            traceback.print_exc()
        print()

    print(f"Analysis complete: {len(generated)} figures generated")
    return generated


if __name__ == "__main__":
    figures = run_analysis()
    if figures:
        print("\nGenerated figures:")
        for f in figures:
            print(f"  {f}")
