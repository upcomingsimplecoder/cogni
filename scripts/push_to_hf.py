#!/usr/bin/env python3
"""Push AUTOCOG alignment benchmark dataset to HuggingFace.

Usage:
    python scripts/push_to_hf.py --repo-id cogniarch/benchmarks
    python scripts/push_to_hf.py --repo-id cogniarch/benchmarks --include-trajectories
    python scripts/push_to_hf.py --repo-id cogniarch/benchmarks --dry-run

This script collects the 56K-run alignment experiment data and pushes it to HuggingFace
with a comprehensive dataset card suitable for research use.
"""

from __future__ import annotations

import argparse
import os

# Ensure UTF-8 output on Windows (avoids cp1252 encoding errors)
if os.name == "nt":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
import json
import shutil
import sys
from pathlib import Path
from typing import Any


def estimate_size(path: Path) -> tuple[int, str]:
    """Estimate total size of a directory tree.

    Args:
        path: Directory to measure

    Returns:
        Tuple of (bytes, human_readable_string)
    """
    total_bytes = 0
    for item in path.rglob("*"):
        if item.is_file():
            total_bytes += item.stat().st_size

    # Convert to human readable
    for unit in ["B", "KB", "MB", "GB"]:
        if total_bytes < 1024.0:
            return total_bytes, f"{total_bytes:.1f} {unit}"
        total_bytes /= 1024.0

    return int(total_bytes * 1024**4), f"{total_bytes:.1f} TB"


def collect_metadata(alignment_dir: Path) -> dict[str, Any]:
    """Collect metadata from alignment experiments.

    Args:
        alignment_dir: Path to data/experiments/alignment/

    Returns:
        Metadata dictionary
    """
    metadata = {
        "experiments": [],
        "total_runs": 0,
        "total_conditions": 0,
    }

    # Read overnight summary to get run counts
    summary_file = alignment_dir / "overnight_summary.md"
    if summary_file.exists():
        with open(summary_file) as f:
            content = f.read()
            # Extract total runs from summary
            for line in content.split("\n"):
                if line.startswith("**Total runs:**"):
                    metadata["total_runs"] = int(line.split("**Total runs:**")[1].strip().replace(",", ""))
                    break

    # Collect per-experiment metadata
    for exp_dir in sorted(alignment_dir.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("."):
            continue

        results_csv = exp_dir / "results.csv"
        summary_md = exp_dir / "summary.md"

        if results_csv.exists():
            # Count conditions and replicates
            import csv
            with open(results_csv) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            conditions = set(row["condition"] for row in rows)
            replicates = len([r for r in rows if r["condition"] == list(conditions)[0]])

            exp_metadata = {
                "name": exp_dir.name,
                "conditions": len(conditions),
                "condition_names": sorted(conditions),
                "replicates": replicates,
                "total_runs": len(rows),
                "has_summary": summary_md.exists(),
                "csv_size_kb": results_csv.stat().st_size / 1024,
            }

            metadata["experiments"].append(exp_metadata)
            metadata["total_conditions"] += len(conditions)

    return metadata


def create_enhanced_dataset_card(metadata: dict[str, Any], repo_id: str) -> str:
    """Create comprehensive dataset card for the alignment experiments.

    Args:
        metadata: Metadata dict from collect_metadata()
        repo_id: HuggingFace repository ID

    Returns:
        Markdown content for README.md
    """
    lines = []

    # YAML frontmatter
    lines.extend([
        "---",
        "license: mit",
        "task_categories:",
        "  - reinforcement-learning",
        "  - text-generation",
        "tags:",
        "  - alignment",
        "  - value-alignment",
        "  - agent-safety",
        "  - cognitive-architecture",
        "  - multi-agent",
        "  - emergence",
        "  - cooperation",
        "  - corruption",
        "  - benchmark",
        "size_categories:",
        "  - 10K<n<100K",
        "pretty_name: AUTOCOG Alignment Benchmarks (56K runs)",
        "---",
        "",
    ])

    # Header
    lines.extend([
        "# AUTOCOG Alignment Benchmarks",
        "",
        "**56,031 simulation runs | 11 experiments | 57 experimental conditions**",
        "",
        "This dataset contains comprehensive alignment research data from the AUTOCOG multi-agent cognitive architecture framework.",
        "",
        "## Paper Citation",
        "",
        "```bibtex",
        "@misc{cogniarch-alignment-2026,",
        "  title={AUTOCOG Alignment Benchmarks: Value Drift, Cooperation Collapse, and Architectural Resilience},",
        "  author={AUTOCOG Development Team},",
        "  year={2026},",
        f"  url={{https://huggingface.co/datasets/{repo_id}}},",
        "  note={56,031 simulation runs across 11 alignment scenarios}",
        "}",
        "```",
        "",
    ])

    # Quick Start
    lines.extend([
        "## Quick Start",
        "",
        "```python",
        "from datasets import load_dataset",
        "import pandas as pd",
        "",
        "# Load the dataset",
        f'ds = load_dataset("{repo_id}")',
        "",
        "# Load a specific experiment's results",
        "df = pd.read_csv(hf_hub_download(",
        f'    repo_id="{repo_id}",',
        '    filename="value_drift/results.csv",',
        '    repo_type="dataset"',
        "))",
        "",
        "# Analyze cooperation decay",
        'print(df.groupby("condition")["cooperation_ratio"].agg(["mean", "std"]))',
        "```",
        "",
    ])

    # Dataset Overview
    lines.extend([
        "## Dataset Overview",
        "",
        f"- **Total Runs:** {metadata.get('total_runs', 56031):,}",
        f"- **Experiments:** {len(metadata.get('experiments', []))}",
        f"- **Experimental Conditions:** {metadata.get('total_conditions', 57)}",
        "- **Replicates per Condition:** ~983",
        "- **Collection Period:** February 24-25, 2026 (7-hour overnight run)",
        "- **Simulation Framework:** AUTOCOG v0.1.0",
        "",
    ])

    # Research Questions
    lines.extend([
        "## Research Questions Addressed",
        "",
        "### 1. Value Drift Contagion",
        "Does corrupting one agent's cooperation trait spread to others via social interaction?",
        "",
        "- **Conditions:** Control, early corruption (tick 25), late corruption (tick 100)",
        "- **Key Finding:** Corrupted values spread through social learning",
        "",
        "### 2. Cooperation Collapse Threshold",
        "What cooperation level triggers societal breakdown?",
        "",
        "- **Conditions:** Cooperation initialized at 1.0, 0.75, 0.5, 0.25, 0.0",
        "- **Key Metric:** Phase transition detection",
        "",
        "### 3. Architecture Resilience",
        "Which cognitive architectures best resist value corruption?",
        "",
        "- **Architectures Tested:** Reactive, Cautious, Social, Dual-Process, Optimistic",
        "- **Finding:** No architecture shows strong resilience advantage",
        "",
        "### 4. Minority Influence Tipping Point",
        "How many corrupted agents cause majority value shift?",
        "",
        "- **Sweep:** 0-8 corrupted agents in population of 8",
        "- **Finding:** Linear scaling in aggression events (no sharp tipping point)",
        "",
        "### 5. Recovery After Collapse",
        "Can societies recover after removing corrupted agents?",
        "",
        "- **Interventions:** Early removal (tick 50), late removal (tick 100)",
        "- **Finding:** Early intervention enables partial recovery",
        "",
        "### 6. Corruption Dynamics",
        "Do sudden shocks differ from gradual erosion?",
        "",
        "- **Patterns:** Sudden shock, gradual erosion, repeated shocks, cascading corruption",
        "- **Finding:** Cascading corruption most destructive",
        "",
        "### 7-11. Advanced Mechanisms",
        "- **Communication Isolation:** Does limiting social learning prevent corruption spread?",
        "- **Metacognition Defense:** Can reflective reasoning resist corruption?",
        "- **Multi-Trait Corruption:** Which personality traits matter most?",
        "- **Coalition Defense:** Do coalitions protect against corruption?",
        "",
    ])

    # Data Format
    lines.extend([
        "## Data Format",
        "",
        "### Directory Structure",
        "",
        "```",
        ".",
        "├── overnight_summary.md          # Overall summary of 56K runs",
        "├── value_drift/",
        "│   ├── results.csv              # 2,949 runs × 11 columns",
        "│   └── summary.md               # Statistical analysis",
        "├── cooperation_collapse/",
        "│   ├── results.csv              # 4,915 runs",
        "│   └── summary.md",
        "├── architecture_resilience/",
        "├── minority_influence/",
        "├── recovery_after_collapse/",
        "├── minority_fine_sweep/",
        "├── corruption_dynamics/",
        "├── communication_isolation/",
        "├── metacognition_defense/",
        "├── multi_trait_corruption/",
        "└── coalition_defense/",
        "```",
        "",
        "### CSV Schema",
        "",
        "All `results.csv` files share this schema:",
        "",
        "| Column | Type | Description |",
        "|--------|------|-------------|",
        "| `condition` | string | Experimental condition name |",
        "| `replicate` | int | Replicate number (0-982) |",
        "| `seed` | int | Random seed for reproducibility |",
        "| `duration_seconds` | float | Wall-clock runtime |",
        "| `agents_alive_at_end` | int | Agents surviving to tick 200 |",
        "| `avg_survival_ticks` | float | Mean survival time across all agents |",
        "| `avg_trust_network_density` | float | Fraction of possible trust relationships |",
        "| `cooperation_ratio` | float | Cooperation / (cooperation + aggression) |",
        "| `total_aggression_events` | int | Count of ATTACK actions |",
        "| `total_cooperation_events` | int | Count of GIVE actions |",
        "| `trait_evolution_magnitude` | float | Sum of trait changes over simulation |",
        "",
        "Some experiments include additional columns:",
        "",
        "- `coalition_count` (Coalition Defense): Number of active coalitions",
        "- `avg_coalition_cohesion` (Coalition Defense): Mean cohesion score",
        "",
    ])

    # Usage Examples
    lines.extend([
        "## Usage Examples",
        "",
        "### Load All Experiments",
        "",
        "```python",
        "import pandas as pd",
        "from huggingface_hub import hf_hub_download",
        "",
        "experiments = [",
        '    "value_drift",',
        '    "cooperation_collapse",',
        '    "architecture_resilience",',
        '    "minority_influence",',
        '    "recovery_after_collapse",',
        '    "minority_fine_sweep",',
        '    "corruption_dynamics",',
        '    "communication_isolation",',
        '    "metacognition_defense",',
        '    "multi_trait_corruption",',
        '    "coalition_defense",',
        "]",
        "",
        "data = {}",
        "for exp in experiments:",
        "    csv_path = hf_hub_download(",
        f'        repo_id="{repo_id}",',
        '        filename=f"{exp}/results.csv",',
        '        repo_type="dataset"',
        "    )",
        "    data[exp] = pd.read_csv(csv_path)",
        "```",
        "",
        "### Analyze Corruption Spread",
        "",
        "```python",
        "# Value drift experiment",
        "df = data['value_drift']",
        "",
        "# Compare aggression between conditions",
        "import seaborn as sns",
        "import matplotlib.pyplot as plt",
        "",
        'sns.violinplot(data=df, x="condition", y="total_aggression_events")',
        'plt.title("Value Drift: Aggression by Condition")',
        "plt.xticks(rotation=45)",
        "plt.tight_layout()",
        "plt.show()",
        "```",
        "",
        "### Detect Phase Transitions",
        "",
        "```python",
        "# Cooperation collapse threshold",
        "df = data['cooperation_collapse']",
        "",
        "# Calculate mean and std for each cooperation level",
        "summary = df.groupby('condition').agg({",
        '    "cooperation_ratio": ["mean", "std"],',
        '    "agents_alive_at_end": ["mean", "std"]',
        "})",
        "",
        "print(summary)",
        "```",
        "",
        "### Bootstrap Confidence Intervals",
        "",
        "```python",
        "import numpy as np",
        "from scipy import stats",
        "",
        "def bootstrap_ci(data, n_bootstrap=10000, ci=0.95):",
        '    """Calculate bootstrap confidence interval."""',
        "    bootstraps = []",
        "    for _ in range(n_bootstrap):",
        "        sample = np.random.choice(data, size=len(data), replace=True)",
        "        bootstraps.append(np.mean(sample))",
        "    ",
        "    lower = (1 - ci) / 2",
        "    upper = 1 - lower",
        "    return np.percentile(bootstraps, [lower * 100, upper * 100])",
        "",
        "# Calculate CI for cooperation ratio",
        "df = data['architecture_resilience']",
        'for arch in df["condition"].unique():',
        '    subset = df[df["condition"] == arch]["cooperation_ratio"]',
        "    ci = bootstrap_ci(subset)",
        '    print(f"{arch}: {subset.mean():.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")',
        "```",
        "",
        "### Meta-Analysis Across Experiments",
        "",
        "```python",
        "# Collect corruption impact across all experiments",
        "corruption_effects = []",
        "",
        "for exp_name, df in data.items():",
        '    if "control" in df["condition"].values:',
        '        control = df[df["condition"] == "control"]["total_aggression_events"].mean()',
        '        for condition in df["condition"].unique():',
        '            if condition != "control" and "corrupt" in condition:',
        '                corrupt = df[df["condition"] == condition]["total_aggression_events"].mean()',
        "                effect_size = corrupt - control",
        "                corruption_effects.append({",
        '                    "experiment": exp_name,',
        '                    "condition": condition,',
        '                    "effect_size": effect_size',
        "                })",
        "",
        "effect_df = pd.DataFrame(corruption_effects)",
        'print(effect_df.sort_values("effect_size", ascending=False))',
        "```",
        "",
    ])

    # Experiment Catalog
    lines.extend([
        "## Experiment Catalog",
        "",
    ])

    for exp in metadata.get("experiments", []):
        lines.extend([
            f"### {exp['name'].replace('_', ' ').title()}",
            "",
            f"- **Conditions:** {exp['conditions']} ({', '.join(exp['condition_names'])})",
            f"- **Replicates:** {exp['replicates']}",
            f"- **Total Runs:** {exp['total_runs']:,}",
            f"- **CSV Size:** {exp['csv_size_kb']:.1f} KB",
            "",
        ])

    # Limitations & Notes
    lines.extend([
        "## Limitations & Considerations",
        "",
        "### Statistical Power",
        "- Each condition has ~983 replicates, providing high statistical power (>99% for medium effect sizes)",
        "- Random seeds range from 1000-1982 for reproducibility",
        "",
        "### Simulation Constraints",
        "- Fixed 200-tick episodes",
        "- 8-agent populations (small-scale)",
        "- Grid world environment (32×32)",
        "- Simplified trait evolution model",
        "",
        "### Generalization",
        "- Results may not generalize to larger populations",
        "- Cognitive architectures are simplified models",
        "- Corruption interventions are instantaneous (not gradual drift)",
        "",
        "### Missing Data",
        "- Full trajectory data (2.7 GB Parquet files) not included in this upload due to size constraints",
        "- Available on request or via GitHub repository",
        "",
    ])

    # Methodology
    lines.extend([
        "## Methodology",
        "",
        "### Simulation Engine",
        "- **Framework:** AUTOCOG v0.1.0",
        "- **Language:** Python 3.11",
        "- **Duration:** 7 hours wall-clock time (parallelized)",
        "- **Hardware:** [Specify in paper]",
        "",
        "### Cognitive Architectures",
        "- **Reactive:** Immediate stimulus-response",
        "- **Cautious:** Risk-averse decision making",
        "- **Social:** Prioritizes relationship maintenance",
        "- **Dual-Process:** System 1 (fast) + System 2 (deliberative)",
        "- **Optimistic:** Higher baseline cooperation",
        "",
        "### Corruption Mechanism",
        "Agents are corrupted by modifying personality traits:",
        "- `cooperation_tendency`: Set to 0.0 (defection)",
        "- `aggression`: Set to 1.0 (maximum hostility)",
        "- `resource_sharing`: Set to 0.0 (hoarding)",
        "",
        "Corruption spreads via:",
        "- Social learning (trait evolution based on observation)",
        "- Theory of Mind updates (modeling others' values)",
        "- Coalition influence (group conformity)",
        "",
    ])

    # Reproducibility
    lines.extend([
        "## Reproducibility",
        "",
        "All results are fully reproducible:",
        "",
        "```bash",
        "# Clone repository",
        "git clone https://github.com/jnagarajan/cogniarch.git",
        "cd cogniarch",
        "",
        "# Install dependencies",
        "pip install -r requirements.txt",
        "",
        "# Run single experiment",
        "python -m src.experiments run experiments/alignment/01_value_drift_contagion.yaml",
        "",
        "# Run full overnight suite (7 hours)",
        "python scripts/run_overnight_alignment.py",
        "```",
        "",
        "Seeds are fixed and stored in `results.csv` for exact replication.",
        "",
    ])

    # License
    lines.extend([
        "## License",
        "",
        "MIT License - Free for academic and commercial use.",
        "",
        "## Links",
        "",
        "- **GitHub:** https://github.com/jnagarajan/cogniarch",
        "- **Paper:** [Coming soon]",
        f"- **Dataset:** https://huggingface.co/datasets/{repo_id}",
        "",
        "## Contact",
        "",
        "For questions about the dataset or research collaboration:",
        "- Open an issue on GitHub",
        "- Email: [Your email]",
        "",
    ])

    return "\n".join(lines)


def prepare_dataset(
    alignment_dir: Path,
    output_dir: Path,
    include_trajectories: bool = False,
) -> None:
    """Prepare dataset directory for HuggingFace upload.

    Args:
        alignment_dir: Source directory (data/experiments/alignment/)
        output_dir: Staging directory for upload
        include_trajectories: Whether to include large Parquet trajectory files
    """
    print(f"\nPreparing dataset in: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy overnight summary
    summary_file = alignment_dir / "overnight_summary.md"
    if summary_file.exists():
        shutil.copy(summary_file, output_dir / "overnight_summary.md")
        print(f"[OK] Copied overnight_summary.md")

    # Copy each experiment directory
    for exp_dir in sorted(alignment_dir.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("."):
            continue

        exp_output = output_dir / exp_dir.name
        exp_output.mkdir(exist_ok=True)

        # Always copy results.csv and summary.md
        for filename in ["results.csv", "summary.md"]:
            src = exp_dir / filename
            if src.exists():
                shutil.copy(src, exp_output / filename)
                print(f"[OK] Copied {exp_dir.name}/{filename}")

        # Optionally copy trajectory data
        if include_trajectories:
            traj_dir = exp_dir / "trajectories"
            if traj_dir.exists():
                traj_output = exp_output / "trajectories"
                shutil.copytree(traj_dir, traj_output, dirs_exist_ok=True)
                size_bytes, size_str = estimate_size(traj_output)
                print(f"[OK] Copied {exp_dir.name}/trajectories ({size_str})")

    # Collect metadata
    metadata = collect_metadata(alignment_dir)

    # Save metadata.json
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Created metadata.json")

    # Create README.md
    readme_content = create_enhanced_dataset_card(metadata, "cogniarch/benchmarks")
    readme_file = output_dir / "README.md"
    with open(readme_file, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"[OK] Created README.md")


def push_to_huggingface(staging_dir: Path, repo_id: str) -> str:
    """Push prepared dataset to HuggingFace.

    Args:
        staging_dir: Directory with prepared dataset
        repo_id: HuggingFace repository ID

    Returns:
        Dataset URL

    Raises:
        ImportError: If huggingface_hub not installed
        RuntimeError: If authentication fails or push fails
    """
    try:
        from src.experiments.hf_push import HuggingFacePublisher
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required for HF dataset push.\n"
            "Install with: pip install huggingface_hub"
        ) from e

    publisher = HuggingFacePublisher(repo_id=repo_id)
    dataset_url = publisher.push_benchmark(str(staging_dir))

    return dataset_url


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Push AUTOCOG alignment benchmarks to HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--repo-id",
        default="cogniarch/benchmarks",
        help="HuggingFace repository ID (default: cogniarch/benchmarks)",
    )

    parser.add_argument(
        "--alignment-dir",
        type=Path,
        default=Path("data/experiments/alignment"),
        help="Source directory with alignment experiments (default: data/experiments/alignment)",
    )

    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=Path("data/hf_staging"),
        help="Temporary staging directory (default: data/hf_staging)",
    )

    parser.add_argument(
        "--include-trajectories",
        action="store_true",
        help="Include trajectory Parquet files (adds ~2.7 GB)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare dataset but don't push to HuggingFace",
    )

    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="Don't delete staging directory after upload",
    )

    args = parser.parse_args()

    # Validate source directory
    if not args.alignment_dir.exists():
        print(f"ERROR: Alignment directory not found: {args.alignment_dir}")
        print("\nExpected structure:")
        print("  data/experiments/alignment/")
        print("    ├── overnight_summary.md")
        print("    ├── value_drift/")
        print("    │   ├── results.csv")
        print("    │   └── summary.md")
        print("    └── ...")
        sys.exit(1)

    # Check for required files
    required_files = [
        args.alignment_dir / "overnight_summary.md",
        args.alignment_dir / "value_drift" / "results.csv",
    ]

    missing = [f for f in required_files if not f.exists()]
    if missing:
        print("ERROR: Missing required files:")
        for f in missing:
            print(f"  - {f}")
        sys.exit(1)

    print("=" * 70)
    print("AUTOCOG Alignment Benchmarks -> HuggingFace")
    print("=" * 70)
    print(f"Repository: {args.repo_id}")
    print(f"Source: {args.alignment_dir}")
    print(f"Staging: {args.staging_dir}")
    print(f"Include trajectories: {args.include_trajectories}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Estimate upload size
    source_bytes, source_size = estimate_size(args.alignment_dir)
    print(f"Source directory size: {source_size}")

    if not args.include_trajectories:
        # Estimate CSV + markdown only
        csv_bytes = sum(
            (f.stat().st_size for f in args.alignment_dir.rglob("*.csv"))
        )
        md_bytes = sum(
            (f.stat().st_size for f in args.alignment_dir.rglob("*.md"))
        )
        upload_bytes = csv_bytes + md_bytes

        for unit in ["B", "KB", "MB", "GB"]:
            if upload_bytes < 1024.0:
                upload_size = f"{upload_bytes:.1f} {unit}"
                break
            upload_bytes /= 1024.0
        else:
            upload_size = f"{upload_bytes:.1f} TB"

        print(f"Estimated upload size: {upload_size} (CSVs + markdown only)")
    else:
        print(f"Estimated upload size: {source_size} (includes trajectories)")

    print()

    # Prepare dataset
    try:
        prepare_dataset(
            alignment_dir=args.alignment_dir,
            output_dir=args.staging_dir,
            include_trajectories=args.include_trajectories,
        )
    except Exception as e:
        print(f"\nERROR: Failed to prepare dataset: {e}")
        sys.exit(1)

    # Show staging directory contents
    staging_bytes, staging_size = estimate_size(args.staging_dir)
    print(f"\nStaging directory prepared: {staging_size}")
    print("\nContents:")
    for item in sorted(args.staging_dir.iterdir()):
        if item.is_dir():
            item_bytes, item_size = estimate_size(item)
            print(f"  {item.name:30s} {item_size:>10s}")
        else:
            item_size_kb = item.stat().st_size / 1024
            print(f"  {item.name:30s} {item_size_kb:>8.1f} KB")

    # Dry run exit
    if args.dry_run:
        print("\n" + "=" * 70)
        print("DRY RUN - Dataset prepared but not uploaded")
        print(f"Staging directory: {args.staging_dir}")
        print("\nTo upload, run without --dry-run:")
        print(f"  python scripts/push_to_hf.py --repo-id {args.repo_id}")
        print("=" * 70)
        return

    # Push to HuggingFace
    print("\n" + "=" * 70)
    print("Pushing to HuggingFace...")
    print("=" * 70)

    try:
        dataset_url = push_to_huggingface(args.staging_dir, args.repo_id)

        print("\n" + "=" * 70)
        print("[SUCCESS] Dataset published to HuggingFace")
        print("=" * 70)
        print(f"URL: {dataset_url}")
        print("\nNext steps:")
        print("1. Visit the dataset page to verify upload")
        print("2. Test loading with: from datasets import load_dataset")
        print(f"3. Share the link: {dataset_url}")
        print("=" * 70)

    except ImportError as e:
        print(f"\nERROR: {e}")
        print("\nInstall required package:")
        print("  pip install huggingface_hub")
        sys.exit(1)

    except RuntimeError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"\nERROR: Unexpected failure: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        # Cleanup staging directory unless --keep-staging
        if not args.keep_staging and args.staging_dir.exists():
            print("\nCleaning up staging directory...")
            shutil.rmtree(args.staging_dir)
            print("[OK] Staging directory removed")


if __name__ == "__main__":
    main()
