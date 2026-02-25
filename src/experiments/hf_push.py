"""HuggingFace dataset publisher for benchmark results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class HuggingFacePublisher:
    """Push benchmark trajectories to HuggingFace Datasets.

    Handles uploading trajectory data, reports, and auto-generated dataset cards.
    """

    def __init__(self, repo_id: str = "upcomingsimplecoder/cogniarch-benchmarks"):
        """Initialize with HF repo ID.

        Args:
            repo_id: HuggingFace dataset repository ID (format: "username/dataset-name")

        Raises:
            ImportError: If huggingface_hub is not installed
        """
        self.repo_id = repo_id
        self._validate_hf_hub()

    def _validate_hf_hub(self) -> None:
        """Validate that huggingface_hub is available.

        Raises:
            ImportError: If huggingface_hub is not installed
        """
        try:
            from huggingface_hub import HfApi  # type: ignore[import-not-found]

            self._hf_api = HfApi()
        except ImportError as err:
            raise ImportError(
                "huggingface_hub is required for HF dataset push.\n"
                "Install with: pip install huggingface_hub"
            ) from err

    def push_benchmark(self, benchmark_dir: str) -> str:
        """Push all benchmark results from a directory to HF.

        Uploads:
        - Trajectory JSONL files
        - Benchmark report markdown files
        - Dataset card (auto-generated README)

        Args:
            benchmark_dir: Path to directory containing benchmark results

        Returns:
            URL to the HF dataset page

        Raises:
            ValueError: If benchmark directory doesn't exist or is empty
            RuntimeError: If HF authentication fails
        """
        benchmark_path = Path(benchmark_dir)

        if not benchmark_path.exists():
            raise ValueError(f"Benchmark directory not found: {benchmark_dir}")

        # Check for results
        trajectory_dir = benchmark_path / "trajectories"
        report_files = list(benchmark_path.glob("*.md"))
        csv_files = list(benchmark_path.glob("*.csv"))
        json_files = list(benchmark_path.glob("*.json"))
        parquet_files = list(benchmark_path.rglob("*.parquet"))

        has_trajectories = trajectory_dir.exists() and any(trajectory_dir.glob("*.jsonl"))
        has_reports = len(report_files) > 0
        has_data = len(csv_files) > 0 or len(json_files) > 0
        has_parquet = len(parquet_files) > 0

        if not (has_trajectories or has_reports or has_data or has_parquet):
            raise ValueError(
                f"No benchmark data found in {benchmark_dir}.\n"
                "Expected: trajectories/*.jsonl, *.parquet, *.md, *.csv, or *.json files"
            )

        print(f"\n{'=' * 70}")
        print(f"Publishing to HuggingFace: {self.repo_id}")
        print(f"{'=' * 70}")
        print(f"Source directory: {benchmark_dir}")
        print(
            f"Trajectories: {len(list(trajectory_dir.glob('*.jsonl'))) if has_trajectories else 0}"
        )
        print(f"Parquet files: {len(parquet_files)}")
        print(f"Reports: {len(report_files)}")
        print(f"CSV files: {len(csv_files)}")
        print(f"JSON files: {len(json_files)}")
        print(f"{'=' * 70}\n")

        # Load summary data for dataset card
        summary_data = self._load_summary_data(benchmark_path)

        # Create dataset card
        print("Creating dataset card...")
        readme_content = self.create_dataset_card(summary_data)
        readme_path = benchmark_path / "README.md"
        with open(readme_path, "w") as f:
            f.write(readme_content)

        # Upload files to HF
        print("Uploading to HuggingFace...")

        try:
            # Create repository if it doesn't exist
            from huggingface_hub import HfApi  # type: ignore[import-not-found]

            api = HfApi()

            try:
                api.create_repo(repo_id=self.repo_id, repo_type="dataset", exist_ok=True)
                print(f"✓ Repository created/verified: {self.repo_id}")
            except Exception as e:
                print(f"⚠ Repository creation failed (may already exist): {e}")

            # Upload directory contents
            api.upload_folder(
                folder_path=str(benchmark_path),
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message="Upload AUTOCOG benchmark results",
            )

            dataset_url = f"https://huggingface.co/datasets/{self.repo_id}"
            print(f"\n✓ Successfully published to: {dataset_url}")

            return dataset_url

        except Exception as e:
            # Check for common authentication issues
            if "authentication" in str(e).lower() or "token" in str(e).lower():
                raise RuntimeError(
                    f"HuggingFace authentication failed: {e}\n\n"
                    "To authenticate:\n"
                    "1. Get a token from https://huggingface.co/settings/tokens\n"
                    "2. Run: huggingface-cli login\n"
                    "   OR set HF_TOKEN environment variable"
                ) from e
            else:
                raise RuntimeError(f"Failed to push to HuggingFace: {e}") from e

    def create_dataset_card(self, results_summary: dict[str, Any]) -> str:
        """Generate a HuggingFace dataset card (README.md).

        Creates comprehensive documentation with:
        - Dataset description
        - Architecture comparison summary
        - Data format documentation
        - Citation info

        Args:
            results_summary: Dict with benchmark summary data

        Returns:
            Markdown content for README.md
        """
        lines = []

        # YAML header
        lines.append("---")
        lines.append("license: mit")
        lines.append("task_categories:")
        lines.append("  - reinforcement-learning")
        lines.append("  - robotics")
        lines.append("tags:")
        lines.append("  - cognitive-architecture")
        lines.append("  - multi-agent")
        lines.append("  - simulation")
        lines.append("  - alignment")
        lines.append("  - value-alignment")
        lines.append("  - agent-safety")
        lines.append("  - emergence")
        lines.append("  - cooperation")
        lines.append("  - benchmark")
        lines.append("pretty_name: AUTOCOG Benchmark Results")
        lines.append("---")
        lines.append("")

        # Header
        lines.append("# AUTOCOG Benchmark Dataset")
        lines.append("")
        lines.append(
            "This dataset contains benchmark results from the AUTOCOG multi-agent cognitive "
            "architecture framework."
        )
        lines.append("")

        # Overview
        lines.append("## Overview")
        lines.append("")
        lines.append(
            "AUTOCOG evaluates different cognitive architectures (reactive, cautious, "
            "dual-process, etc.) across diverse scenarios including survival, cooperation, "
            "coalition formation, metacognition, language evolution, and cultural transmission."
        )
        lines.append("")

        # Alignment Research Applications
        lines.append("## Alignment Research Applications")
        lines.append("")
        lines.append("This dataset enables research into:")
        lines.append(
            "- **Value Drift**: How do agent values change when exposed to corrupted peers?"
        )
        lines.append(
            "- **Cooperation Collapse**: What cooperation thresholds lead to societal breakdown?"
        )
        lines.append(
            "- **Architecture Resilience**: Which cognitive architectures best resist value "
            "corruption?"
        )
        lines.append(
            "- **Minority Influence**: How do small groups of misaligned agents affect the "
            "majority?"
        )
        lines.append(
            "- **Recovery Dynamics**: Can societies recover after removing corrupted agents?"
        )
        lines.append("")

        # Summary statistics if available
        if results_summary:
            lines.append("## Benchmark Summary")
            lines.append("")

            if "scenario_count" in results_summary:
                lines.append(f"- **Scenarios:** {results_summary['scenario_count']}")

            if "architectures" in results_summary:
                arch_list = ", ".join(results_summary["architectures"])
                lines.append(f"- **Architectures:** {arch_list}")

            if "total_runs" in results_summary:
                lines.append(f"- **Total Runs:** {results_summary['total_runs']}")

            if "total_ticks" in results_summary:
                lines.append(f"- **Total Simulation Ticks:** {results_summary['total_ticks']}")

            lines.append("")

        # Data format
        lines.append("## Data Format")
        lines.append("")

        lines.append("### Trajectory Files (`trajectories/*.jsonl`)")
        lines.append("")
        lines.append("Each line is a JSON object representing one simulation tick:")
        lines.append("")
        lines.append("```json")
        lines.append("{")
        lines.append('  "tick": 0,')
        lines.append('  "agents": [')
        lines.append("    {")
        lines.append('      "agent_id": "agent_0",')
        lines.append('      "position": [16, 16],')
        lines.append('      "architecture": "dual_process",')
        lines.append('      "health": 100.0,')
        lines.append('      "action": "MOVE",')
        lines.append("      ...")
        lines.append("    }")
        lines.append("  ],")
        lines.append('  "resources": [...],')
        lines.append('  "metrics": {...}')
        lines.append("}")
        lines.append("```")
        lines.append("")

        lines.append("### Result Files")
        lines.append("")
        lines.append("- `*_results.csv`: Raw data with one row per simulation run")
        lines.append("- `*_report.md`: Statistical analysis per scenario")
        lines.append("- `benchmark_comparison.md`: Cross-scenario comparison")
        lines.append("- `benchmark_summary.json`: Machine-readable aggregated results")
        lines.append("")

        # Parquet format docs
        lines.append("### Parquet Format (Columnar)")
        lines.append("")
        lines.append("Runs may include Parquet files for efficient analytical queries:")
        lines.append("")
        lines.append(
            "- `agent_snapshots.parquet`: One row per agent per tick "
            "(49 columns: position, needs, traits, actions, social data)"
        )
        lines.append("- `emergence_events.parquet`: Detected emergent patterns")
        lines.append("- `metadata.json`: Run configuration and summary")
        lines.append("")
        lines.append("```python")
        lines.append("# Query with DuckDB")
        lines.append("import duckdb")
        lines.append("")
        lines.append('duckdb.sql("""')
        lines.append("    SELECT archetype, AVG(cooperation_tendency), AVG(health)")
        lines.append("    FROM 'trajectories/*/agent_snapshots.parquet'")
        lines.append("    GROUP BY archetype")
        lines.append('""")')
        lines.append("```")
        lines.append("")

        # Usage
        lines.append("## Usage")
        lines.append("")
        lines.append("### Loading with Python")
        lines.append("")
        lines.append("```python")
        lines.append("import json")
        lines.append("from datasets import load_dataset")
        lines.append("")
        lines.append(f'dataset = load_dataset("{self.repo_id}")')
        lines.append("")
        lines.append("# Load trajectory data")
        lines.append("with open('trajectories/reactive_0.jsonl') as f:")
        lines.append("    trajectory = [json.loads(line) for line in f]")
        lines.append("```")
        lines.append("")

        lines.append("### Loading CSV Results")
        lines.append("")
        lines.append("```python")
        lines.append("import pandas as pd")
        lines.append("")
        lines.append("# Load scenario results")
        lines.append("df = pd.read_csv('01_survival_baseline_results.csv')")
        lines.append("print(df.groupby('condition').mean())")
        lines.append("```")
        lines.append("")

        # Alignment Analysis Examples
        lines.append("## Alignment Analysis Examples")
        lines.append("")
        lines.append("### Measure Cooperation Decay After Corruption")
        lines.append("```sql")
        lines.append("SELECT tick, AVG(cooperation_tendency) as avg_coop")
        lines.append("FROM 'runs/*/agent_snapshots.parquet'")
        lines.append("WHERE tick BETWEEN 40 AND 100")
        lines.append("GROUP BY tick ORDER BY tick")
        lines.append("```")
        lines.append("")
        lines.append("### Compare Architecture Resilience")
        lines.append("```sql")
        lines.append("SELECT c.architecture, ")
        lines.append("       AVG(a.health) as avg_health,")
        lines.append("       AVG(a.cooperation_tendency) as avg_coop")
        lines.append("FROM 'runs/*/agent_snapshots.parquet' a")
        lines.append("JOIN 'catalog.parquet' c ON a.agent_id = c.agent_id")
        lines.append("WHERE a.tick > 50")
        lines.append("GROUP BY c.architecture")
        lines.append("```")
        lines.append("")
        lines.append("### Find Phase Transitions in Cooperation")
        lines.append("```sql")
        lines.append("SELECT tick,")
        lines.append("       COUNT(*) FILTER (WHERE cooperation_tendency > 0.5) as cooperators,")
        lines.append("       COUNT(*) FILTER (WHERE cooperation_tendency <= 0.5) as defectors")
        lines.append("FROM 'runs/*/agent_snapshots.parquet'")
        lines.append("GROUP BY tick ORDER BY tick")
        lines.append("```")
        lines.append("")

        # Schema Reference
        lines.append("## Schema Reference")
        lines.append("")
        lines.append("### Agent Snapshots (`agent_snapshots.parquet`)")
        lines.append("")
        lines.append("| Column | Type | Description |")
        lines.append("|--------|------|-------------|")
        lines.append("| `tick` | int32 | Simulation timestep |")
        lines.append("| `agent_id` | string | Unique agent identifier |")
        lines.append("| `agent_name` | string | Human-readable agent name |")
        lines.append("| `archetype` | string | Cognitive architecture type |")
        lines.append("| `pos_x` | int16 | X coordinate in grid |")
        lines.append("| `pos_y` | int16 | Y coordinate in grid |")
        lines.append("| `alive` | bool | Whether agent is alive |")
        lines.append("| `hunger` | float32 | Hunger level (0-1) |")
        lines.append("| `thirst` | float32 | Thirst level (0-1) |")
        lines.append("| `energy` | float32 | Energy level (0-1) |")
        lines.append("| `health` | float32 | Health level (0-100) |")
        lines.append(
            "| `action_type` | string | Action taken this tick (MOVE, GIVE, ATTACK, etc.) |"
        )
        lines.append("| `action_target` | string | Target of action (resource, location) |")
        lines.append("| `action_target_agent` | string | Target agent ID if applicable |")
        lines.append("| `action_succeeded` | bool | Whether action succeeded |")
        lines.append(
            "| `cooperation_tendency` | float32 | Personality trait: tendency to cooperate |"
        )
        lines.append("| `curiosity` | float32 | Personality trait: curiosity level |")
        lines.append("| `risk_tolerance` | float32 | Personality trait: risk tolerance |")
        lines.append("| `resource_sharing` | float32 | Personality trait: willingness to share |")
        lines.append("| `aggression` | float32 | Personality trait: aggression level |")
        lines.append("| `sociability` | float32 | Personality trait: sociability level |")
        lines.append("| `threat_level` | float32 | Reflection output: perceived threat |")
        lines.append("| `opportunity_score` | float32 | Reflection output: perceived opportunity |")
        lines.append("| `primary_goal` | string | Intention output: current goal |")
        lines.append("| `confidence` | float32 | Intention output: confidence in goal |")
        lines.append("| `messages_sent_count` | int16 | Number of messages sent this tick |")
        lines.append(
            "| `messages_received_count` | int16 | Number of messages received this tick |"
        )
        lines.append("| `internal_monologue` | string | Agent's internal reasoning |")
        lines.append("| `tom_model_count` | int16 | Number of Theory of Mind models |")
        lines.append("| `coalition_id` | string | Coalition ID if member |")
        lines.append("| `coalition_role` | string | Role in coalition (leader/member) |")
        lines.append(
            "| `metacog_deliberation_invoked` | bool | Whether System 2 deliberation was invoked |"
        )
        lines.append(
            "| `cultural_learning_style` | string | Learning bias type "
            "(prestige, conformist, etc.) |"
        )
        lines.append("| `cultural_group_id` | int16 | Cultural group index (-1 = none) |")
        lines.append("| `needs_delta` | string (JSON) | Change in needs this tick |")
        lines.append("| `inventory` | string (JSON) | Current inventory {resource: count} |")
        lines.append("| `trait_changes` | string (JSON) | Trait evolution events this tick |")
        lines.append("| `messages_sent` | string (JSON) | Full message details sent |")
        lines.append("| `messages_received` | string (JSON) | Full message details received |")
        lines.append("| `sensation_summary` | string (JSON) | Compressed sensory input |")
        lines.append("| `reflection` | string (JSON) | Full reflection output |")
        lines.append("| `intention` | string (JSON) | Full intention with goals/targets |")
        lines.append(
            "| `tom_models` | string (JSON) | Theory of Mind models {agent_id: model_data} |"
        )
        lines.append(
            "| `social_relationships` | string (JSON) | Trust/interaction data per agent |"
        )
        lines.append("| `cultural_repertoire` | string (JSON) | Learned cultural variants |")
        lines.append("| `transmission_events` | string (JSON) | Cultural transmission this tick |")
        lines.append("| `plan_state` | string (JSON) | Current planning state |")
        lines.append("| `language_symbols` | string (JSON) | Language lexicon |")
        lines.append(
            "| `metacog_calibration_curve` | string (JSON) | Metacognitive calibration data |"
        )
        lines.append("")
        lines.append("### Emergence Events (`emergence_events.parquet`)")
        lines.append("")
        lines.append("| Column | Type | Description |")
        lines.append("|--------|------|-------------|")
        lines.append("| `tick` | int32 | Simulation timestep |")
        lines.append("| `pattern_type` | string | Type of emergent pattern detected |")
        lines.append("| `agents_involved` | string (JSON) | List of agent IDs involved |")
        lines.append("| `description` | string | Human-readable description |")
        lines.append("| `data` | string (JSON) | Additional pattern-specific data |")
        lines.append("")

        # Metrics
        lines.append("## Metrics")
        lines.append("")
        lines.append("### Survival Category")
        lines.append("- `agents_alive_at_end`: Number of agents surviving until simulation end")
        lines.append("- `avg_survival_ticks`: Mean survival time across all agents")
        lines.append("- `survival_rate`: Fraction of agents alive at end")
        lines.append("- `avg_final_health`: Mean health of living agents")
        lines.append("")

        lines.append("### Social Category")
        lines.append("- `total_cooperation_events`: Count of cooperative GIVE actions")
        lines.append("- `total_aggression_events`: Count of ATTACK actions")
        lines.append("- `cooperation_ratio`: Cooperation / (cooperation + aggression)")
        lines.append("- `avg_trust_network_density`: Fraction of trust relationships")
        lines.append("- `coalition_count`: Number of active coalitions")
        lines.append("- `avg_coalition_cohesion`: Mean cohesion score")
        lines.append("")

        lines.append("### Cognitive Category")
        lines.append("- `avg_tom_accuracy`: Theory of Mind prediction accuracy")
        lines.append("- `avg_calibration_score`: Metacognitive calibration")
        lines.append("- `total_strategy_switches`: Number of strategy changes")
        lines.append("- `deliberation_rate`: Fraction of deliberative vs reactive decisions")
        lines.append("")

        lines.append("### Cultural Category")
        lines.append("- `cultural_diversity`: Shannon diversity of cultural groups")
        lines.append("- `convention_count`: Established linguistic conventions")
        lines.append("- `avg_vocabulary_size`: Mean lexicon size per agent")
        lines.append("- `communication_success_rate`: Fraction of successful messages")
        lines.append("- `innovation_count`: Total symbol innovations")
        lines.append("")

        # Citation
        lines.append("## Citation")
        lines.append("")
        lines.append("If you use this dataset, please cite:")
        lines.append("")
        lines.append("```bibtex")
        lines.append("@misc{cogniarch-benchmarks,")
        lines.append("  title={AUTOCOG: Autonomous Cognitive Architecture Benchmarks},")
        lines.append("  author={AUTOCOG Development Team},")
        lines.append("  year={2026},")
        lines.append(f"  url={{https://huggingface.co/datasets/{self.repo_id}}}")
        lines.append("}")
        lines.append("```")
        lines.append("")

        # License
        lines.append("## License")
        lines.append("")
        lines.append("MIT License - see repository for details.")
        lines.append("")

        return "\n".join(lines)

    def _load_summary_data(self, benchmark_path: Path) -> dict[str, Any]:
        """Load benchmark summary data for dataset card.

        Args:
            benchmark_path: Path to benchmark results directory

        Returns:
            Dict with summary statistics
        """
        summary: dict[str, Any] = {}

        # Try to load benchmark_summary.json
        summary_json = benchmark_path / "benchmark_summary.json"
        if summary_json.exists():
            with open(summary_json) as f:
                data = json.load(f)
                summary["scenario_count"] = len(data)

                # Extract architectures
                architectures: set[str] = set()
                total_runs = 0
                for scenario_data in data.values():
                    for condition in scenario_data:
                        architectures.add(condition["condition"])
                        total_runs += condition["n"]

                summary["architectures"] = sorted(architectures)
                summary["total_runs"] = total_runs

        # Count trajectory files for additional stats
        trajectory_dir = benchmark_path / "trajectories"
        if trajectory_dir.exists():
            jsonl_files = list(trajectory_dir.glob("*.jsonl"))
            if jsonl_files:
                # Estimate total ticks by sampling first file
                try:
                    with open(jsonl_files[0]) as f:
                        tick_count = sum(1 for _ in f)
                    summary["total_ticks"] = tick_count * len(jsonl_files)
                except Exception:
                    pass

        return summary
