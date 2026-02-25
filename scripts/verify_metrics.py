"""Verification script for new metrics in ExperimentRunner.

This script verifies that all new metrics can be extracted without errors.
"""

from src.experiments.config import ExperimentConfig, ExperimentCondition
from src.experiments.runner import ExperimentRunner


def main():
    """Test all new metrics."""
    # Define all new metrics to test
    new_metrics = [
        # Survival
        "survival_rate",
        "avg_final_needs",
        # Social
        "cooperation_ratio",
        "avg_trust_network_density",
        "coalition_count",
        "max_coalition_size",
        "avg_coalition_cohesion",
        # Cognitive
        "avg_tom_accuracy",
        "avg_calibration_score",
        "total_strategy_switches",
        "deliberation_rate",
        # Cultural
        "cultural_diversity",
        "convention_count",
        "avg_vocabulary_size",
        "communication_success_rate",
        "innovation_count",
        # Temporal
        "emergence_diversity",
        "trait_evolution_magnitude",
    ]

    # Keep existing metrics for comparison
    existing_metrics = [
        "agents_alive_at_end",
        "avg_survival_ticks",
        "total_cooperation_events",
        "total_aggression_events",
        "emergence_event_count",
        "avg_final_health",
    ]

    all_metrics = existing_metrics + new_metrics

    print(f"Testing {len(all_metrics)} metrics ({len(existing_metrics)} existing + {len(new_metrics)} new)")
    print("=" * 70)

    # Create a minimal config with a short simulation
    config = ExperimentConfig(
        name="Metric Verification Test",
        description="Test all metrics can be extracted",
        base={
            "world_width": 16,
            "world_height": 16,
            "max_ticks": 50,  # Longer to allow some behaviors to emerge
            "num_agents": 4,
            "coalitions_enabled": True,
            "cultural_transmission_enabled": True,
            "language_enabled": True,
            "metacognition_enabled": True,
        },
        conditions=[ExperimentCondition("test", {})],
        replicates=1,
        metrics=all_metrics,
        output_dir="data/verify_metrics_test",
    )

    # Run the experiment
    print("Running test simulation...")
    runner = ExperimentRunner(config)
    results = runner.run_all()

    # Check results
    print("\nResults:")
    print("-" * 70)
    result = results[0]

    # Categorize metrics
    categories = {
        "Survival": ["agents_alive_at_end", "avg_survival_ticks", "survival_rate", "avg_final_needs", "avg_final_health"],
        "Social": ["total_cooperation_events", "total_aggression_events", "cooperation_ratio",
                   "avg_trust_network_density", "coalition_count", "max_coalition_size", "avg_coalition_cohesion"],
        "Cognitive": ["avg_tom_accuracy", "avg_calibration_score", "total_strategy_switches", "deliberation_rate"],
        "Cultural": ["cultural_diversity", "convention_count", "avg_vocabulary_size",
                     "communication_success_rate", "innovation_count"],
        "Temporal": ["emergence_event_count", "emergence_diversity", "trait_evolution_magnitude"],
    }

    # Print by category
    for category, metric_names in categories.items():
        print(f"\n{category} Metrics:")
        for metric_name in metric_names:
            if metric_name in result.metrics:
                value = result.metrics[metric_name]
                status = "OK" if value >= 0 else "FAIL"
                print(f"  [{status:>4s}] {metric_name:35s} = {value:.4f}")
            else:
                print(f"  [FAIL] {metric_name:35s} = MISSING")

    # Summary
    print("\n" + "=" * 70)
    print(f"Total metrics: {len(all_metrics)}")
    print(f"Successfully extracted: {len(result.metrics)}")

    if len(result.metrics) == len(all_metrics):
        print("\n[OK] All metrics extracted successfully!")
        return 0
    else:
        missing = set(all_metrics) - set(result.metrics.keys())
        print(f"\n[FAIL] Missing metrics: {missing}")
        return 1


if __name__ == "__main__":
    exit(main())
