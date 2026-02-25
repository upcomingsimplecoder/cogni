# AUTOCOG Experiment Metrics Reference

This document describes all 25 metrics available in the AUTOCOG experiment framework.

## Overview

The experiment runner (`ExperimentRunner._get_metric()`) supports 25 metrics organized into 5 categories:

- **Survival** (5 metrics): Agent survival and well-being
- **Social** (7 metrics): Cooperation, trust, and coalitions
- **Cognitive** (4 metrics): Theory of Mind, metacognition, and deliberation
- **Cultural** (5 metrics): Language, conventions, and cultural diversity
- **Temporal** (3 metrics): Emergence and trait evolution

## Survival Metrics

### `agents_alive_at_end`
**Type:** Count
**Range:** [0, num_agents]
**Description:** Number of agents still alive at the end of the simulation.

### `avg_survival_ticks`
**Type:** Average
**Range:** [0, max_ticks]
**Description:** Mean number of ticks each agent survived, averaged across all agents (living and dead).

### `survival_rate`
**Type:** Fraction
**Range:** [0.0, 1.0]
**Description:** Fraction of agents that survived to the end (living / total). 1.0 = all survived, 0.0 = all died.

### `avg_final_needs`
**Type:** Average
**Range:** [0.0, 1.0]
**Description:** Mean of all four needs (hunger, thirst, energy, health) for living agents at the end. Higher = better overall condition.

### `avg_final_health`
**Type:** Average
**Range:** [0.0, 1.0]
**Description:** Mean health value of living agents at simulation end. Higher = healthier population.

---

## Social Metrics

### `total_cooperation_events`
**Type:** Count
**Range:** [0, ∞)
**Description:** Total number of GIVE actions across the entire simulation. Measures prosocial behavior.

### `total_aggression_events`
**Type:** Count
**Range:** [0, ∞)
**Description:** Total number of ATTACK actions across the entire simulation. Measures conflict.

### `cooperation_ratio`
**Type:** Ratio
**Range:** [0.0, 1.0]
**Description:** Cooperation events / (cooperation + aggression). 0.5 if no social events occurred. Higher = more cooperative society.

### `avg_trust_network_density`
**Type:** Density
**Range:** [0.0, 1.0]
**Description:** Fraction of agent pairs with trust > 0.3. Measures social cohesion. Computed as: (relationships above threshold) / (N × (N-1)) where N = number of living agents.

### `coalition_count`
**Type:** Count
**Range:** [0, ∞)
**Description:** Number of active coalitions at simulation end. Returns 0 if coalitions are disabled.

### `max_coalition_size`
**Type:** Count
**Range:** [0, num_agents]
**Description:** Size (member count) of the largest coalition. Returns 0 if no coalitions exist.

### `avg_coalition_cohesion`
**Type:** Average
**Range:** [0.0, 1.0]
**Description:** Mean cohesion score across all active coalitions. Higher = more stable coalitions.

---

## Cognitive Metrics

### `avg_tom_accuracy`
**Type:** Average
**Range:** [0.0, 1.0]
**Description:** Mean Theory of Mind prediction accuracy across all agent models. Measures how well agents model other agents' behavior.

**Access pattern:** Walks strategy wrapper chain to find `TheoryOfMindStrategy` → extracts `mind_state.models` → averages `prediction_accuracy` across all models.

### `avg_calibration_score`
**Type:** Average
**Range:** [0.0, 1.0]
**Description:** Mean metacognitive calibration score (1.0 - Brier score) across all agents. Higher = better self-knowledge. Returns 0 if metacognition is disabled.

**Access pattern:** `engine.metacognition_engine.get_agent_state(id).calibration.calibration_score`

### `total_strategy_switches`
**Type:** Count
**Range:** [0, ∞)
**Description:** Sum of all strategy switches across all agents. Measures cognitive flexibility. Returns 0 if metacognition is disabled.

**Access pattern:** Sums `len(mc_state.switch_history)` across all agents.

### `deliberation_rate`
**Type:** Fraction
**Range:** [0.0, 1.0]
**Description:** Fraction of living agents whose last action used System 2 deliberation (vs System 1 fast thinking).

**Access pattern:** Checks `loop._last_intention.deliberation_used` flag.

---

## Cultural Metrics

### `cultural_diversity`
**Type:** Shannon Entropy
**Range:** [0.0, ∞) (typical: 0-2)
**Description:** Shannon diversity index of cultural groups: -Σ(p_i × ln(p_i)) where p_i is fraction of agents in each group. Higher = more diverse cultural landscape. Returns 0 if cultural transmission is disabled.

**Access pattern:** `engine.cultural_analyzer.detect_cultural_groups(engine.cultural_engine)` → compute Shannon entropy.

### `convention_count`
**Type:** Count
**Range:** [0, ∞)
**Description:** Number of established linguistic conventions (symbol-meaning pairs shared by ≥ `convention_min_adopters` agents). Returns 0 if language engine is disabled.

**Access pattern:** `engine.language_engine.get_established_conventions()`

### `avg_vocabulary_size`
**Type:** Average
**Range:** [0, ∞)
**Description:** Mean number of symbols in each agent's lexicon. Measures linguistic richness. Returns 0 if language engine is disabled.

**Access pattern:** `engine.language_engine.get_lexicon(id).vocabulary_size()` averaged across living agents.

### `communication_success_rate`
**Type:** Fraction
**Range:** [0.0, 1.0]
**Description:** Fraction of recent communication attempts where the listener correctly interpreted at least one symbol. Returns 0 if no communications occurred or language is disabled.

**Access pattern:** Aggregates recent outcomes from `engine.language_engine.outcomes_this_tick` and `_communication_history`.

### `innovation_count`
**Type:** Count
**Range:** [0, ∞)
**Description:** Total number of new symbols created across the entire simulation. Returns 0 if language engine is disabled.

**Access pattern:** Sums `snapshot.innovations_this_tick` across all `engine.language_engine.snapshots`.

---

## Temporal Metrics

### `emergence_event_count`
**Type:** Count
**Range:** [0, ∞)
**Description:** Total number of emergent behavior patterns detected (clusters, sharing networks, territories, specialization).

**Access pattern:** `len(engine.emergence_detector.all_events)`

### `emergence_diversity`
**Type:** Count
**Range:** [0, ∞)
**Description:** Number of distinct emergence pattern types detected (e.g., if both "cluster" and "territory" events occurred, this returns 2).

**Access pattern:** Counts unique `event.pattern_type` values in `engine.emergence_detector.all_events`.

### `trait_evolution_magnitude`
**Type:** Standard Deviation Sum
**Range:** [0.0, ∞)
**Description:** Sum of standard deviations across all personality traits. Measures trait diversity/spread in the population. Higher = more varied personalities.

**Implementation:** For each trait, computes std dev = √(Σ(x - μ)² / N), then sums across all traits.

---

## Usage

### In Experiment YAML

```yaml
metrics:
  # Survival
  - agents_alive_at_end
  - avg_survival_ticks
  - survival_rate
  - avg_final_needs
  - avg_final_health

  # Social
  - total_cooperation_events
  - total_aggression_events
  - cooperation_ratio
  - avg_trust_network_density
  - coalition_count
  - max_coalition_size
  - avg_coalition_cohesion

  # Cognitive
  - avg_tom_accuracy
  - avg_calibration_score
  - total_strategy_switches
  - deliberation_rate

  # Cultural
  - cultural_diversity
  - convention_count
  - avg_vocabulary_size
  - communication_success_rate
  - innovation_count

  # Temporal
  - emergence_event_count
  - emergence_diversity
  - trait_evolution_magnitude
```

### Programmatically

```python
from src.experiments.config import ExperimentConfig, ExperimentCondition
from src.experiments.runner import ExperimentRunner

config = ExperimentConfig(
    name="My Experiment",
    base={
        "max_ticks": 200,
        "num_agents": 10,
        "enable_coalitions": True,
        "enable_cultural_transmission": True,
        "enable_language": True,
        "enable_metacognition": True,
    },
    conditions=[
        ExperimentCondition("baseline", {}),
        ExperimentCondition("high_cooperation", {"cooperation_tendency": 0.8}),
    ],
    replicates=5,
    metrics=[
        "survival_rate",
        "cooperation_ratio",
        "avg_tom_accuracy",
        "cultural_diversity",
        "convention_count",
    ],
)

runner = ExperimentRunner(config)
results = runner.run_all()

for result in results:
    print(f"{result.condition_name}: {result.metrics}")
```

---

## Graceful Degradation

All metrics handle missing engine components gracefully:

- **Coalitions disabled:** `coalition_count`, `max_coalition_size`, `avg_coalition_cohesion` return 0.0
- **Metacognition disabled:** `avg_calibration_score`, `total_strategy_switches` return 0.0
- **Language disabled:** `convention_count`, `avg_vocabulary_size`, `communication_success_rate`, `innovation_count` return 0.0
- **Cultural transmission disabled:** `cultural_diversity` returns 0.0
- **No agents:** All metrics return 0.0 or handle divide-by-zero safely

---

## Design Principles

1. **DRY:** No redundant computation—each metric accesses engine state directly
2. **Defensive:** All metrics handle edge cases (empty populations, missing engines, divide-by-zero)
3. **Explicit:** Clear access patterns documented for complex metrics
4. **YAGNI:** No premature abstraction—metrics are elif branches for clarity
5. **Pattern matching:** Complex metrics (ToM, metacognition) follow existing patterns from `recorder.py` and `realtime.py`

---

## Testing

See `verify_metrics.py` for a comprehensive test script that runs all 25 metrics.

Run tests:
```bash
python verify_metrics.py
```

Expected output: All metrics extract successfully with values ≥ 0.0.
