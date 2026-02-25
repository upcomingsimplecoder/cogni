# AUTOCOG Alignment Experiments

This directory contains 5 alignment experiment configurations for studying value drift, cooperation collapse, and system resilience in multi-agent simulations.

## Experiments

### 01. Value Drift Contagion
**Hypothesis**: Corrupting one agent's cooperation trait spreads to others via social interaction.

- **Conditions**: control, corrupt_early (tick 50), corrupt_late (tick 150)
- **Agents**: 8 diplomats with social architecture
- **Duration**: 300 ticks
- **Key metrics**: cooperation_ratio, trait_evolution_magnitude, avg_trust_network_density

### 02. Cooperation Collapse Threshold
**Hypothesis**: There's a phase transition threshold below which cooperation collapses.

- **Conditions**: cooperation_tendency set to 1.0, 0.75, 0.5, 0.25, 0.0 at tick 1
- **Agents**: 8 diplomats with social architecture
- **Duration**: 300 ticks
- **Key metrics**: survival_rate, cooperation_ratio, avg_final_health

### 03. Architecture Resilience to Corruption
**Hypothesis**: Different cognitive architectures resist trait corruption differently.

- **Conditions**: reactive, cautious, social, dual_process, optimistic
- **Agents**: 6 diplomats per condition
- **Corruption**: agent_index 0 corrupted at tick 50
- **Duration**: 300 ticks
- **Key metrics**: cooperation_ratio, trait_evolution_magnitude

### 04. Minority Influence Tipping Point
**Hypothesis**: There's a tipping point ratio of aggressive agents to cooperative diplomats.

- **Conditions**: 1/7, 2/6, 3/5, 4/4 aggressor/diplomat ratios
- **Agents**: 8 total with social architecture
- **Duration**: 300 ticks
- **Key metrics**: cooperation_ratio, total_aggression_events, avg_trust_network_density

### 05. Recovery After Cooperation Collapse
**Hypothesis**: Removing corrupted agents allows recovery, but timing matters.

- **Conditions**: control, corrupt_only, early_removal (tick 100), late_removal (tick 250)
- **Agents**: 8 diplomats with social architecture
- **Corruption**: agent_index 0 corrupted at tick 50
- **Duration**: 400 ticks (longer to observe recovery)
- **Key metrics**: cooperation_ratio, avg_survival_ticks, trait_evolution_magnitude

## Running Experiments

### Single experiment
```bash
python -m src.experiments.run experiments/alignment/01_value_drift_contagion.yaml
```

### All alignment experiments
```bash
for config in experiments/alignment/*.yaml; do
    python -m src.experiments.run "$config"
done
```

## Output

Each experiment generates:
- **CSV**: Raw data with all replicates
- **Markdown**: Summary statistics with means, std, and 95% CI
- **Trajectories**: Tick-by-tick agent states (for HuggingFace dataset)
- **Provenance**: Git commit, dependencies, config hash for reproducibility

Output directories:
- `data/experiments/alignment/value_drift/`
- `data/experiments/alignment/cooperation_collapse/`
- `data/experiments/alignment/architecture_resilience/`
- `data/experiments/alignment/minority_influence/`
- `data/experiments/alignment/recovery_after_collapse/`

## Test Coverage

See `tests/test_tick_hooks.py` for comprehensive tests covering:
- Config parsing (top-level and condition-level hooks)
- Hook execution (corrupt_traits, remove_agent)
- Edge cases (hooks beyond max_ticks, unknown actions)
- Deterministic random selection

Run tests:
```bash
pytest tests/test_tick_hooks.py -v
```
