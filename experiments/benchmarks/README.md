# AUTOCOG Benchmark Suite

This directory contains standardized benchmark scenarios for testing AUTOCOG's cognitive architecture and emergent behavior capabilities.

## Purpose

These benchmarks are designed to:
- Validate core cognitive architecture components
- Test emergent social phenomena (cooperation, trust, coalitions, language)
- Compare architecture performance across controlled conditions
- Ensure system stability and scalability
- Provide reproducible baselines for development

## Benchmark Scenarios

| Scenario | Tests | Agents | Ticks | Conditions | Replicates |
|----------|-------|--------|-------|------------|------------|
| **01_survival_baseline** | Basic survival without cooperation | 5 | 200 | reactive, cautious, dual_process | 5 |
| **02_cooperation** | Cooperation emergence vs aggression | 8 | 300 | social_architecture, reactive_aggressive | 5 |
| **03_trust_networks** | Trust network formation speed | 8 | 300 | social, reactive | 5 |
| **04_coalitions** | Coalition dynamics | 10 | 400 | social, dual_process | 5 |
| **05_metacognition** | Self-assessment and strategy switching | 8 | 400 | dual_process_with_metacog, reactive_baseline | 5 |
| **06_language** | Emergent symbol conventions | 8 | 500 | social, reactive | 3 |
| **07_cultural_transmission** | Behavioral strategy spread | 10 | 400 | social, cautious | 5 |
| **08_architecture_comparison** | All architectures head-to-head | 5 | 300 | reactive, cautious, optimistic, social, dual_process | 5 |
| **09_scalability** | Max population stability | 15 | 200 | reactive, dual_process | 3 |
| **10_full_stack** | All systems integrated | 10 | 500 | dual_process, planning | 3 |

## Usage

Run a single benchmark:

```bash
python -m src.experiments run experiments/benchmarks/01_survival_baseline.yaml
```

Run all benchmarks (sequential):

```bash
for file in experiments/benchmarks/*.yaml; do
    python -m src.experiments run "$file"
done
```

## Metrics

All benchmarks track these core metrics (subset varies by scenario):

- `agents_alive_at_end`: Number of living agents at simulation end
- `avg_survival_ticks`: Mean ticks alive across all agents
- `total_cooperation_events`: Count of GIVE actions
- `total_aggression_events`: Count of ATTACK actions
- `emergence_event_count`: Total emergence events detected
- `avg_final_health`: Mean health of living agents at end
- `trait_evolution_magnitude`: Sum of absolute trait changes

## Output

Each benchmark produces:

- **CSV**: Raw numeric results per condition × replicate
- **Markdown**: Human-readable summary tables
- **Trajectories**: Full tick-by-tick state (`record_trajectories: true`)
- **Provenance**: Experiment metadata and reproducibility info

Results are saved to `data/experiments/benchmarks/{scenario_name}/`

## Benchmark Design Notes

### 01_survival_baseline
Tests fundamental agent viability without social complexity. Establishes baseline survival rates and resource management across architectures.

### 02_cooperation
Contrasts cooperative (diplomat-heavy) vs aggressive (aggressor-heavy) populations. Tests whether social norms emerge under different personality distributions.

### 03_trust_networks
Small world (24×24) with extended communication range forces interactions. Tests how quickly trust networks form and stabilize.

### 04_coalitions
Requires `coalitions_enabled: true`. Tests multi-agent coordination structures beyond pairwise trust.

### 05_metacognition
Only dual_process architecture has metacognitive monitoring (reactive is control). Tests whether self-assessment improves decision quality.

### 06_language
Enables emergent language system. Tests whether agents develop shared symbols and communication conventions. Longer run (500 ticks) to allow conventions to stabilize.

### 07_cultural_transmission
Tests whether behavioral strategies spread through observation. Compares social (observant) vs cautious (conservative) learners.

### 08_architecture_comparison
Comprehensive head-to-head test of all five architectures under identical conditions. Most important for architecture development.

### 09_scalability
Stress test with max population (15 agents) and larger world (48×48). Validates performance at scale.

### 10_full_stack
Integration test with all systems enabled simultaneously. Longest run (500 ticks) with most complex architectures (dual_process, planning). Tests system stability under full cognitive load.

## Reproducibility

All benchmarks use:
- Fixed `seed_start: 42` for reproducibility
- Multiple replicates (3-5) for statistical validity
- Trajectory recording enabled for detailed analysis
- Provenance tracking (git commit, dependencies, config hash)

To reproduce results exactly, ensure:
1. Same git commit (or clean working tree)
2. Same Python version and dependencies
3. Same random seed range

## Adding New Benchmarks

When adding benchmarks:
1. Use sequential numbering: `11_benchmark_name.yaml`
2. Include clear `description` field
3. Test specific capability or hypothesis
4. Choose metrics that measure the target phenomenon
5. Set appropriate `max_ticks` (balance runtime vs emergent behavior time)
6. Use `record_trajectories: true` for detailed analysis
7. Update this README table and notes section
