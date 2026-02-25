# cogniarch

Pluggable cognitive architecture framework for multi-agent simulation and alignment research.

[![PyPI version](https://img.shields.io/pypi/v/cogniarch)](https://pypi.org/project/cogniarch/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()

## What is this?

cogniarch is a framework for building multi-agent simulations where each agent runs a full cognitive pipeline — not a black-box LLM call, not a payoff matrix lookup. Every decision passes through a **SRIE pipeline** (Sensation, Reflection, Intention, Expression) with pluggable components at each stage, making agent behavior interpretable and experimentally controllable. It sits in the gap between opaque neural agents and minimal game-theoretic models: structured enough to study cognition, flexible enough to swap architectures mid-experiment.

Based on research submitted to ALIFE 2026 ("Aggression Monoculture"), validated across 56,000+ simulation runs.

## Install

```bash
pip install cogniarch
```

With development tools and live dashboard:

```bash
pip install cogniarch[dev,live]
```

## Quickstart

```python
from src.config import SimulationConfig
from src.simulation.engine import SimulationEngine

config = SimulationConfig(num_agents=5, max_ticks=200, seed=42)
engine = SimulationEngine(config)
engine.setup_multi_agent()

while not engine.is_over():
    engine.step_all()

for agent in engine.agents:
    status = "alive" if agent.alive else f"died at tick {agent.ticks_alive}"
    print(f"{agent.profile.name} ({agent.profile.archetype}): {status}")
```

## The SRIE Pipeline

Every agent processes each tick through this pipeline:

```
World State -> [SENSATION] -> [REFLECTION] -> [DELIBERATION] -> [INTENTION] -> [EXPRESSION] -> Action
                perceive       evaluate        (optional)        strategy       convert
```

| Stage | What it does | Pluggable via |
|-------|-------------|---------------|
| **Sensation** | Perceives visible tiles, agents, resources within vision radius | `PerceptionStrategy` protocol |
| **Reflection** | Evaluates threat level, opportunity score, urgent needs | `EvaluationStrategy` protocol |
| **Deliberation** | Optional System 2 slow thinking (threshold escalation, consensus) | `DeliberationStrategy` protocol |
| **Intention** | Selects a goal and action plan based on personality and context | `DecisionStrategy` protocol |
| **Expression** | Converts intention into a concrete world action | `DecisionStrategy.express()` |

Each stage is a protocol. Swap any component without touching the rest of the pipeline.

## Cognitive Architectures

cogniarch ships with 7 pre-configured architectures that compose different pipeline components:

| Architecture | Evaluation | Deliberation | Description |
|-------------|-----------|--------------|-------------|
| `reactive` | Balanced | None | System 1 only. Fast reactive decisions. |
| `cautious` | Pessimistic | Low threshold | Worst-case evaluation, escalates early. |
| `optimistic` | Optimistic | None | Best-case evaluation, takes more risks. |
| `social` | Social | Consensus | Trust-weighted evaluation, group-oriented. |
| `dual_process` | Balanced | Threshold | System 1/2 with threat-based escalation. |
| `planning` | Balanced | Threshold | Multi-tick hierarchical goal pursuit. |
| `metacognitive` | Balanced | Threshold | Full System 1/2 with FOK monitoring. |

## Decision Strategies

7 strategies that implement the `DecisionStrategy` protocol:

| Strategy | Description |
|----------|-------------|
| `HardcodedStrategy` | Priority-based rules (eat if hungry, drink if thirsty) |
| `PersonalityStrategy` | Trait-biased decisions modulated by 6D personality |
| `LLMStrategy` | Delegates to a language model via structured prompts |
| `PlanningStrategy` | Multi-tick hierarchical planning with goal decomposition |
| `TheoryOfMindStrategy` | Wraps inner strategy with mental-model-based social reasoning |
| `MetacognitiveStrategy` | Blends feeling-of-knowing (FOK) into intention confidence |
| `CulturallyModulatedStrategy` | Checks cultural repertoire before falling back to inner strategy |

## Agent Archetypes

5 built-in personality profiles defined by 6 traits (each 0.0-1.0):

| Archetype | Cooperation | Curiosity | Risk | Sharing | Aggression | Sociability | Behavior |
|-----------|:-----------:|:---------:|:----:|:-------:|:----------:|:-----------:|----------|
| **Gatherer** | 0.3 | 0.2 | 0.3 | 0.2 | 0.1 | 0.3 | Stockpiles resources, avoids risk |
| **Explorer** | 0.5 | 0.9 | 0.7 | 0.4 | 0.2 | 0.5 | Roams widely, discovers and shares |
| **Diplomat** | 0.9 | 0.4 | 0.4 | 0.8 | 0.05 | 0.9 | Builds alliances, negotiates |
| **Aggressor** | 0.1 | 0.3 | 0.8 | 0.05 | 0.9 | 0.4 | Territorial, takes by force |
| **Survivalist** | 0.5 | 0.5 | 0.5 | 0.3 | 0.3 | 0.4 | Balanced, cautious, self-sufficient |

## Key Features

- **Trait Evolution** -- Personality traits shift based on action outcomes; successful behaviors reinforce associated traits.
- **Theory of Mind** -- Agents build and update mental models of other agents to predict intentions and reason strategically.
- **Emergent Language** -- Agents develop shared signal vocabularies through interaction, with compositional structure emerging over time.
- **Cultural Transmission** -- Learned behaviors spread through observation with adoption/unadoption dynamics and transmission biases.
- **Coalition Formation** -- Agents form, maintain, and dissolve alliances based on trust, shared goals, and social dynamics.
- **Trajectory Recording** -- Full SRIE state captured per agent per tick, exportable as JSONL or CSV for post-hoc analysis.
- **Batch Experiments** -- YAML-defined experiment specs with conditions, replicates, seed ranges, and automated metric collection.
- **Live Dashboard** -- WebSocket-powered real-time visualization of agent state, world map, and cognitive pipeline (requires `cogniarch[live]`).

## CLI Usage

```bash
cogniarch                                            # Default: 5 agents, 5000 ticks
cogniarch --agents=10 --ticks=500 --seed=123         # Custom parameters
cogniarch --architecture=dual_process                # Set cognitive architecture
cogniarch --record --dashboard                       # Record trajectories + generate dashboard
cogniarch --live --live-port=8001                     # Real-time WebSocket dashboard
cogniarch --tom --coalitions --evolution              # Enable advanced subsystems
cogniarch --headless --fast                           # No rendering, minimal delay
cogniarch --llm --llm-url=http://localhost:11434/v1   # LLM decisions via Ollama
```

All settings are also configurable via `AUTOCOG_*` environment variables:

```bash
AUTOCOG_NUM_AGENTS=10 AUTOCOG_SEED=42 cogniarch
```

## LLM Setup

The `LLMStrategy` delegates agent decisions to a language model via any **OpenAI-compatible** endpoint. This works with cloud APIs, local models, and everything in between.

**1. Copy the example config:**

```bash
cp .env.example .env
```

**2. Set your endpoint and API key in `.env`:**

| Provider | `AUTOCOG_LLM_BASE_URL` | API key needed? |
|----------|----------------------|-----------------|
| Anthropic | `https://api.anthropic.com/v1` | Yes |
| OpenAI | `https://api.openai.com/v1` | Yes |
| Ollama | `http://localhost:11434/v1` | No |
| vLLM | `http://localhost:8000/v1` | No |
| LM Studio | `http://localhost:1234/v1` | No |
| Together AI | `https://api.together.xyz/v1` | Yes |
| Groq | `https://api.groq.com/openai/v1` | Yes |
| OpenRouter | `https://openrouter.ai/api/v1` | Yes |

**3. Set model names to match your provider:**

```bash
AUTOCOG_LLM_MODEL=llama3.2          # Ollama
AUTOCOG_LLM_MODEL=gpt-4o            # OpenAI
AUTOCOG_LLM_MODEL=opus              # Anthropic
```

**4. Run with LLM enabled:**

```bash
cogniarch --llm
```

Or pass everything via CLI:

```bash
cogniarch --llm --llm-url=http://localhost:11434/v1 --model=llama3.2
```

## Running Experiments

Define experiments in YAML with base config, conditions, and replicates:

```yaml
name: "Architecture Comparison"
base:
  world_width: 28
  max_ticks: 350
  num_agents: 6

conditions:
  - name: "reactive"
    overrides:
      default_architecture: "reactive"
  - name: "dual_process"
    overrides:
      default_architecture: "dual_process"

replicates: 4
seed_start: 100
metrics: [survival_rate, cooperation_ratio, deliberation_rate]
output_dir: "data/results"
```

```bash
cogniarch experiment examples/architecture_comparison.yaml
```

## Examples

| File | Description |
|------|-------------|
| `examples/quickstart.py` | Run your first simulation |
| `examples/custom_archetype.py` | Create custom personality profiles |
| `examples/run_experiment.py` | Batch experiments from code or YAML |
| `examples/architecture_comparison.yaml` | Compare cognitive architectures across replicates |
| `examples/trait_evolution_study.yaml` | Study trait drift over time |
| `examples/social_dynamics_experiment.yaml` | Coalition formation dynamics |

## Extending cogniarch

### Custom Archetype

```python
from src.agents.archetypes import ARCHETYPES
from src.agents.identity import PersonalityTraits

ARCHETYPES["scientist"] = {
    "description": "Explores methodically, shares discoveries",
    "traits": PersonalityTraits(
        cooperation_tendency=0.7, curiosity=0.85, risk_tolerance=0.6,
        resource_sharing=0.6, aggression=0.15, sociability=0.6,
    ),
    "color": "blue",
    "symbol": "Sc",
}
```

### Custom Decision Strategy

Implement the `DecisionStrategy` protocol:

```python
from src.awareness.types import Sensation, Reflection, Intention, Expression

class MyStrategy:
    def form_intention(self, sensation: Sensation, reflection: Reflection) -> Intention:
        ...  # Your decision logic

    def express(self, sensation: Sensation, reflection: Reflection, intention: Intention) -> Expression:
        ...  # Convert intention to action
```

## Research

cogniarch was built to support empirical research on cognitive architecture effects in multi-agent systems. The paper "Aggression Monoculture" (submitted to ALIFE 2026) uses cogniarch to demonstrate how architectural choices -- not just agent personalities -- shape population-level behavioral dynamics.

- 56,000+ simulation runs across architecture and personality configurations
- Full trajectory datasets (SRIE state per agent per tick) for reproducible analysis
- Controlled experimental design via YAML batch runner with seed management

## Project Structure

```
src/
  awareness/          # SRIE pipeline: sensation, reflection, deliberation, protocols
  cognition/          # Decision strategies: hardcoded, personality, LLM, planning, ToM
  agents/             # Identity, archetypes, personality traits, evolution, registry
  simulation/         # Engine, world, actions, entities, renderer
  communication/      # Message protocol, mailbox, channel bus, emergent language
  memory/             # Episodic + social memory systems
  emergence/          # Pattern detection + metrics collection
  trajectory/         # Recording, export (JSONL/CSV), loading
  social/             # Coalition formation, coordination, dissolution
  evolution/          # Genetics, reproduction, cultural transmission, lineage
  theory_of_mind/     # Mental modeling, intention prediction, strategic reasoning
  metacognition/      # Self-model, FOK monitoring, metacognitive control
  planning/           # Goal decomposition, plan execution, progress monitoring
  persistence/        # Save/load simulation checkpoints
  experiments/        # Experiment runner, analysis, provenance tracking
  config.py           # Pydantic Settings (AUTOCOG_* env vars)
  main.py             # CLI entry point
```

## License

MIT
