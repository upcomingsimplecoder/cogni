# 🎬 AUTOCOG Playground Scenarios — Complete Delivery Summary

**Created**: 7 pre-built scenario YAML configs + 1 comprehensive narrated README
**Location**: `experiments/scenarios/`
**Status**: ✅ Ready for use

---

## 📦 What Was Created

### Scenario Files (7 YAML configs)

Each YAML is a complete, self-contained simulation scenario with:
- ✅ Valid YAML syntax
- ✅ All fields validated against `src/config.py`
- ✅ Feature configuration specific to the cognitive phenomenon
- ✅ `replicates: 1` (for watching, not statistics)
- ✅ `trajectory_recording: true` (for frame-by-frame replay)
- ✅ Custom `scenario_metadata` for visualization hints
- ✅ Appropriate metrics for the phenomenon type

**Scenarios (by difficulty)**:

| # | Name | Difficulty | Agents | World | Focus |
|---|------|-----------|--------|-------|-------|
| 1 | **the_betrayal.yaml** | Beginner | 5 | 20×20 | Theory of Mind, trust |
| 2 | **the_philosopher.yaml** | Beginner | 5 | 28×28 | Dual-process cognition |
| 3 | **overconfidence.yaml** | Intermediate | 5 | 28×28 | Metacognition, confidence |
| 4 | **babel.yaml** | Intermediate | 8 | 40×20 | Emergent language |
| 5 | **the_innovator.yaml** | Advanced | 8 | 32×32 | Cultural transmission |
| 6 | **coalition_wars.yaml** | Advanced | 10 | 40×40 | Coalitions, geopolitics |
| 7 | **emergence.yaml** | Expert | 10 | 36×36 | Full-stack emergence |

### Documentation

**README.md** (~2,500 words):
- Vivid, accessible introduction (Nicky Case voice)
- One detailed section per scenario with:
  - **Setup** (world conditions, agents)
  - **What to Watch** (step-by-step with tick markers)
  - **Why It Matters** (cognitive science connection)
  - **Suggested Lenses** (which visualization modes to use)
  - **Key Moments** (frame-by-frame guide)
- **How to Use This Guide** (user workflow)
- **The Lenses Explained** (legend for 6 visualization modes)
- **Tips for Deep Watching** (exploration strategies)
- **A Note on Interpretation** (what these demonstrate)

**Supporting files**:
- `DELIVERY.md` (detailed technical summary)
- `QUICKSTART.sh` (quick reference)

---

## 🎨 Design Philosophy

### These scenarios are NOT traditional experiments.

Traditional experiments = measure outcomes across conditions.
These scenarios = **watch emergence happen in real-time**.

### Each demonstrates a specific cognitive phenomenon:

| Scenario | Shows | Mechanism |
|----------|-------|-----------|
| **The Betrayal** | How mental models break | Theory of Mind + negative evidence |
| **The Philosopher** | How thinking styles switch | Metacognition detects doubt → triggers System 2 |
| **Overconfidence** | System 1 → System 2 transition | Confidence meter → thinking style choice |
| **Babel** | Language divergence + collision | Independent innovation → incompatible symbols → crisis |
| **The Innovator** | How ideas spread | One agent invents → others observe → meme adoption |
| **Coalition Wars** | Emergent geopolitics | Theory of Mind + scarcity → faction formation → competition |
| **Emergence** | Layered complexity | All mechanisms interacting → maximum emergence |

---

## 📋 Configuration Details

### All scenarios include:

```yaml
name: "Scenario Name"
description: "What it demonstrates"

base:
  world_width: X
  world_height: X
  max_ticks: N
  num_agents: N
  # Feature flags (Theory of Mind, metacognition, language, etc.)
  # tailored to the phenomenon
  trajectory_recording: true
  trajectory_output_dir: "data/scenarios/{name}"

conditions:
  - name: "condition_name"
    overrides:
      # Feature configuration specific to scenario

replicates: 1
seed_start: 42

metrics:
  # Relevant measurements (cooperation, aggression, survival, etc.)

output_dir: "data/scenarios/{name}"
formats: ["csv"]

scenario_metadata:
  highlight_lens: "social"  # which lens to default to
  difficulty: "beginner"     # learning curve
  watch_for:                # what to observe
    - "Specific events..."
  narrative: "Vivid scene-setting text"
  key_moments:              # tick-by-tick guidance
    - tick: 50
      description: "What happens"
```

### Feature configurations tailored to each scenario:

**The Betrayal** (Theory of Mind emergence):
- `default_architecture: "social"`
- `theory_of_mind_enabled: true`
- Agent mix: 4 cooperative + 1 aggressor
- Small world: 20×20 (forces interactions)

**Overconfidence** (Metacognition & dual-process):
- `default_architecture: "dual_process"`
- `metacognition_enabled: true`
- `metacognition_switch_threshold: 0.35`
- 400 ticks (long enough for clear transition)

**Babel** (Emergent language):
- `language_enabled: true`
- `language_innovation_rate: 0.08` (high innovation)
- `language_communication_range: 8`
- World 40×20 (geographic separation for divergence)

**The Innovator** (Cultural transmission):
- `evolution_enabled: true`
- `cultural_transmission_enabled: true`
- `cultural_observation_range: 6`
- Mixed archetypes (creates niche roles)

**Coalition Wars** (Coalitions + Theory of Mind):
- `coalitions_enabled: true`
- `theory_of_mind_enabled: true`
- Large sparse world: 40×40 (resource scarcity)
- 10 agents (enough for faction dynamics)

**The Philosopher** (Dual-process cognition):
- `default_architecture: "dual_process"`
- `metacognition_enabled: true`
- Focused: exploring thinking style switching
- 300 ticks (clear arc)

**Emergence** (All mechanisms):
- Every cognitive feature enabled
- 10 agents, 36×36 world, 500 ticks
- Designed to demonstrate maximum emergence

---

## 📖 README Quality

### Vivid, accessible language:

Examples of tone:

> *"Your brain is two brains. One is a chess grandmaster, making moves in milliseconds. The other is a meticulous accountant, weighing every factor."*

> *"In a world of cooperators, one wolf hides among sheep. Watch the diplomats extend trust, model intentions, predict the future. Everything seems fine—until it isn't."*

> *"Innovation doesn't require genius—just desperate exploration. One agent stumbles onto a better way. Others are watching. Not through instruction, but through imitation and cultural osmosis, a new way of being spreads like a virus."*

### Specific, actionable guidance:

Each scenario includes:

- **Setup** (what's the world like?)
- **What to Watch** (1-6 numbered steps with tick markers)
- **Why It Matters** (connection to human cognition)
- **Suggested Lenses** (which visualization modes to use)
- **Key Moments** (exactly when to look: "Around tick 80-120: The betrayal")

Example from The Betrayal:

> 1. Switch to the **Social Lens** (press `2` in the visualization)
> 2. In the early game (tick 30-80), notice the white trust links forming...
> 3. The diplomats are optimistic...
> 4. Then, around tick 80-120, something breaks...
> 5. Watch how the surviving agents reorganize...

### Educational scaffolding:

- **Beginner scenarios** first (The Betrayal, The Philosopher)
- **Intermediate** next (Overconfidence, Babel, The Innovator)
- **Advanced** (Coalition Wars)
- **Expert** (Emergence)

Each teaches progressively more complex mechanisms.

---

## 🎮 User Experience

### Typical user journey:

1. Opens playground
2. Reads `README.md` → gets intrigued by a scenario
3. Loads the YAML file
4. Hits Play
5. Watches agents interact in real-time
6. Switches lenses (keys 1-6) to see different aspects
7. Pauses/rewinds key moments
8. Follows README guidance ("around tick X, watch for Y")
9. Explores variations (different seed → different outcome)

### No cognitive burden:

- ✅ Non-experts don't need to understand configuration
- ✅ Non-experts don't need to understand metrics
- ✅ Non-experts don't need to understand YAML
- ✅ They just press Play and Watch

---

## ✅ Quality Checklist

### Configuration validation:
- ✅ All fields are valid `SimulationConfig` attributes
- ✅ No typos in field names
- ✅ Data types are correct (bool, float, int, list)
- ✅ All metrics are supported
- ✅ Output directories follow convention

### Scenario design:
- ✅ Each focuses on ONE main phenomenon
- ✅ Feature configuration is tailored to phenomenon
- ✅ World size/agent count appropriate for phenomenon
- ✅ Tick duration long enough to see emergence
- ✅ All use `replicates: 1` (watching, not stats)
- ✅ All use `trajectory_recording: true` (enable replay)

### Documentation quality:
- ✅ ~2,500 words comprehensive
- ✅ Vivid, accessible language
- ✅ Specific tick markers for key moments
- ✅ Explains all 6 lens modes
- ✅ Provides deep watching tips
- ✅ Connects to cognitive science concepts
- ✅ No jargon (or immediately explained)

---

## 🚀 Ready to Use

All files are in `experiments/scenarios/`:

```
experiments/scenarios/
├── README.md                    ← START HERE (main guide)
├── DELIVERY.md                  (technical summary)
├── QUICKSTART.sh               (quick reference)
├── the_betrayal.yaml
├── the_philosopher.yaml
├── overconfidence.yaml
├── babel.yaml
├── the_innovator.yaml
├── coalition_wars.yaml
└── emergence.yaml
```

**Next steps**:
1. Open the playground
2. Load a YAML file (start with the_betrayal.yaml or the_philosopher.yaml)
3. Hit Play
4. Follow the README for guidance

---

## 💡 Philosophy

These scenarios transform AUTOCOG Playground from:

**Before**: "Run experiments to collect data"
**After**: "Watch minds and societies emerge in real-time"

They're designed to be:
- **Mesmerizing** (agents make decisions, betray, create language)
- **Educational** (each teaches a cognitive phenomenon)
- **Explorable** (vary seed, watch different outcomes)
- **Accessible** (no technical knowledge required)

Welcome to the AUTOCOG Playground. Watch emergence happen.
