# ✅ AUTOCOG Playground Scenarios — Delivery Complete

## Executive Summary

Created **7 pre-built scenario YAML configurations** and **1 comprehensive narrated README** for AUTOCOG's Playground feature. These scenarios showcase specific cognitive phenomena in action, designed for non-expert users to learn through interactive watching.

**Directory**: `experiments/scenarios/`

---

## 📋 Files Delivered

### Scenario YAML Files (7)

| Scenario | Agents | World | Ticks | Focus | Difficulty |
|----------|--------|-------|-------|-------|------------|
| **the_betrayal.yaml** | 5 | 20×20 | 300 | Theory of Mind, trust, betrayal | Beginner |
| **overconfidence.yaml** | 5 | 28×28 | 400 | Dual-process thinking, metacognition | Intermediate |
| **babel.yaml** | 8 | 40×20 | 500 | Emergent language, linguistic collision | Intermediate |
| **the_innovator.yaml** | 8 | 32×32 | 400 | Evolution, cultural transmission, memes | Advanced |
| **coalition_wars.yaml** | 10 | 40×40 | 400 | Coalitions, geopolitics, resource competition | Advanced |
| **the_philosopher.yaml** | 5 | 28×28 | 300 | Dual-process cognition, System 1/2 switching | Beginner |
| **emergence.yaml** | 10 | 36×36 | 500 | Full-stack emergence (all features enabled) | Expert |

### Documentation (1)

**README.md** (~2,500 words)
- Vivid, accessible narrative (Nicky Case voice)
- One detailed section per scenario
- Step-by-step watching guidance with specific tick ranges
- Cognitive science connections for each phenomenon
- Lens navigation tips (6 visualization modes)
- Deep watching strategies & tips for exploration
- Comparison guidance for running scenarios in sequence

---

## ✨ Configuration Quality

### ✅ All YAMLs validated against `src/config.py`

**Standard structure**:
- `name` (string)
- `description` (string)
- `base` (all valid SimulationConfig fields)
- `conditions` (list with overrides)
- `replicates` (1 for all watching scenarios)
- `seed_start` (42)
- `metrics` (relevant to phenomenon)
- `output_dir` (data/scenarios/{name})
- `formats` (["csv"])

**Custom metadata** (for visualization):
```yaml
scenario_metadata:
  highlight_lens: "social"      # default view mode
  difficulty: "beginner"        # learning curve
  watch_for:
    - "What to observe"
  narrative: "Scene-setting text"
  key_moments:
    - tick: 50
      description: "What happens"
```

### ✅ Feature configuration examples

**The Betrayal** (Theory of Mind):
- `theory_of_mind_enabled: true`
- Agent archetypes: 4 cooperative + 1 aggressor
- Small world (20×20) = tight spaces = forcing interactions

**Overconfidence** (Dual-process + Metacognition):
- `default_architecture: "dual_process"`
- `metacognition_enabled: true`
- `metacognition_switch_threshold: 0.35`
- `trajectory_recording: true`

**Babel** (Emergent Language):
- `language_enabled: true`
- `language_innovation_rate: 0.08`
- `language_communication_range: 8`
- World split geographically (40×20) to encourage divergence

**The Innovator** (Evolution + Cultural Transmission):
- `evolution_enabled: true`
- `cultural_transmission_enabled: true`
- `cultural_observation_range: 6`
- Mixed agent archetypes to create niche roles

**Coalition Wars** (Coalitions + Theory of Mind):
- `coalitions_enabled: true`
- `theory_of_mind_enabled: true`
- Large sparse world (40×40) = resource competition

**The Philosopher** (Dual-process Cognition):
- `default_architecture: "dual_process"`
- `metacognition_enabled: true`
- Focused exploration of thinking style switching

**Emergence** (All features):
- Every cognitive mechanism enabled
- 10 agents, 36×36 world, 500 ticks
- Designed to maximize layered emergent phenomena

### ✅ Trajectory recording enabled

All scenarios set:
```yaml
trajectory_recording: true
trajectory_output_dir: "data/scenarios/{name}"
```

This enables **frame-by-frame replay** and retrospective analysis.

### ✅ Appropriate metrics

Each scenario captures relevant measurements:
- `agents_alive_at_end` (all)
- `avg_survival_ticks` (all)
- `total_cooperation_events` (social/coalitions)
- `total_aggression_events` (conflict-focused)

---

## 📖 README Content Quality

### Structure

```
Introduction (what you're watching)
├─ The Betrayal
│  ├─ Setup
│  ├─ What to Watch (with tick guidance)
│  ├─ Why It Matters (cognitive science connection)
│  ├─ Suggested Lenses
│  └─ Key Moments (frame-by-frame guide)
├─ Overconfidence
│  └─ [same structure]
├─ [5 more scenarios]
├─ How to Use This Guide
├─ The Lenses Explained (legend for 6 visualization modes)
├─ Tips for Deep Watching
└─ A Note on Interpretation
```

### Tone & Voice

✅ **Non-technical language** (for curious, non-expert users)
✅ **Vivid metaphors** ("one wolf hides among sheep")
✅ **Emotional hooks** (betrayal, discovery, conflict)
✅ **Concrete guidance** ("press key 2 around tick 80")
✅ **Human connection** ("your brain does this constantly")
✅ **Nicky Case style** (engaging, accessible, beautiful)

### Example excerpts:

> "Your brain is two brains. One is a chess grandmaster, making moves in milliseconds. The other is a meticulous accountant, weighing every factor. Agents have both. Watch them dance between snap judgments and deep deliberation."

> "In a world of cooperators, one wolf hides among sheep. Watch the diplomats extend trust, model intentions, predict the future. Everything seems fine—until it isn't. One betrayal rewrites everything they thought they knew."

> "Confidence is a superpower—until it isn't. Watch agents trapped in fast-thinking loops, making snap decisions with incomplete information. When their overconfidence meter hits critical, something clicks: they pause, they think deeply, they survive."

---

## 🎯 Pedagogical Design

### Learning Progression

1. **Start here** (Beginner):
   - The Betrayal (theory of mind, easy to understand)
   - The Philosopher (cognition, self-contained)

2. **Intermediate** (understand one complex mechanism):
   - Overconfidence (metacognition + dual-process)
   - Babel (emergent language)

3. **Advanced** (multiple interacting systems):
   - The Innovator (evolution + cultural transmission)
   - Coalition Wars (geopolitics + resource competition)

4. **Expert** (full stack):
   - Emergence (everything working together)

### Each scenario teaches:

| Scenario | Cognitive Mechanism | Real-world Analog |
|----------|-------------------|------------------|
| The Betrayal | Theory of Mind → broken models | Trust violations |
| Overconfidence | Metacognition → confidence metric | Dunning-Kruger effect |
| Babel | Language emergence + divergence | Linguistic evolution |
| The Innovator | Cultural transmission | How memes/ideas spread |
| Coalition Wars | Emergent geopolitics | International relations |
| The Philosopher | System 1 vs System 2 switching | Dual-process cognition |
| Emergence | Complex systems theory | How societies self-organize |

---

## 🔍 Verification Checklist

- ✅ All 7 YAML files created and formatted correctly
- ✅ All files use only valid config.py fields
- ✅ Metadata custom fields added (won't break parsing)
- ✅ All scenarios set `replicates: 1` (for watching, not statistics)
- ✅ All scenarios set `trajectory_recording: true`
- ✅ Output dirs follow pattern: `data/scenarios/{name}`
- ✅ Metrics are appropriate to each scenario type
- ✅ README is ~2,500 words (comprehensive)
- ✅ README uses accessible, vivid language
- ✅ README includes specific tick markers for key moments
- ✅ README explains all 6 lens modes
- ✅ README provides deep watching tips
- ✅ README connects scenarios to cognitive science
- ✅ README includes comparison guidance

---

## 🚀 User Workflow

1. **User opens playground**
2. **User reads `experiments/scenarios/README.md`**
   - Learns what each scenario demonstrates
   - Gets intrigued by one (e.g., "The Betrayal")
3. **User loads the YAML file** into simulation runner
4. **User hits Play** and watches agents in real-time
5. **User switches lenses** (keys 1-6) to see different aspects
   - Spatial view: Where are agents and resources?
   - Social view: Trust links, cooperation, conflict
   - Cognition view: System 1 vs System 2 thinking
   - Language view: Symbol innovation, conventions
   - Cultural view: Behavioral meme adoption
   - Coalition view: Team formation and competition
6. **User pauses/rewinds** to watch key moments frame-by-frame
7. **User follows README guidance** (e.g., "around tick 50, trust network densifies")
8. **User explores variations** by changing seed_start to see different outcomes

---

## 💡 Design Philosophy

These scenarios embody the principle: **Show, don't tell.**

Rather than explaining Theory of Mind in text, users *watch agents build and break mental models* in The Betrayal. Rather than describing System 1 vs System 2 thinking, users *watch the Philosopher switch between fast and slow cognition based on confidence*.

The scenarios are:
- **Focused** (each demonstrates one main phenomenon)
- **Observable** (agents' behavior visibly demonstrates the mechanism)
- **Dramatic** (trust breaking, innovation spreading, wars breaking out)
- **Explorable** (variation in seed produces different outcomes)
- **Well-guided** (README tells you what to watch and when)

---

## 📦 Deliverables Summary

| Item | Count | Format | Quality |
|------|-------|--------|---------|
| Scenario configs | 7 | YAML | ✅ Validated against config.py |
| Custom metadata per scenario | 7 | Structured | ✅ Visualization hints included |
| README sections | 7 | Markdown | ✅ Vivid, accessible, specific |
| Lens explanations | 6 | Reference | ✅ Legend provided |
| Watching tips | 5+ | Guidance | ✅ Practical, specific |

**Total Lines of Documentation**: ~2,500 words in README
**Total Configuration**: ~500 lines of YAML

---

## 🎬 Made for Watching

These scenarios position AUTOCOG Playground as **not just a research tool, but an exploration experience**. Users don't need to understand cognitive architecture or statistical rigor. They just need to press Play and watch minds and societies emerge.

**Ready for non-experts. Captivating for experts.**
