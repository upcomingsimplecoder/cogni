# AUTOCOG Playground Scenarios — Summary

## ✅ Delivery Complete

Created 7 pre-built scenario YAML configs + 1 comprehensive narrated README in `experiments/scenarios/`

---

## Files Created

### Scenario YAMLs (7 files)

1. **the_betrayal.yaml** — Theory of Mind & betrayal dynamics
   - 5 agents, 20×20 world, 300 ticks
   - Difficulty: Beginner
   - Lens: Social

2. **overconfidence.yaml** — Dual-process thinking & metacognition
   - 5 agents, 28×28 world, 400 ticks
   - Difficulty: Intermediate
   - Lens: Cognition

3. **babel.yaml** — Emergent language & linguistic divergence
   - 8 agents, 40×20 world, 500 ticks
   - Difficulty: Intermediate
   - Lens: Language

4. **the_innovator.yaml** — Evolution & cultural transmission (memes)
   - 8 agents, 32×32 world, 400 ticks
   - Difficulty: Advanced
   - Lens: Cultural

5. **coalition_wars.yaml** — Coalitions, Theory of Mind, geopolitics
   - 10 agents, 40×40 world, 400 ticks
   - Difficulty: Advanced
   - Lens: Coalition

6. **the_philosopher.yaml** — Dual-process thinking & cognition rhythm
   - 5 agents, 28×28 world, 300 ticks
   - Difficulty: Beginner
   - Lens: Cognition

7. **emergence.yaml** — Full-stack: all features enabled
   - 10 agents, 36×36 world, 500 ticks
   - Difficulty: Expert
   - Lens: Multi

### Documentation (1 file)

**README.md** (~2,500 words)
- Vivid, accessible narration (Nicky Case voice)
- One section per scenario with:
  - **Setup**: Context & initial conditions
  - **What to Watch**: Step-by-step guidance with tick ranges
  - **Why It Matters**: Cognitive science connection
  - **Suggested Lenses**: Which views reveal what
  - **Key Moments**: Frame-by-frame guide to emergence
- Lens legend explaining all 6 visualization modes
- Tips for deep watching & comparing runs
- Interpretation guidance

---

## Configuration Details

### All YAMLs include:

✓ **Standard fields** (name, description, base, conditions, replicates, seed_start, metrics, output_dir, formats)

✓ **Custom metadata** (for visualization hints):
```yaml
scenario_metadata:
  highlight_lens: "social"          # default view
  difficulty: "beginner"            # learning curve
  watch_for:                        # what to observe
    - "Trust links forming..."
  narrative: "Engaging story..."    # scene-setting
  key_moments:                      # tick-by-tick guidance
    - tick: 50
      description: "..."
```

✓ **Feature flags** enabled as appropriate (theory_of_mind, metacognition, language, cultural_transmission, coalitions, evolution)

✓ **Trajectory recording** enabled (`trajectory_recording: true`)

✓ **Tuned parameters** for each scenario type (world size, agent count, tick duration, resource density, etc.)

✓ **Relevant metrics** for each phenomenon (cooperation, aggression, survival, language conventions, etc.)

---

## Narrative Voice

The README uses accessible, vivid language designed for curious non-experts:

- **Avoids jargon** (or explains it immediately)
- **Uses concrete metaphors** ("one wolf hides among sheep")
- **Creates emotional hooks** (betrayal, trust, discovery)
- **Provides specific guidance** (which tick to watch, which lens to use)
- **Connects to human experience** (your brain does this too)

Example tone:
> *"Your brain is two brains. One is a chess grandmaster, making moves in milliseconds. The other is a meticulous accountant, weighing every factor. Agents have both. Watch them dance between snap judgments and deep deliberation."*

---

## How Users Will Interact

1. **Browse README** → Choose a scenario that interests them
2. **Load YAML** → Into AUTOCOG Playground runner
3. **Hit Play** → Watch agents in real-time (with pause/rewind)
4. **Switch Lenses** → Keys 1-6 to see different phenomena
5. **Follow Guidance** → README tells them what to watch and when
6. **Explore Variations** → Change seed or scenario parameters to compare outcomes

---

## Design Rationale

Each scenario teaches a specific cognitive phenomenon:

| Scenario | Teaches | Mechanism |
|----------|---------|-----------|
| The Betrayal | How trust models break | Theory of Mind + negative evidence |
| Overconfidence | System 1 vs System 2 | Metacognition detects doubt, triggers switch |
| Babel | Language emergence | Independent symbol innovation → collision |
| The Innovator | Cultural transmission | Observation → imitation → meme spread |
| Coalition Wars | Emergent geopolitics | Theory of Mind + resource scarcity → factions |
| The Philosopher | Thinking style choice | Confidence metric → System 1/2 switching |
| Emergence | Full complexity | All features interact = maximum emergence |

Progression: Start with simple scenarios (Betrayal, Philosopher) → move to medium (Overconfidence, Babel, Innovator) → master complex dynamics (Coalition Wars, Emergence).

---

## Configuration Format Compliance

All YAMLs validated against `src/config.py`:
- Uses only valid SimulationConfig fields
- Respects data types (booleans, floats, ints, lists)
- No typos in field names
- All metric names are supported
- Output directories follow convention: `data/scenarios/{name}`

---

## Notes for Future Enhancement

Potential extensions (not included, but enabled by this framework):

1. **Conditional checkpoints**: Save/restore state at specific ticks for comparison
2. **Replay controls**: Frame-by-frame review with metadata overlay
3. **Metric dashboards**: Real-time graphs of cooperation, survival, language growth
4. **Scenario variants**: "What if we disable language in Emergence?"
5. **Challenge modes**: "Can you make coalition_wars last 500 ticks without collapse?"
6. **Narrative UI**: Embed key_moments as interactive timeline in player

These are naturally supported by the metadata structure but left for future implementation.

---

## Summary

Users get:
- ✅ 7 carefully curated scenarios showcasing different cognitive phenomena
- ✅ Engaging, accessible README guide (not developer docs)
- ✅ Specific guidance on what to watch & when
- ✅ Lens navigation tips
- ✅ Connections to cognitive science
- ✅ Suggestions for exploration & comparison

This transforms AUTOCOG Playground from "run experiments" to "watch cognition emerge" — positioning it as both a learning tool and a mesmerizing exploration of how minds and societies self-organize.
