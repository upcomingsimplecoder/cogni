# 🎬 AUTOCOG Playground Scenarios — Delivery Complete ✅

## Summary

**Created**: 7 pre-built scenario YAML configs + comprehensive narrated README
**Location**: `experiments/scenarios/`
**Total files**: 11 (7 YAML + 3 markdown + 1 script)
**Status**: Ready for use

---

## 📂 Files Overview

### Core Documentation

| File | Purpose | Audience |
|------|---------|----------|
| **README.md** | Main guide with vivid scenario descriptions | Non-expert users |
| **INDEX.md** | Technical summary and design philosophy | Developers/curators |
| **DELIVERY.md** | Detailed delivery checklist and validation | Project leads |
| **QUICKSTART.sh** | Quick reference guide | All users |

### Scenario YAMLs (7 files)

| Scenario | Size | Features | Agents | World | Ticks |
|----------|------|----------|--------|-------|-------|
| **the_betrayal.yaml** | 57 lines | Theory of Mind | 5 | 20×20 | 300 |
| **overconfidence.yaml** | 50 lines | Dual-process + Metacognition | 5 | 28×28 | 400 |
| **babel.yaml** | 54 lines | Emergent Language | 8 | 40×20 | 500 |
| **the_innovator.yaml** | 48 lines | Evolution + Culture | 8 | 32×32 | 400 |
| **coalition_wars.yaml** | 55 lines | Coalitions + ToM | 10 | 40×40 | 400 |
| **the_philosopher.yaml** | 50 lines | Dual-process Cognition | 5 | 28×28 | 300 |
| **emergence.yaml** | 92 lines | All features | 10 | 36×36 | 500 |

**Total YAML**: ~406 lines (well-commented, full configuration)

---

## 🎯 Scenario Matrix

### By Cognitive Mechanism

| Mechanism | Scenarios | Difficulty |
|-----------|-----------|------------|
| **Theory of Mind** | The Betrayal, Coalition Wars, Emergence | Beginner → Advanced |
| **Dual-Process Thinking** | Overconfidence, The Philosopher | Beginner → Intermediate |
| **Metacognition** | Overconfidence, The Philosopher, Emergence | Intermediate → Expert |
| **Emergent Language** | Babel, Emergence | Intermediate → Expert |
| **Cultural Transmission** | The Innovator, Emergence | Advanced → Expert |
| **Coalitions** | Coalition Wars, Emergence | Advanced → Expert |

### By Learning Path

```
Beginner ──────────> Intermediate ──────────> Advanced ──────────> Expert
    ↓                     ↓                        ↓                  ↓
The Betrayal        Overconfidence         The Innovator         Emergence
The Philosopher     Babel                  Coalition Wars
                    (+ others above)
```

---

## 📖 README Quality Metrics

| Aspect | Status | Details |
|--------|--------|---------|
| **Length** | ✅ ~2,500 words | Comprehensive coverage |
| **Accessibility** | ✅ Non-technical | Vivid metaphors, clear language |
| **Specificity** | ✅ Tick-by-tick | "Around tick 80-120, watch for X" |
| **Guidance** | ✅ 6-section structure | Setup → Watch → Why → Lenses → Moments |
| **Voice** | ✅ Nicky Case style | Engaging, beautiful, accessible |
| **Lens explanations** | ✅ Complete | All 6 visualization modes explained |
| **Exploration tips** | ✅ Provided | How to compare, vary, investigate |

---

## 🔧 Configuration Quality

### All scenarios feature:

✅ **Valid YAML syntax** (tested against parser)
✅ **Config.py validation** (all fields are supported attributes)
✅ **Appropriate complexity** (5-10 agents, 300-500 ticks)
✅ **Feature tailoring** (config matched to phenomenon)
✅ **Trajectory recording enabled** (for replay)
✅ **Metadata hints** (for visualization layer)
✅ **Relevant metrics** (measured against the phenomenon)
✅ **Consistent seed** (`seed_start: 42`)
✅ **Single replicate** (`replicates: 1` — for watching, not stats)

---

## 📝 Scenario Descriptions

### The Betrayal (Theory of Mind)
- **Focus**: How trust models break under betrayal
- **Narrative**: "One wolf hides among sheep"
- **Key insight**: Theory of Mind builds mental models of others. One betrayal shatters everything.
- **Difficulty**: Beginner
- **Lens**: Social (watch trust links form then break)

### The Philosopher (Dual-Process Cognition)
- **Focus**: How agents switch between System 1 (fast) and System 2 (slow)
- **Narrative**: "Your brain is two brains"
- **Key insight**: When confident, think fast. When uncertain, think slow.
- **Difficulty**: Beginner
- **Lens**: Cognition (watch confidence meter trigger switches)

### Overconfidence (Metacognition)
- **Focus**: How metacognition detects overconfidence and triggers System 2
- **Narrative**: "Confidence is a superpower—until it isn't"
- **Key insight**: Agents use metacognition to realize they don't know, then switch thinking styles.
- **Difficulty**: Intermediate
- **Lens**: Cognition (watch System 1 → System 2 transition)

### Babel (Emergent Language)
- **Focus**: Language divergence and collision
- **Narrative**: "Geography is destiny"
- **Key insight**: Separate groups invent incompatible languages. When they meet, crisis.
- **Difficulty**: Intermediate
- **Lens**: Language (watch symbols innovate then collide)

### The Innovator (Cultural Transmission)
- **Focus**: How winning strategies spread through observation & imitation
- **Narrative**: "A better idea spreads like a virus"
- **Key insight**: Memes (cultural units) propagate through imitation, not instruction.
- **Difficulty**: Advanced
- **Lens**: Cultural (watch meme adoption cascade)

### Coalition Wars (Coalitions + Theory of Mind)
- **Focus**: Emergent geopolitics and faction competition
- **Narrative**: "Cooperation and betrayal, forever"
- **Key insight**: Agents form alliances, coalitions compete, geopolitics emerges.
- **Difficulty**: Advanced
- **Lens**: Coalition (watch teams form and war)

### Emergence (Full Stack)
- **Focus**: All mechanisms interacting
- **Narrative**: "The full symphony"
- **Key insight**: Language + culture + cognition + coalitions + ToM = maximum complexity.
- **Difficulty**: Expert
- **Lens**: Multi (watch layers of emergence)

---

## 💡 Design Philosophy

### Not experiments. Demonstrations.

Traditional use: Run 100 replicas, collect aggregate statistics.
Playground use: Load 1 scenario, watch 1 run, observe emergence.

### Each scenario teaches one thing.

- Don't mix phenomena
- Focus enables deep understanding
- Clear visual demonstration of mechanism

### Vivid, accessible language.

- Non-experts can understand
- Technical concepts explained via metaphor
- Specific guidance (which tick, which lens)

### Observable mechanics.

- Agents' behavior visibly demonstrates the mechanism
- Users see cause → effect directly
- No black-box neural networks or abstract statistics

---

## 🚀 User Journey

1. **User curious about cognition** → Opens AUTOCOG Playground
2. **Reads README.md** → "Ooh, The Betrayal sounds fascinating"
3. **Loads the_betrayal.yaml** → Into simulation runner
4. **Hits Play** → Agents move, interact in real-time
5. **Switches to Social Lens (key 2)** → Sees trust links forming
6. **Pauses at tick 80** → Watches the betrayal frame-by-frame
7. **Rewinds, watches again** → "Oh, so that's how Theory of Mind breaks"
8. **Loads overconfidence.yaml** → Explores another mechanism
9. **Runs emergence.yaml** → Watches maximum complexity
10. **Varies seed, runs again** → Sees different outcomes from same setup

---

## ✅ Validation Checklist

- ✅ All 7 YAML files created
- ✅ All YAML files syntactically valid
- ✅ All fields validated against src/config.py
- ✅ All scenarios set replicates: 1
- ✅ All scenarios enable trajectory_recording: true
- ✅ All scenarios have scenario_metadata with hints
- ✅ All scenarios have appropriate metrics
- ✅ Output directories follow convention: data/scenarios/{name}
- ✅ README is comprehensive (~2,500 words)
- ✅ README uses vivid, accessible language
- ✅ README includes all 7 scenarios with equal detail
- ✅ README explains all 6 lenses
- ✅ README provides exploration guidance
- ✅ Supporting documentation complete (INDEX, DELIVERY, QUICKSTART)

---

## 📊 Content Breakdown

| Category | Items | Details |
|----------|-------|---------|
| **Scenarios** | 7 | Beginner (2) → Intermediate (2) → Advanced (2) → Expert (1) |
| **Documentation** | 4 files | README (user-facing) + INDEX + DELIVERY + QUICKSTART |
| **YAML content** | ~406 lines | Detailed config + metadata for each scenario |
| **Markdown content** | ~4,000+ lines | Comprehensive guides and documentation |
| **Cognitive phenomena** | 7 types | Theory of Mind, Dual-process, Language, Culture, Coalitions, Metacognition, Emergence |

---

## 🎬 Ready to Deploy

All files are in `experiments/scenarios/` and ready for use:

1. **Users**: Read README.md, load a YAML, hit Play
2. **Developers**: Load scenarios into simulation runner
3. **Educators**: Use as teaching tool for cognitive science
4. **Researchers**: Extend with custom metrics or features

**Next steps**:
- Integrate with visualization/replay system
- Add interactive timeline for key_moments
- Create scenario selector UI
- Optional: Add difficulty badges, estimated run time

---

## 🌟 Highlights

### What makes this delivery special:

1. **Multiple learning paths** — Start simple (Betrayal, Philosopher), progress to complex (Emergence)
2. **Vivid narration** — Scenarios aren't just config, they're stories
3. **Accessible to non-experts** — No background in ML/cognitive science required
4. **Specific guidance** — Users know exactly what to watch and when
5. **Exploration-friendly** — Easy to vary seed and compare outcomes
6. **Well-documented** — Multiple guides for different audiences
7. **Extensible** — Custom metadata structure supports future visualization enhancements

---

**AUTOCOG Playground Scenarios: Ready for watching minds emerge.** 🧠✨
