#!/usr/bin/env bash

# Quick Start Guide: AUTOCOG Playground Scenarios
# Location: experiments/scenarios/

# 📖 Start here:
#   experiments/scenarios/README.md
#   (~2,500 words, vivid explanations, tick-by-tick guidance)

# 🎮 Scenarios to explore:

# 1️⃣  For beginners:
#   - the_betrayal.yaml (5 agents, 20×20, 300 ticks)
#     → Watch Theory of Mind models break after betrayal
#   - the_philosopher.yaml (5 agents, 28×28, 300 ticks)
#     → Watch agents switch between System 1 (fast) and System 2 (slow)

# 2️⃣  For intermediate learners:
#   - overconfidence.yaml (5 agents, 28×28, 400 ticks)
#     → Watch metacognition detect overconfidence and trigger thinking shift
#   - babel.yaml (8 agents, 40×20, 500 ticks)
#     → Watch two groups invent different languages, then collide

# 3️⃣  For advanced:
#   - the_innovator.yaml (8 agents, 32×32, 400 ticks)
#     → Watch cultural memes spread through population via observation
#   - coalition_wars.yaml (10 agents, 40×40, 400 ticks)
#     → Watch geopolitics emerge from Theory of Mind + resource scarcity

# 4️⃣  For experts:
#   - emergence.yaml (10 agents, 36×36, 500 ticks)
#     → All features enabled: language + coalitions + culture + cognition
#     → Maximum emergence complexity

# 🚀 To run a scenario:
# 1. Load the YAML into AUTOCOG playground
# 2. Hit Play
# 3. Switch lenses (keys 1-6) to see different aspects
# 4. Pause/rewind to watch key moments
# 5. Follow README guidance for specific tick markers

# 💡 Key insight:
# These are NOT experiments to measure. They're DEMONSTRATIONS to watch.
# Set replicates: 1, enable trajectory_recording: true, and observe.

# 📊 File structure:
# experiments/scenarios/
# ├── README.md                    (Start here! Main guide for all scenarios)
# ├── DELIVERY.md                  (Detailed delivery summary)
# ├── the_betrayal.yaml            (Trust dynamics, Theory of Mind)
# ├── overconfidence.yaml          (Dual-process cognition, metacognition)
# ├── babel.yaml                   (Emergent language, linguistic collision)
# ├── the_innovator.yaml           (Evolution, cultural transmission)
# ├── coalition_wars.yaml          (Coalitions, geopolitics, competition)
# ├── the_philosopher.yaml         (Cognition, System 1 vs System 2)
# └── emergence.yaml               (Full-stack emergence)

echo "AUTOCOG Playground Scenarios"
echo "============================"
echo ""
echo "📖 Main guide: experiments/scenarios/README.md"
echo ""
echo "7 scenarios ready to explore:"
echo "  1. The Betrayal (beginner)"
echo "  2. Overconfidence (intermediate)"
echo "  3. Babel (intermediate)"
echo "  4. The Innovator (advanced)"
echo "  5. Coalition Wars (advanced)"
echo "  6. The Philosopher (beginner)"
echo "  7. Emergence (expert)"
echo ""
echo "Start with the README for full guidance."
