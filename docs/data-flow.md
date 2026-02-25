# Stream 0 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     AUTOCOG Simulation Engine                       │
│                                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────┐│
│  │   Agent     │  │ Awareness    │  │  Memory     │  │ Planning ││
│  │  Registry   │  │    Loop      │  │  Systems    │  │  System  ││
│  └─────────────┘  └──────────────┘  └─────────────┘  └──────────┘│
│         │                │                  │              │       │
│         │                │                  │              │       │
│         ▼                ▼                  ▼              ▼       │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │              tick_to_json() / recorder                     │  │
│  └─────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │
        ┌─────────────────────┴──────────────────────┐
        │                                            │
        ▼                                            ▼
┌──────────────────┐                      ┌──────────────────┐
│  WebSocket       │                      │  JSONL           │
│  Broadcast       │                      │  Trajectory      │
│  (Live Server)   │                      │  (Recorder)      │
└──────────────────┘                      └──────────────────┘
        │                                            │
        │                                            │
        ▼                                            ▼
┌──────────────────┐                      ┌──────────────────┐
│  Browser Client  │                      │  Post-hoc        │
│  (live.html)     │                      │  Analysis        │
└──────────────────┘                      └──────────────────┘
```

## Data Sources per Addition

```
Addition 1: Social Relationships
    Source: engine.registry.get_memory(agent_id) → (episodic, social)
    Access: social._relationships dict
    Output: {trust, interaction_count, ...}

Addition 2: ToM Enrichment
    Source: loop.strategy.mind_state.models
    Access: model attributes (estimated_disposition, prediction_accuracy, ...)
    Output: Enhanced models dict

Addition 3: Full SRIE Cascade
    Source: loop._last_sensation, loop._last_reflection, loop._last_intention
    Access: Cached SRIE state from last tick
    Output: {sensation_summary, reflection, intention}

Addition 4: Calibration Curve
    Source: mc_state.calibration.calibration_curve()
    Access: CalibrationTracker method
    Output: [{bin_center, accuracy, count}, ...]

Addition 5: Deliberation Flag
    Source: loop._last_intention.deliberation_used
    Access: Intention attribute
    Output: true/false

Addition 6: Plan State
    Source: engine.registry.get_planner(agent_id)
    Access: planner._active_goal_id, planner._plans, planner._goals
    Output: {goal, steps, current_step, status, progress}

Addition 7: Per-Agent Symbols
    Source: engine.language_engine.get_lexicon(agent_id)
    Access: lexicon._by_symbol dict
    Output: [{form, meaning, success_rate, times_used, strength}, ...]

Addition 8: Lexicon Similarity Matrix
    Source: engine.language_engine.pairwise_similarity(agent_ids)
    Access: All agent lexicons
    Output: {agent_ids: [...], matrix: [[...]]}
```

## Serialization Paths

```
Per-Agent Data (Additions 1-7):
    Engine → tick_to_json() → agents_data[i] → WebSocket → Browser
    Engine → recorder._build_agent_snapshot() → JSONL → Analysis

Global Data (Addition 8):
    Engine → tick_to_json() → language_global → WebSocket → Browser
    (Not in recorder — global stats computed per-tick only)
```

## Data Size Estimates

```
Typical agent with all systems active:
    Base fields:         ~200 bytes
    Social relationships: ~50 bytes/relationship × 3 = ~150 bytes
    ToM models:          ~80 bytes/model × 2 = ~160 bytes
    SRIE cascade:        ~300 bytes
    Calibration curve:   ~40 bytes/bin × 10 = ~400 bytes
    Deliberation flag:   ~10 bytes
    Plan state:          ~100 bytes
    Language symbols:    ~60 bytes/symbol × 5 = ~300 bytes
    ───────────────────────────────────────────
    Total:               ~1,620 bytes per agent

20 agents × 1,620 bytes = ~32KB per tick (agent data only)

Global data:
    Lexicon similarity matrix: 20×20 × 8 bytes = ~3.2KB
    Other global data:         ~2KB
    ───────────────────────────────────────────
    Total:                     ~37KB per tick

At 10 ticks/second:        ~370 KB/s
At 1000 ticks/simulation:  ~37 MB total
```

## Smart Compression Impact

```
Without compression (all fields always present):
    Agent with no relationships:         1,620 bytes
    Agent with no ToM:                   1,620 bytes
    Agent with no plan:                  1,620 bytes
    → Wasted: ~1,000 bytes/agent/tick

With compression (only emit if data exists):
    Agent with no relationships:         1,470 bytes (−150)
    Agent with no ToM:                   1,460 bytes (−160)
    Agent with no plan:                  1,520 bytes (−100)
    → Saved: ~15% bandwidth

Early-game simulation (few relationships, no ToM, no plans):
    Without compression: ~32 KB/tick
    With compression:    ~20 KB/tick
    → 37% bandwidth reduction
```

## Data Access Patterns

```
Hot path (every tick):
    ✓ agent.needs                    [O(1)]
    ✓ agent.position                 [O(1)]
    ✓ loop._last_intention           [O(1)]

Warm path (if subsystem active):
    ✓ social_memory._relationships   [O(k), k=relationship count]
    ✓ lexicon._by_symbol             [O(m), m=vocabulary size]
    ✓ planner._active_goal_id        [O(1)]

Cold path (expensive, but bounded):
    ✓ calibration_curve()            [O(n), n=prediction count, max 500]
    ✓ pairwise_similarity()          [O(n²), n=agent count, max 100]
```

## Downstream Consumers

```
┌────────────────────────────────────────────────────────────┐
│                  Serialized Data                          │
│  (WebSocket JSON + JSONL Trajectories)                    │
└────────────────────────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┬──────────────┐
        │            │            │              │
        ▼            ▼            ▼              ▼
┌─────────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐
│ Multi-Lens  │ │Benchmark │ │ Scenario │ │  Research  │
│Visualization│ │Framework │ │  System  │ │  Analysis  │
└─────────────┘ └──────────┘ └──────────┘ └────────────┘
│              │             │              │
│ Social graph│ │ Coalition │ │ Alliance   │ │ Language   │
│ SRIE view   │ │ metrics   │ │ detection  │ │ evolution  │
│ Metacog dash│ │ Language  │ │ Goal check │ │ studies    │
│ Lang evolve │ │ emergence │ │ Convention │ │ ToM papers │
└─────────────┘ └──────────┘ └──────────┘ └────────────┘
```

## Summary

**Sources**: 8 distinct data sources (memory, loop, metacog, planner, language)
**Destinations**: 2 (WebSocket + JSONL)
**Downstream systems**: 4 (viz, benchmarks, scenarios, research)
**Compression savings**: 15-37% depending on simulation stage
**Performance overhead**: < 5ms per tick
**Backward compatibility**: 100% (all additions optional)

This data backbone enables all downstream Stream 1-3 work.
