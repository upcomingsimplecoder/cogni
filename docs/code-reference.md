# Stream 0 Implementation — Code Reference Guide

Quick reference for navigating the 8 additions across all modified files.

---

## File 1: `src/visualization/realtime.py`

### Addition 2: ToM Enrichment (Per Agent)
**Lines**: 325-357
**What**: Extended existing ToM dict comprehension with 5 new fields
**Fields added**: estimated_disposition, prediction_accuracy, ticks_observed, times_helped_me, times_attacked_me
```python
tom_data = {
    "model_count": len(mind_state.models),
    "models": {
        other_id: {
            "trust": getattr(m, 'estimated_trust', 0.5),
            "threat": getattr(m, 'estimated_threat', 0.0),
            "last_seen": m.last_observed_tick,
            "estimated_disposition": m.estimated_disposition,  # NEW
            "prediction_accuracy": m.prediction_accuracy,      # NEW
            "ticks_observed": m.ticks_observed,                # NEW
            "times_helped_me": m.times_helped_me,              # NEW
            "times_attacked_me": m.times_attacked_me,          # NEW
        }
        for other_id, m in mind_state.models.items()
    },
}
```

### Addition 4 & 5: Metacognition Enrichment
**Lines**: 291-327
**What**: Added calibration_curve and deliberation_invoked to metacog_data
**New fields**: calibration_curve (list), deliberation_invoked (bool)
```python
# Calibration curve data
calibration_curve_data = []
if hasattr(mc_state.calibration, 'calibration_curve'):
    calibration_curve_data = [
        {"bin_center": bin_center, "accuracy": accuracy, "count": count}
        for bin_center, accuracy, count in mc_state.calibration.calibration_curve()
    ]

# Deliberation flag
deliberation_invoked = False
if loop and hasattr(loop, '_last_intention') and loop._last_intention:
    deliberation_invoked = getattr(loop._last_intention, 'deliberation_used', False)

metacog_data = {
    ...
    "calibration_curve": calibration_curve_data,    # NEW
    "deliberation_invoked": deliberation_invoked,   # NEW
}
```

### Addition 1: Social Relationships (Per Agent)
**Lines**: 373-389
**What**: Extract social relationships from memory system
**New field**: social_relationships (dict)
```python
social_relationships = {}
memory_tuple = engine.registry.get_memory(agent.agent_id)
if memory_tuple:
    episodic_memory, social_memory = memory_tuple
    if social_memory and hasattr(social_memory, '_relationships'):
        social_relationships = {
            str(other_id): {
                "trust": round(rel.trust, 3),
                "interaction_count": rel.interaction_count,
                "net_resources_given": rel.net_resources_given,
                "was_attacked_by": rel.was_attacked_by,
                "was_helped_by": rel.was_helped_by,
                "last_interaction_tick": rel.last_interaction_tick,
            }
            for other_id, rel in social_memory._relationships.items()
        }
```

### Addition 3: Full SRIE Cascade (Per Agent)
**Lines**: 391-431
**What**: Extract full sensation-reflection-intention from loop
**New field**: srie (dict with 3 sub-dicts)
```python
srie_data = {}
if loop:
    sensation_summary = {}
    if loop._last_sensation:
        s = loop._last_sensation
        sensation_summary = {
            "visible_agent_count": len(s.visible_agents),
            "visible_resource_tiles": sum(1 for t in s.visible_tiles if t.resources),
            "total_resources": sum(qty for t in s.visible_tiles for _, qty in t.resources),
            "message_count": len(s.incoming_messages),
            "time_of_day": s.time_of_day,
        }

    # ... reflection_dict and intention_dict similar ...

    if sensation_summary or reflection_dict or intention_dict:
        srie_data = {
            "sensation_summary": sensation_summary,
            "reflection": reflection_dict,
            "intention": intention_dict,
        }
```

### Addition 6: Plan State (Per Agent)
**Lines**: 433-447
**What**: Extract active plan from planning system
**New field**: plan (dict)
```python
plan_data = {}
if hasattr(engine.registry, 'get_planner'):
    planner = engine.registry.get_planner(agent.agent_id)
    if planner and planner._active_goal_id:
        active_plan = planner._plans.get(planner._active_goal_id)
        active_goal = planner._goals.get(planner._active_goal_id)
        if active_plan and active_goal:
            plan_data = {
                "goal": active_goal.description,
                "steps": len(active_plan.steps),
                "current_step": active_plan.current_step_index,
                "status": active_plan.status,
                "progress": round(active_plan.progress, 2),
            }
```

### Agent Dict Construction
**Lines**: 450-486
**What**: Build agent dict and conditionally add optional fields
```python
agent_dict = {
    ... base fields ...
}

# Add optional fields only if they exist (smart compression)
if social_relationships:
    agent_dict["social_relationships"] = social_relationships
if srie_data:
    agent_dict["srie"] = srie_data
if plan_data:
    agent_dict["plan"] = plan_data

agents_data.append(agent_dict)
```

### Addition 8: Lexicon Similarity Matrix (Global)
**Lines**: 548-565
**What**: Compute pairwise Jaccard similarity between all agent lexicons
**New field**: language.lexicon_similarity (dict with agent_ids and matrix)
```python
language_global = {
    ... existing fields ...
}

# Compute lexicon similarity matrix
living_agent_ids = [str(a.agent_id) for a in engine.registry.living_agents()]
if living_agent_ids:
    language_global["lexicon_similarity"] = engine.language_engine.pairwise_similarity(living_agent_ids)
```

### Addition 7: Per-Agent Language Symbols
**Lines**: 567-594
**What**: Extract individual symbol details from lexicon
**New field**: language.symbols (list)
```python
for agent_data in agents_data:
    aid = agent_data["id"]
    lex = engine.language_engine.get_lexicon(aid)
    if lex:
        symbols_list = []
        if hasattr(lex, '_by_symbol'):
            for sym_form, assocs in lex._by_symbol.items():
                if assocs:
                    best_assoc = max(assocs, key=lambda a: a.strength)
                    if best_assoc.strength >= 0.2:
                        symbols_list.append({
                            "form": sym_form,
                            "meaning": f"{best_assoc.meaning.meaning_type.value}:{best_assoc.meaning.referent}",
                            "success_rate": round(best_assoc.symbol.success_rate, 3),
                            "times_used": best_assoc.symbol.times_used,
                            "strength": round(best_assoc.strength, 3),
                        })

        agent_data["language"] = {
            ... existing fields ...
            "symbols": symbols_list,  # NEW
        }
```

---

## File 2: `src/trajectory/recorder.py`

### Addition 1: Social Relationships
**Lines**: 243-258
**What**: Extract social relationships for trajectory recording
```python
social_relationships_data = {}
memory_tuple = engine.registry.get_memory(agent_record.agent_id)
if memory_tuple:
    episodic_memory, social_memory = memory_tuple
    if social_memory and hasattr(social_memory, '_relationships'):
        social_relationships_data = { ... }
```

### Addition 4 & 5: Metacognition Enrichment
**Lines**: 261-274
**What**: Extract calibration curve and deliberation flag
```python
metacog_calibration_curve = []
metacog_deliberation_invoked = False
if hasattr(engine, 'metacognition_engine') and engine.metacognition_engine:
    aid = str(agent_record.agent_id)
    mc_state = engine.metacognition_engine.get_agent_state(aid)
    if mc_state and hasattr(mc_state.calibration, 'calibration_curve'):
        metacog_calibration_curve = [
            {"bin_center": bin_center, "accuracy": accuracy, "count": count}
            for bin_center, accuracy, count in mc_state.calibration.calibration_curve()
        ]
    if loop and loop._last_intention:
        metacog_deliberation_invoked = getattr(loop._last_intention, 'deliberation_used', False)
```

### Addition 6: Plan State
**Lines**: 276-288
**What**: Extract active plan state
```python
plan_state_data = {}
if hasattr(engine.registry, 'get_planner'):
    planner = engine.registry.get_planner(agent_record.agent_id)
    if planner and planner._active_goal_id:
        active_plan = planner._plans.get(planner._active_goal_id)
        active_goal = planner._goals.get(planner._active_goal_id)
        if active_plan and active_goal:
            plan_state_data = { ... }
```

### Addition 7: Language Symbols
**Lines**: 290-305
**What**: Extract symbol details from lexicon
```python
language_symbols_data = []
if hasattr(engine, 'language_engine') and engine.language_engine:
    aid = str(agent_record.agent_id)
    lex = engine.language_engine.get_lexicon(aid)
    if lex and hasattr(lex, '_by_symbol'):
        for sym_form, assocs in lex._by_symbol.items():
            ... extract best association ...
```

### AgentSnapshot Construction
**Lines**: 307-343 (updated)
**What**: Pass new fields to AgentSnapshot constructor
```python
return AgentSnapshot(
    ... existing fields ...,
    social_relationships=social_relationships_data,           # NEW
    metacog_calibration_curve=metacog_calibration_curve,     # NEW
    metacog_deliberation_invoked=metacog_deliberation_invoked, # NEW
    plan_state=plan_state_data,                              # NEW
    language_symbols=language_symbols_data,                  # NEW
)
```

---

## File 3: `src/trajectory/schema.py`

### New Fields in AgentSnapshot
**Lines**: 77-87
**What**: Added 5 new dataclass fields with default factories
```python
@dataclass
class AgentSnapshot:
    ... existing fields ...

    # Social relationships (Addition 1)
    social_relationships: dict = field(default_factory=dict)

    # Metacognition enrichment (Addition 4 & 5)
    metacog_calibration_curve: list[dict] = field(default_factory=list)
    metacog_deliberation_invoked: bool = False

    # Planning state (Addition 6)
    plan_state: dict = field(default_factory=dict)

    # Language symbols (Addition 7)
    language_symbols: list[dict] = field(default_factory=list)
```

---

## File 4: `src/communication/language_engine.py`

### Addition 8: Pairwise Similarity Method
**Lines**: 762-798 (new method)
**What**: Compute Jaccard similarity matrix between agent lexicons
```python
def pairwise_similarity(self, agent_ids: list[str]) -> dict[str, Any]:
    """Compute Jaccard similarity between each pair of agent lexicons.

    Args:
        agent_ids: List of agent IDs to compute similarity for

    Returns:
        Dict with 'agent_ids' and 'matrix' where matrix[i][j] is the
        Jaccard similarity between agent i and agent j's lexicon
    """
    n = len(agent_ids)
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 1.0
                continue

            lex_i = self._lexicons.get(agent_ids[i])
            lex_j = self._lexicons.get(agent_ids[j])

            if not lex_i or not lex_j:
                matrix[i][j] = 0.0
                continue

            # Jaccard similarity: |intersection| / |union|
            symbols_i = set(lex_i.all_symbols())
            symbols_j = set(lex_j.all_symbols())

            if not symbols_i and not symbols_j:
                matrix[i][j] = 1.0  # Both empty
            elif not symbols_i or not symbols_j:
                matrix[i][j] = 0.0  # One empty
            else:
                intersection = len(symbols_i & symbols_j)
                union = len(symbols_i | symbols_j)
                matrix[i][j] = round(intersection / union, 3) if union > 0 else 0.0

    return {
        "agent_ids": agent_ids,
        "matrix": matrix,
    }
```

---

## Quick Navigation

### To find Addition N:
1. **Addition 1 (Social)**: realtime.py:373, recorder.py:243, schema.py:77
2. **Addition 2 (ToM)**: realtime.py:325
3. **Addition 3 (SRIE)**: realtime.py:391
4. **Addition 4 (Calibration)**: realtime.py:296, recorder.py:261, schema.py:80
5. **Addition 5 (Deliberation)**: realtime.py:308, recorder.py:272, schema.py:81
6. **Addition 6 (Plan)**: realtime.py:433, recorder.py:276, schema.py:84
7. **Addition 7 (Symbols)**: realtime.py:567, recorder.py:290, schema.py:87
8. **Addition 8 (Similarity)**: realtime.py:562, language_engine.py:762

### To test each addition:
```python
# Test Addition 1 (Social)
data['agents'][0]['social_relationships']

# Test Addition 2 (ToM)
data['agents'][0]['tom']['models']['agent_2']['estimated_disposition']

# Test Addition 3 (SRIE)
data['agents'][0]['srie']['sensation_summary']

# Test Addition 4 (Calibration)
data['agents'][0]['metacognition']['calibration_curve']

# Test Addition 5 (Deliberation)
data['agents'][0]['metacognition']['deliberation_invoked']

# Test Addition 6 (Plan)
data['agents'][0]['plan']['goal']

# Test Addition 7 (Symbols)
data['agents'][0]['language']['symbols']

# Test Addition 8 (Similarity)
data['language']['lexicon_similarity']['matrix']
```

---

## Grep Commands for Quick Validation

```bash
# Find all additions in realtime.py
grep -n "Addition [1-8]" src/visualization/realtime.py

# Find all additions in recorder.py
grep -n "Addition [1-8]" src/trajectory/recorder.py

# Find new schema fields
grep -n "social_relationships\|metacog_calibration_curve\|metacog_deliberation_invoked\|plan_state\|language_symbols" src/trajectory/schema.py

# Find pairwise_similarity method
grep -n "def pairwise_similarity" src/communication/language_engine.py
```

---

## Summary

**Total lines modified**: ~285
**Total files modified**: 4
**New methods**: 1
**New schema fields**: 5
**Comments added**: 8 (one per addition)

All code is tagged with `Addition N` comments for easy grepping.
