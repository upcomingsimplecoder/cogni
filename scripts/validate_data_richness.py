"""Validate what data the simulation actually produces with all systems enabled.

Run with: python scripts/validate_data_richness.py
"""
import json
import sys
sys.path.insert(0, '.')

from src.config import SimulationConfig
from src.simulation.engine import SimulationEngine
from src.visualization.realtime import tick_to_json

# Enable EVERYTHING
config = SimulationConfig(
    num_agents=8,
    seed=42,
    max_ticks=200,
    default_architecture='dual_process',
    theory_of_mind_enabled=True,
    evolution_enabled=True,
    coalitions_enabled=True,
    language_enabled=True,
    metacognition_enabled=True,
)

print("=== INITIALIZING ENGINE WITH ALL SYSTEMS ===")
engine = SimulationEngine(config)

# Check what systems actually got instantiated
systems = {
    'cultural_engine': engine.cultural_engine,
    'metacognition_engine': engine.metacognition_engine,
    'language_engine': engine.language_engine,
    'coalition_manager': engine.coalition_manager,
    'population_manager': getattr(engine, 'population_manager', None),
}
print("\nSystem instantiation:")
for name, obj in systems.items():
    status = f"{type(obj).__name__}" if obj else "NOT INITIALIZED"
    print(f"  {name}: {status}")

# Check ToM wiring
print("\nTheory of Mind wiring:")
for agent in engine.registry.living_agents():
    aid = str(agent.agent_id)
    loop = engine.registry.get_awareness_loop(agent.agent_id)
    if loop:
        has_tom = hasattr(loop, '_tom_modeler') or hasattr(loop, 'mind_modeler')
        strategy = loop.strategy
        strat_name = type(strategy).__name__
        print(f"  {agent.profile.name}: loop={type(loop).__name__}, strategy={strat_name}, tom_wired={has_tom}")
        # Check strategy chain
        inner = getattr(strategy, 'inner_strategy', None)
        if inner:
            print(f"    inner_strategy: {type(inner).__name__}")
            inner2 = getattr(inner, 'inner_strategy', None)
            if inner2:
                print(f"      inner_inner: {type(inner2).__name__}")

# Check social memory
print("\nSocial memory:")
for agent in list(engine.registry.living_agents())[:3]:
    mem = engine.registry.get_memory(agent.agent_id)
    if mem:
        print(f"  {agent.profile.name}: memory={type(mem).__name__}")
        if hasattr(mem, 'social'):
            print(f"    social: {type(mem.social).__name__}")
            if hasattr(mem.social, 'relationships'):
                print(f"    relationships: {len(mem.social.relationships)} entries")
        if hasattr(mem, 'episodic'):
            print(f"    episodic: {type(mem.episodic).__name__}")
            if hasattr(mem.episodic, 'episodes'):
                print(f"    episodes: {len(mem.episodic.episodes)} entries")
    else:
        print(f"  {agent.profile.name}: NO MEMORY")

# Run 100 ticks and check data at intervals
print("\n=== RUNNING 100 TICKS ===")
check_ticks = [1, 10, 25, 50, 75, 100]
for tick_num in range(1, 101):
    tick_record = engine.step_all()

    if tick_num in check_ticks:
        # Get what tick_to_json would produce
        data = tick_to_json(engine, tick_record)

        print(f"\n--- TICK {tick_num} ---")
        print(f"  Living agents: {data['living_count']}")

        # Check per-agent data
        for agent_data in data['agents'][:2]:  # First 2 agents
            name = agent_data.get('name', 'unknown')
            print(f"\n  Agent: {name}")
            print(f"    cultural: {json.dumps(agent_data.get('cultural', {}))[:150]}")
            print(f"    metacognition: {json.dumps(agent_data.get('metacognition', {}))[:200]}")
            print(f"    language: {json.dumps(agent_data.get('language', {}))[:150]}")

        # Check global data
        cultural_global = data.get('cultural', {})
        print(f"\n  Global cultural: tx_events={len(cultural_global.get('transmission_events', []))}, groups={len(cultural_global.get('cultural_groups', []))}")

        metacog_global = data.get('metacognition', {})
        print(f"  Global metacog: {json.dumps(metacog_global)[:200]}")

        language_global = data.get('language', {})
        print(f"  Global language: {json.dumps(language_global)[:200]}")

        # Check emergence events
        events = data.get('emergent_events', [])
        print(f"  Emergent events: {len(events)}")

# Now check social memory and ToM after interactions
print("\n\n=== POST-100-TICK DATA DEPTH ===")
for agent in list(engine.registry.living_agents())[:3]:
    print(f"\nAgent: {agent.profile.name}")

    # Social memory
    mem = engine.registry.get_memory(agent.agent_id)
    if mem and hasattr(mem, 'social') and hasattr(mem.social, 'relationships'):
        rels = mem.social.relationships
        print(f"  Social relationships: {len(rels)}")
        for other_id, rel in list(rels.items())[:3]:
            print(f"    → {str(other_id)[:8]}: trust={rel.trust:.2f}, interactions={rel.interaction_count}")

    # Episodic memory
    if mem and hasattr(mem, 'episodic') and hasattr(mem.episodic, 'episodes'):
        eps = mem.episodic.episodes
        print(f"  Episodic memories: {len(eps)}")

    # Theory of Mind
    # Check if there's a mind state
    loop = engine.registry.get_awareness_loop(agent.agent_id)
    if loop:
        # Look for ToM data through strategy chain
        strat = loop.strategy
        tom_data = None
        while strat:
            if hasattr(strat, 'mind_modeler') or hasattr(strat, '_mind_modeler'):
                modeler = getattr(strat, 'mind_modeler', None) or getattr(strat, '_mind_modeler', None)
                if modeler:
                    print(f"  ToM modeler: {type(modeler).__name__}")
                    if hasattr(modeler, 'models'):
                        print(f"    Models of other agents: {len(modeler.models)}")
                        for other_id, model in list(modeler.models.items())[:2]:
                            print(f"      → {str(other_id)[:8]}: {json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in vars(model).items()})[:200]}")
                break
            strat = getattr(strat, 'inner_strategy', None)

    # Coalitions
    if engine.coalition_manager:
        agent_coalitions = [c for c in engine.coalition_manager.all_coalitions()
                          if str(agent.agent_id) in [str(m) for m in c.members]] if hasattr(engine.coalition_manager, 'all_coalitions') else []
        print(f"  Coalitions: {len(agent_coalitions)}")

print("\n=== VALIDATION COMPLETE ===")
