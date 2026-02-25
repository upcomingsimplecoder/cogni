"""Custom Archetype Tutorial: Creating and Using Custom Agent Personalities

This tutorial demonstrates how to:
1. Define a custom archetype with specific personality traits
2. Register it in the ARCHETYPES dictionary
3. Configure a simulation to include your custom archetype
4. Run a short simulation and observe behavior
5. Compare trait evolution between different archetypes

EDUCATIONAL FOCUS:
- Understanding how personality traits influence decision-making
- Learning the archetype registration pattern
- Observing how traits evolve through experience
- Comparing behavior across different personality profiles

WHY CREATE CUSTOM ARCHETYPES?
- Test specific personality configurations
- Model real-world agent behaviors (e.g., scientist, merchant, guard)
- Explore emergence patterns from specific trait combinations
- Run controlled experiments on trait-behavior relationships

HOW TO RUN:
    python examples/custom_archetype.py

WHAT TO EXPECT:
- 5 agents spawn: 1 scientist (custom) + 4 built-in archetypes
- Scientists exhibit high exploration behavior due to curiosity=0.85
- Trait evolution shows how experiences shape personalities
- Comparison table reveals behavioral differences
"""

from src.agents.archetypes import ARCHETYPES
from src.agents.identity import PersonalityTraits
from src.config import SimulationConfig
from src.simulation.engine import SimulationEngine


def define_scientist_archetype() -> None:
    """Define and register the 'scientist' archetype.

    TRAIT DESIGN PHILOSOPHY:
    - High curiosity (0.85): Scientists explore to discover patterns
    - Moderate-high cooperation (0.7): Science benefits from collaboration
    - Moderate risk tolerance (0.6): Willing to take calculated risks
    - Moderate resource sharing (0.6): Share knowledge/resources for collective benefit
    - Low aggression (0.15): Focus on discovery, not conflict
    - Moderate sociability (0.6): Seek interaction for knowledge exchange

    This creates an agent that:
    - Explores extensively (high curiosity)
    - Seeks out other agents for information (sociability + cooperation)
    - Takes measured risks in pursuit of discoveries (risk tolerance)
    - Avoids conflict to focus on research (low aggression)
    """
    ARCHETYPES["scientist"] = {
        "description": "Explores methodically, shares discoveries, collaborates on resource optimization",
        "traits": PersonalityTraits(
            cooperation_tendency=0.7,   # Collaborative, believes in knowledge sharing
            curiosity=0.85,              # Driven to explore and understand
            risk_tolerance=0.6,          # Takes calculated risks for discovery
            resource_sharing=0.6,        # Shares findings and resources
            aggression=0.15,             # Avoids conflict, focuses on research
            sociability=0.6,             # Seeks peers for knowledge exchange
        ),
        "color": "blue",                 # Blue = knowledge, reason
        "symbol": "Sc",                  # Sc for Scientist
    }
    print("[OK] Registered 'scientist' archetype")
    print(f"  Traits: {ARCHETYPES['scientist']['traits'].as_dict()}")


def create_simulation_config() -> SimulationConfig:
    """Create a config that includes the scientist archetype.

    KEY CONFIGURATION CHOICES:
    - agent_archetypes: Include "scientist" so it gets spawned
    - num_agents: 5 agents total (1 of each archetype + scientist)
    - max_ticks: 150 ticks for meaningful trait evolution
    - default_architecture: "reactive" for fast, instinct-driven decisions
    - trait_learning_rate: 0.01 = gradual trait evolution (prevents wild swings)

    The setup_multi_agent() method reads agent_archetypes and spawns one agent
    per archetype in a round-robin fashion until num_agents is reached.
    """
    config = SimulationConfig(
        world_width=24,
        world_height=24,
        seed=42,
        num_agents=5,
        max_ticks=150,  # Short run for tutorial (150 ticks ≈ 6 simulated days)
        agent_archetypes=["scientist", "gatherer", "explorer", "diplomat", "aggressor"],
        default_architecture="reactive",
        trait_learning_rate=0.01,  # Slow but steady trait evolution
        vision_radius=5,
        communication_range=8,
        # Disable advanced features for cleaner tutorial
        evolution_enabled=False,
        coalitions_enabled=False,
        language_enabled=False,
        metacognition_enabled=False,
    )
    print(f"\n[OK] Created SimulationConfig")
    print(f"  Archetypes: {config.agent_archetypes}")
    print(f"  Agents: {config.num_agents}, Ticks: {config.max_ticks}")
    return config


def run_simulation(config: SimulationConfig) -> SimulationEngine:
    """Run the simulation and return the engine for analysis.

    SIMULATION PHASES:
    1. setup_multi_agent(): Spawns agents from archetypes
    2. step_all() loop: Each tick, all agents:
       - Sense environment
       - Decide action based on traits
       - Execute action
       - Update memories
       - Evolve traits based on outcomes
    3. Termination: When max_ticks reached or all agents dead
    """
    engine = SimulationEngine(config)
    engine.setup_multi_agent()

    print(f"\n{'='*60}")
    print("SIMULATION START")
    print(f"{'='*60}")
    print(f"Initial agent population: {len(engine.agents)}")

    for agent in engine.agents:
        if agent.profile:
            print(f"  - {agent.profile.name} ({agent.profile.archetype})")

    print(f"\n{'='*60}")
    print("RUNNING SIMULATION (150 ticks)...")
    print(f"{'='*60}\n")

    # Run simulation
    tick_count = 0
    while not engine.is_over():
        engine.step_all()
        tick_count += 1

        # Progress indicator every 30 ticks
        if tick_count % 30 == 0:
            living = len(engine.agents)
            print(f"Tick {tick_count:3d}: {living} agents alive")

    print(f"\n{'='*60}")
    print("SIMULATION COMPLETE")
    print(f"{'='*60}")
    print(f"Final tick: {engine.state.tick}")
    print(f"Surviving agents: {len(engine.agents)}/{config.num_agents}")

    return engine


def analyze_trait_evolution(engine: SimulationEngine) -> None:
    """Analyze and display trait evolution for all agents.

    WHAT THIS SHOWS:
    - Initial vs final trait values (evolution over 150 ticks)
    - Which traits changed most (indicates experience impact)
    - Archetype-specific behavioral patterns

    KEY INSIGHTS TO LOOK FOR:
    - Scientists should show high curiosity maintenance
    - Aggressors may increase aggression if successful
    - Diplomats should increase cooperation/sharing
    - Gatherers focus on resource accumulation (low sharing)
    """
    print(f"\n{'='*60}")
    print("TRAIT EVOLUTION ANALYSIS")
    print(f"{'='*60}\n")

    if not engine.agents:
        print("No surviving agents to analyze.")
        return

    # Collect trait data
    trait_data = []
    for agent in engine.agents:
        if not agent.profile:
            continue

        traits = agent.profile.traits.as_dict()
        trait_data.append({
            "name": agent.profile.name,
            "archetype": agent.profile.archetype,
            "ticks_alive": agent.ticks_alive,
            "traits": traits,
        })

    # Sort by archetype for cleaner display
    trait_data.sort(key=lambda x: x["archetype"])

    # Display trait table
    print(f"{'Agent':<18} {'Archetype':<12} {'Alive':<6} | {'Coop':>5} {'Curio':>5} {'Risk':>5} {'Share':>5} {'Aggr':>5} {'Soc':>5}")
    print("-" * 90)

    for data in trait_data:
        t = data["traits"]
        print(
            f"{data['name']:<18} {data['archetype']:<12} {data['ticks_alive']:<6} | "
            f"{t['cooperation_tendency']:>5.2f} {t['curiosity']:>5.2f} {t['risk_tolerance']:>5.2f} "
            f"{t['resource_sharing']:>5.2f} {t['aggression']:>5.2f} {t['sociability']:>5.2f}"
        )

    # Highlight scientist behavior
    scientist_data = next((d for d in trait_data if d["archetype"] == "scientist"), None)
    if scientist_data:
        print(f"\n{'='*60}")
        print("SCIENTIST ARCHETYPE SPOTLIGHT")
        print(f"{'='*60}")
        print(f"Name: {scientist_data['name']}")
        print(f"Survived: {scientist_data['ticks_alive']} ticks")
        print(f"\nKey Traits:")
        t = scientist_data["traits"]
        print(f"  Curiosity:   {t['curiosity']:.3f} (started at 0.85)")
        print(f"  Cooperation: {t['cooperation_tendency']:.3f} (started at 0.70)")
        print(f"  Aggression:  {t['aggression']:.3f} (started at 0.15)")
        print(f"\nBehavioral Pattern:")
        if t['curiosity'] > 0.75:
            print("  [+] Maintained high exploration drive")
        if t['cooperation_tendency'] > 0.6:
            print("  [+] Remained collaborative")
        if t['aggression'] < 0.25:
            print("  [+] Avoided conflict successfully")


def compare_archetypes(engine: SimulationEngine) -> None:
    """Compare survival and behavior metrics across archetypes.

    METRICS EXPLAINED:
    - Ticks Alive: Survival fitness (higher = better adaptation)
    - Curiosity: Exploration tendency (affects movement patterns)
    - Cooperation: Willingness to help others (affects social dynamics)
    - Aggression: Conflict tendency (affects combat behavior)

    EXPECTED PATTERNS:
    - Explorers: High movement, moderate survival
    - Gatherers: High resource accumulation, good survival
    - Diplomats: High cooperation, moderate survival
    - Aggressors: High conflict, variable survival
    - Scientists: High exploration + cooperation, good survival (if balanced)
    """
    print(f"\n{'='*60}")
    print("ARCHETYPE COMPARISON")
    print(f"{'='*60}\n")

    # Group by archetype
    archetype_groups: dict[str, list] = {}
    for agent in engine.agents:
        if not agent.profile:
            continue
        arch = agent.profile.archetype
        if arch not in archetype_groups:
            archetype_groups[arch] = []
        archetype_groups[arch].append(agent)

    print(f"{'Archetype':<15} {'Count':<7} {'Avg Survival':<15} {'Avg Curiosity':<15} {'Avg Cooperation':<17} {'Avg Aggression':<15}")
    print("-" * 105)

    for archetype in sorted(archetype_groups.keys()):
        agents = archetype_groups[archetype]
        avg_survival = sum(a.ticks_alive for a in agents) / len(agents)
        avg_curiosity = sum(a.profile.traits.curiosity for a in agents if a.profile) / len(agents)
        avg_cooperation = sum(a.profile.traits.cooperation_tendency for a in agents if a.profile) / len(agents)
        avg_aggression = sum(a.profile.traits.aggression for a in agents if a.profile) / len(agents)

        print(
            f"{archetype:<15} {len(agents):<7} {avg_survival:<15.1f} "
            f"{avg_curiosity:<15.3f} {avg_cooperation:<17.3f} {avg_aggression:<15.3f}"
        )


def display_tutorial_summary() -> None:
    """Display educational summary of what was demonstrated."""
    print(f"\n{'='*60}")
    print("TUTORIAL SUMMARY: What You Learned")
    print(f"{'='*60}\n")

    print("1. ARCHETYPE DEFINITION:")
    print("   - Archetypes are personality templates stored in ARCHETYPES dict")
    print("   - Each has: description, traits, color, symbol")
    print("   - Traits are PersonalityTraits dataclass instances")
    print()

    print("2. TRAIT DESIGN:")
    print("   - Each trait is 0.0-1.0 (low to high)")
    print("   - Traits bias decisions, don't determine them")
    print("   - Design traits to match desired behavior profile")
    print()

    print("3. REGISTRATION:")
    print("   - Add to ARCHETYPES dict: ARCHETYPES['name'] = {...}")
    print("   - Include in config.agent_archetypes list")
    print("   - setup_multi_agent() spawns from this list")
    print()

    print("4. TRAIT EVOLUTION:")
    print("   - Traits shift based on action outcomes")
    print("   - Learning rate controls evolution speed")
    print("   - Successful behaviors reinforce associated traits")
    print()

    print("5. BEHAVIORAL EMERGENCE:")
    print("   - Personality + environment + experience = behavior")
    print("   - Same archetype can evolve differently")
    print("   - Social interactions shape trait evolution")
    print()

    print("NEXT STEPS:")
    print("  - Modify scientist traits and observe changes")
    print("  - Create your own archetype (e.g., 'merchant', 'guard')")
    print("  - Increase max_ticks to observe longer-term evolution")
    print("  - Enable coalitions/language to study social complexity")
    print()


def main():
    """Main tutorial execution."""
    print("\n" + "="*60)
    print("CUSTOM ARCHETYPE TUTORIAL")
    print("Creating a 'Scientist' Agent Personality")
    print("="*60 + "\n")

    # Step 1: Define and register custom archetype
    define_scientist_archetype()

    # Step 2: Create simulation config
    config = create_simulation_config()

    # Step 3: Run simulation
    engine = run_simulation(config)

    # Step 4: Analyze results
    analyze_trait_evolution(engine)
    compare_archetypes(engine)

    # Step 5: Educational summary
    display_tutorial_summary()

    print("="*60)
    print("TUTORIAL COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
