"""Cogniarch Quickstart — Your First Simulation

This script demonstrates how to run a cogniarch simulation programmatically.
You'll see agents with different personalities (gatherer, explorer, diplomat,
aggressor, survivalist) navigate a procedurally-generated world, make decisions
using the SRIE (Sensation-Reflection-Intention-Expression) cognitive pipeline,
and either survive or perish based on their choices.

Run with:
    python examples/quickstart.py

Watch as agents:
- Sense their environment (nearby resources, other agents, terrain)
- Reflect on threats and opportunities
- Form intentions based on their archetype
- Take actions to satisfy their needs (hunger, thirst, energy)
"""

from src.config import SimulationConfig
from src.simulation.engine import SimulationEngine


def main():
    # Create a small world for quick demonstration
    # 5 agents, 200 ticks max, seed=42 for reproducibility
    config = SimulationConfig(
        world_width=20,
        world_height=20,
        num_agents=5,
        max_ticks=200,
        seed=42,
    )

    # Initialize the simulation engine and spawn agents
    engine = SimulationEngine(config)
    engine.setup_multi_agent()

    print("Cogniarch Simulation Started")
    print(f"  World: {config.world_width}x{config.world_height}")
    print(f"  Agents: {engine.registry.count_living}")
    print()

    # Show initial agent lineup
    print("Agent Roster:")
    for agent in engine.agents:
        print(f"  - {agent.profile.name} ({agent.profile.archetype})")
    print()

    # Run the simulation loop
    print("Running simulation...")
    print()

    # Peek into one agent's mind at tick 10
    peek_tick = 10
    peek_done = False

    while not engine.is_over():
        tick_record = engine.step_all()

        # Show cognitive pipeline at peek_tick
        if engine.state.tick == peek_tick and not peek_done:
            agent = engine.agents[0]  # Look at first living agent
            loop = engine.registry.get_awareness_loop(agent.profile.agent_id)

            print(f"Inside {agent.profile.name}'s mind (Tick {peek_tick}):")

            if loop and loop.last_sensation:
                sensation = loop.last_sensation
                print(f"  SENSATION: Sees {len(sensation.visible_tiles)} tiles, "
                      f"{len(sensation.visible_agents)} agents nearby")

            if loop and loop.last_reflection:
                reflection = loop.last_reflection
                print(f"  REFLECTION: Threat={reflection.threat_level:.2f}, "
                      f"Opportunity={reflection.opportunity_score:.2f}")

            if loop and loop.last_intention:
                intention = loop.last_intention
                print(f"  INTENTION: {intention.primary_goal}")

            print()
            peek_done = True

    # Simulation complete -- print final summary
    print(f"Simulation complete at tick {engine.state.tick}")
    print()
    print("Final Report:")
    print(f"  Survivors: {engine.registry.count_living}/{config.num_agents}")
    print()

    # Show who survived and for how long
    for agent in engine.agents:
        status = "alive" if agent.alive else f"died at tick {agent.ticks_alive}"
        print(f"  {agent.profile.name} ({agent.profile.archetype}): {status}")


if __name__ == "__main__":
    main()
