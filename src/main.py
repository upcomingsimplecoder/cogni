"""Entry point for the cogniarch simulation."""

from __future__ import annotations

import sys
import time

from src.cognition.loop import HardcodedCognitiveLoop
from src.config import SimulationConfig
from src.simulation.engine import SimulationEngine
from src.simulation.renderer import Renderer


def main():
    """Run the simulation."""
    config = SimulationConfig()

    # Parse CLI args
    tick_delay = 0.1
    headless = False
    multi_agent = True
    follow_idx = 0
    checkpoint_path = None
    generate_dashboard = False
    viz_path = None
    serve_dashboard = False
    live_mode = False
    live_port = 8001

    for arg in sys.argv[1:]:
        if arg == "--fast":
            tick_delay = 0.01
        elif arg == "--headless":
            headless = True
            tick_delay = 0.0
        elif arg.startswith("--delay="):
            tick_delay = float(arg.split("=")[1])
        elif arg.startswith("--ticks="):
            config.max_ticks = int(arg.split("=")[1])
        elif arg.startswith("--agents="):
            config.num_agents = int(arg.split("=")[1])
        elif arg == "--single":
            multi_agent = False
        elif arg == "--llm":
            config.llm_strategy_enabled = True
        elif arg.startswith("--llm-url="):
            config.llm_base_url = arg.split("=", 1)[1]
        elif arg.startswith("--llm-key="):
            config.llm_api_key = arg.split("=", 1)[1]
        elif arg.startswith("--model="):
            config.llm_model = arg.split("=", 1)[1]
        elif arg.startswith("--follow="):
            follow_idx = int(arg.split("=")[1])
        elif arg.startswith("--seed="):
            config.seed = int(arg.split("=")[1])
        elif arg == "--record":
            config.trajectory_recording = True
        elif arg == "--dashboard":
            generate_dashboard = True
            config.trajectory_recording = True  # Dashboard requires recording
        elif arg.startswith("--viz="):
            viz_path = arg.split("=", 1)[1]
        elif arg == "--serve":
            serve_dashboard = True
        elif arg.startswith("--architecture="):
            config.default_architecture = arg.split("=", 1)[1]
        elif arg.startswith("--load="):
            checkpoint_path = arg.split("=", 1)[1]
        elif arg.startswith("--checkpoint="):
            config.checkpoint_interval = int(arg.split("=")[1])
        elif arg == "--tom":
            config.theory_of_mind_enabled = True
        elif arg == "--evolution":
            config.evolution_enabled = True
        elif arg == "--coalitions":
            config.coalitions_enabled = True
        elif arg == "--live":
            live_mode = True
        elif arg.startswith("--live-port="):
            live_mode = True
            live_port = int(arg.split("=")[1])
        elif arg == "--language":
            config.language_enabled = True
        elif arg == "--help" or arg == "-h":
            print("cogniarch v0.1.0")
            print()
            print("Usage: python -m src.main [OPTIONS]")
            print()
            print("Options:")
            print("  --agents=N            Number of agents (default: 5)")
            print("  --seed=N              Random seed (default: 42)")
            print("  --ticks=N             Max simulation ticks (default: 5000)")
            print("  --delay=N             Seconds between frames (default: 0.1)")
            print("  --fast                Set delay to 0.01s")
            print("  --headless            No rendering, print summary only")
            print("  --single              Single-agent legacy mode")
            print("  --llm                 Enable LLM-driven decisions")
            print("  --llm-url=URL         LLM endpoint (any OpenAI-compatible URL)")
            print("  --llm-key=KEY         API key for the LLM endpoint")
            print("  --model=NAME          LLM model name (default: opus)")
            print("  --follow=N            Follow agent index N in renderer")
            print("  --record              Enable trajectory recording")
            print(
                "  --dashboard           Auto-generate dashboard after simulation "
                "(requires --record)"
            )
            print(
                "  --viz=PATH            Standalone: load trajectory from PATH, "
                "generate dashboard, exit"
            )
            print("  --serve               Start HTTP server after generating dashboard")
            print(
                "  --architecture=NAME   Cognitive architecture "
                "(reactive/cautious/dual_process/social/planning/optimistic)"
            )
            print("  --load=PATH           Load from checkpoint")
            print("  --checkpoint=N        Auto-checkpoint every N ticks")
            print("  --tom                 Enable Theory of Mind")
            print("  --evolution           Enable genetic/cultural evolution")
            print("  --coalitions          Enable coalition formation")
            print("  --live                Start live browser dashboard (WebSocket)")
            print("  --live-port=N         Live dashboard port (default: 8001)")
            print("  --language            Enable emergent language system")
            print()
            print("Environment variables (override any setting):")
            print("  AUTOCOG_LLM_BASE_URL, AUTOCOG_LLM_API_KEY, AUTOCOG_LLM_MODEL, etc.")
            sys.exit(0)

    # Standalone visualization mode
    if viz_path:
        from pathlib import Path

        from src.trajectory.loader import TrajectoryLoader
        from src.visualization.dashboard import DashboardGenerator

        print(f"  Loading trajectory from {viz_path}...")
        viz_p = Path(viz_path)
        if viz_p.is_dir():
            dataset = TrajectoryLoader.from_run_dir(viz_path)
        else:
            dataset = TrajectoryLoader.from_jsonl(viz_path)
        gen = DashboardGenerator(dataset)

        if viz_p.is_dir():
            output_path = str(viz_p / "dashboard.html")
        else:
            output_path = str(viz_p.parent / "dashboard.html")
        gen.generate(output_path)
        print(f"  Dashboard generated: {output_path}")

        if serve_dashboard:
            from src.visualization.server import start_server

            print("  Starting server...")
            start_server(output_path)

        sys.exit(0)

    # Load from checkpoint if specified
    if checkpoint_path:
        from src.persistence.checkpoint import CheckpointManager

        checkpoint_mgr = CheckpointManager(config.checkpoint_dir)
        engine = checkpoint_mgr.load(checkpoint_path)
        print(f"  Loaded checkpoint from {checkpoint_path}")
    else:
        engine = SimulationEngine(config)

    renderer = Renderer()
    renderer.set_follow(follow_idx)

    if multi_agent:
        _run_multi_agent(
            engine,
            renderer,
            config,
            tick_delay,
            headless,
            generate_dashboard,
            serve_dashboard,
            live_mode,
            live_port,
        )
    else:
        _run_single_agent(engine, renderer, config, tick_delay, headless)


def _run_multi_agent(
    engine: SimulationEngine,
    renderer: Renderer,
    config: SimulationConfig,
    tick_delay: float,
    headless: bool,
    generate_dashboard: bool = False,
    serve_dashboard: bool = False,
    live_mode: bool = False,
    live_port: int = 8001,
) -> None:
    """Run multi-agent simulation."""
    # Setup engine if not loaded from checkpoint
    if engine.state.tick == 0:
        engine.setup_multi_agent()

    print("  cogniarch v0.1.0")
    print(f"  World: {config.world_width}x{config.world_height} | Seed: {config.seed}")
    print(f"  Agents: {engine.registry.count_living} | Max ticks: {config.max_ticks}")
    print(f"  Archetypes: {', '.join(a.profile.archetype for a in engine.agents if a.profile)}")
    if config.default_architecture != "reactive":
        print(f"  Architecture: {config.default_architecture}")
    print()

    # Initialize trajectory recorder if enabled
    recorder = None
    if config.trajectory_recording:
        from src.trajectory.recorder import TrajectoryRecorder

        recorder = TrajectoryRecorder(output_dir=config.trajectory_output_dir)
        recorder.start_run(engine)
        print(f"  Recording trajectory to {recorder.output_dir}")

    # Initialize checkpoint manager if enabled
    checkpoint_mgr = None
    if config.checkpoint_interval > 0:
        from src.persistence.checkpoint import CheckpointManager

        checkpoint_mgr = CheckpointManager(
            checkpoint_dir=config.checkpoint_dir,
            auto_interval=config.checkpoint_interval,
            max_checkpoints=config.checkpoint_max,
        )
        print(f"  Auto-checkpointing every {config.checkpoint_interval} ticks")

    # Initialize live server if enabled
    live_server = None
    if live_mode:
        from src.visualization.realtime import LiveServer, tick_to_json

        live_server = LiveServer(port=live_port, open_browser=True)
        live_server.set_engine(engine)
        live_server.start()
        # Wait for server to be ready before starting simulation
        live_server.wait_until_ready(timeout=5.0)

    try:
        while not engine.is_over():
            tick_record = engine.step_all()

            # Record trajectory if enabled
            if recorder:
                recorder.record_tick(engine, tick_record)

            # Broadcast to live dashboard if enabled
            if live_server:
                live_server.broadcast(tick_to_json(engine, tick_record))

            # Auto-checkpoint if enabled
            if checkpoint_mgr:
                saved_path = checkpoint_mgr.auto_checkpoint(engine)
                if saved_path and not headless:
                    print(f"  Checkpoint saved: {saved_path}")

            if not headless:
                renderer.print_multi_frame(engine, tick_record)
                time.sleep(tick_delay)
            elif live_server:
                # In headless+live mode, throttle so clients can receive frames
                time.sleep(0.05)

        # Final output
        if not headless:
            renderer.print_multi_frame(engine)

        renderer.print_multi_summary(engine)

        # Print performance summary if available
        if engine._perf_monitor is not None:
            summary = engine.perf_monitor.summary
            if summary:
                print()
                print("Performance:")
                avg_tick = summary.get("avg_tick_ms", 0)
                slowest = summary.get("slowest_tick_ms", 0)
                print(f"  Avg tick: {avg_tick:.1f}ms | Slowest: {slowest:.1f}ms")

                strategy_breakdown = summary.get("strategy_breakdown", {})
                if strategy_breakdown:
                    total_strategy_ms = sum(s["total_ms"] for s in strategy_breakdown.values())
                    breakdown_str = " | ".join(
                        f"{name} {s['total_ms'] / total_strategy_ms * 100:.0f}%"
                        for name, s in sorted(
                            strategy_breakdown.items(), key=lambda x: x[1]["total_ms"], reverse=True
                        )[:3]  # Top 3
                    )
                    print(f"  Strategy breakdown: {breakdown_str}")

                llm_calls = summary.get("llm_calls", 0)
                llm_avg = summary.get("llm_avg_ms", 0)
                llm_failures = summary.get("llm_parse_failures", 0)
                if llm_calls > 0:
                    print(
                        f"  LLM: {llm_calls} calls, avg {llm_avg:.0f}ms, "
                        f"{llm_failures} parse failures"
                    )

    except KeyboardInterrupt:
        print(f"\n  Stopped at tick {engine.state.tick}")
        renderer.print_multi_summary(engine)

    finally:
        # Stop live server
        if live_server:
            live_server.stop()

        # Finalize trajectory recording
        if recorder:
            recorder.end_run(engine)
            print(f"  Trajectory saved to {recorder.output_dir / 'trajectory.jsonl'}")

            # Generate dashboard if requested
            if generate_dashboard:
                from src.trajectory.loader import TrajectoryLoader
                from src.visualization.dashboard import DashboardGenerator

                print("  Generating dashboard...")
                dataset = TrajectoryLoader.from_jsonl(str(recorder.output_dir / "trajectory.jsonl"))
                gen = DashboardGenerator(dataset)
                output_path = str(recorder.output_dir / "dashboard.html")
                gen.generate(output_path)
                print(f"  Dashboard: {output_path}")

                if serve_dashboard:
                    from src.visualization.server import start_server

                    print("  Starting server...")
                    start_server(output_path)


def _run_single_agent(
    engine: SimulationEngine,
    renderer: Renderer,
    config: SimulationConfig,
    tick_delay: float,
    headless: bool,
) -> None:
    """Run single-agent simulation (backward compatible)."""
    engine.setup_legacy_mode()
    brain = HardcodedCognitiveLoop()

    print(
        f"  cogniarch v0.1.0 | World {config.world_width}x{config.world_height} | "
        f"Seed {config.seed}"
    )
    print(f"  Max ticks: {config.max_ticks} | Tick delay: {tick_delay}s")
    print()

    try:
        while not engine.is_over():
            action = brain.decide(engine)
            engine.step(action)

            if not headless:
                renderer.print_frame(engine)
                time.sleep(tick_delay)

        if not headless:
            renderer.print_frame(engine)

        if engine.agent and engine.agent.needs.is_alive():
            renderer.print_complete(engine)
        else:
            renderer.print_death(engine)

    except KeyboardInterrupt:
        print(f"\n  Stopped at tick {engine.state.tick}")
        if not headless:
            renderer.print_frame(engine)


if __name__ == "__main__":
    main()
