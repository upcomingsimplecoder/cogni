"""Real-time WebSocket server for live AUTOCOG simulation visualization.

Provides a FastAPI app with WebSocket endpoint that broadcasts TickRecord data
from the simulation engine to connected browser clients in real-time.

Dependencies:
    This module requires FastAPI and uvicorn. Install with:
        pip install fastapi uvicorn[standard]
"""

from __future__ import annotations

import asyncio
import logging
import threading
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

try:
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from src.simulation.engine import SimulationEngine, TickRecord

logger = logging.getLogger(__name__)

# Archetype color mappings (matching src.visualization.dashboard)
ARCHETYPE_COLORS = {
    "gatherer": "#22c55e",
    "explorer": "#06b6d4",
    "diplomat": "#eab308",
    "aggressor": "#ef4444",
    "survivalist": "#f8fafc",
}


class LiveServer:
    """Real-time WebSocket server for broadcasting simulation state.

    Runs a FastAPI + uvicorn server in a daemon thread, broadcasting tick data
    to all connected WebSocket clients.

    Usage:
        server = LiveServer(port=8001, open_browser=True)
        server.set_engine(engine)
        server.start()

        while not engine.is_over():
            tick_record = engine.step_all()
            server.broadcast(tick_to_json(engine, tick_record))

        server.stop()
    """

    def __init__(self, port: int = 8001, open_browser: bool = False):
        """Initialize the live server.

        Args:
            port: Port to serve on (default: 8001)
            open_browser: Whether to open browser automatically (default: False)

        Raises:
            ImportError: If FastAPI and uvicorn are not installed
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError(
                "FastAPI and uvicorn are required for the live server. "
                "Install with: pip install fastapi uvicorn[standard]"
            )

        self.port = port
        self.open_browser = open_browser
        self._clients: list[WebSocket] = []
        self._server_thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self.engine: SimulationEngine | None = None

        # Use lifespan context manager for startup/shutdown events
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            self._ready.set()
            yield

        self.app = FastAPI(title="AUTOCOG Live Simulation", lifespan=lifespan)

        self._setup_routes()

    def _setup_routes(self) -> None:
        """Setup FastAPI routes for WebSocket and static files."""

        # Mount static files for JS modules
        static_path = Path(__file__).parent / "static"
        if static_path.exists():
            self.app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket) -> None:
            await websocket.accept()
            self._clients.append(websocket)
            logger.info(f"Client connected. Total clients: {len(self._clients)}")

            try:
                # Keep connection alive — wait for disconnect
                while True:
                    await websocket.receive_text()
            except WebSocketDisconnect:
                pass
            except Exception as e:
                logger.warning(f"WebSocket error: {e}")
            finally:
                if websocket in self._clients:
                    self._clients.remove(websocket)
                logger.info(f"Client disconnected. Total clients: {len(self._clients)}")

        @self.app.get("/", response_class=HTMLResponse)
        async def serve_dashboard() -> HTMLResponse:
            # Try new template first, fallback to old
            template_path = Path(__file__).parent / "templates" / "live_new.html"
            if not template_path.exists():
                template_path = Path(__file__).parent / "templates" / "live.html"
            if not template_path.exists():
                return HTMLResponse(
                    content="<h1>Live Dashboard Not Found</h1>"
                    "<p>Template file live.html is missing.</p>",
                    status_code=404,
                )
            return HTMLResponse(content=template_path.read_text(encoding="utf-8"))

        @self.app.get("/api/config")
        async def get_config() -> dict[str, Any]:
            if self.engine is None:
                return {"error": "Engine not configured"}

            config = self.engine.config
            return {
                "world_width": config.world_width,
                "world_height": config.world_height,
                "num_agents": config.num_agents,
                "agent_archetypes": config.agent_archetypes,
                "vision_radius": config.vision_radius,
                "communication_range": config.communication_range,
                "ticks_per_day": config.ticks_per_day,
                "max_ticks": config.max_ticks,
                "archetype_colors": ARCHETYPE_COLORS,
                "cultural_transmission_enabled": (
                    hasattr(self.engine, "cultural_engine")
                    and self.engine.cultural_engine is not None
                )
                if self.engine
                else False,
                "metacognition_enabled": (
                    hasattr(self.engine, "metacognition_engine")
                    and self.engine.metacognition_engine is not None
                )
                if self.engine
                else False,
                "language_enabled": (
                    hasattr(self.engine, "language_engine")
                    and self.engine.language_engine is not None
                )
                if self.engine
                else False,
            }

    def set_engine(self, engine: SimulationEngine) -> None:
        """Set the simulation engine for config access."""
        self.engine = engine

    def wait_until_ready(self, timeout: float = 5.0) -> None:
        """Block until the server is accepting connections.

        Args:
            timeout: Max seconds to wait. Logs a warning if exceeded.
        """
        if not self._ready.wait(timeout=timeout):
            logger.warning(f"Live server did not become ready within {timeout}s")

    def broadcast(self, tick_data: dict[str, Any]) -> None:
        """Broadcast tick data to all connected WebSocket clients.

        Thread-safe — schedules async send on the server's event loop.
        Always schedules even if no clients are currently tracked,
        since client list is managed on the async thread.
        """
        if self._loop and not self._loop.is_closed():
            import json

            payload = json.dumps(tick_data)
            asyncio.run_coroutine_threadsafe(self._broadcast_async(payload), self._loop)

    async def _broadcast_async(self, payload: str) -> None:
        """Send pre-serialized tick data to all connected clients."""
        disconnected = []
        for client in self._clients:
            try:
                await client.send_text(payload)
            except Exception:
                disconnected.append(client)

        for client in disconnected:
            if client in self._clients:
                self._clients.remove(client)

    def start(self) -> None:
        """Start the uvicorn server in a daemon thread."""

        def run_server() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            config = uvicorn.Config(
                self.app,
                host="127.0.0.1",
                port=self.port,
                log_level="warning",
                loop="asyncio",
            )
            server = uvicorn.Server(config)

            url = f"http://localhost:{self.port}"
            logger.info(f"Starting live server at {url}")
            print(f"  Live dashboard: {url}")

            if self.open_browser:
                threading.Timer(1.0, lambda: webbrowser.open(url)).start()

            self._loop.run_until_complete(server.serve())

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()

    def stop(self) -> None:
        """Stop the server and cleanup."""
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=2.0)
        logger.info("Live server stopped")


def tick_to_json(engine: SimulationEngine, tick_record: TickRecord) -> dict[str, Any]:
    """Serialize a TickRecord and engine state to a JSON-serializable dict.

    Extracts agent states, intentions, messages, and emergence events
    into a flat structure suitable for WebSocket broadcast.
    """
    # Recent messages
    recent_messages = []
    for msg in engine.message_bus.recent_messages(10):
        recent_messages.append(
            {
                "id": str(msg.id),
                "tick": msg.tick,
                "sender": str(msg.sender_id),
                "receiver": str(msg.receiver_id) if msg.receiver_id else "broadcast",
                "type": msg.message_type.value,
                "content": msg.content,
            }
        )

    # Agent states
    agents_data = []
    for record in tick_record.agent_records:
        agent = engine.registry.get(record.agent_id)
        if not agent:
            continue

        # Action
        action_type = record.action.type.value if record.action else "none"
        action_success = record.result.success if record.result else False

        # Intention (Intention is a dataclass with .primary_goal, .confidence)
        intention_goal = "unknown"
        intention_confidence = 0.0
        loop = engine.registry.get_awareness_loop(agent.agent_id)
        if loop:
            intention = loop.last_intention
            if intention is not None:
                intention_goal = getattr(intention, "primary_goal", "unknown")
                intention_confidence = getattr(intention, "confidence", 0.0)

        # Traits
        traits_data = {}
        if agent.profile and agent.profile.traits:
            traits_data = agent.profile.traits.as_dict()

        # Cultural data (Phase 2)
        cultural_data = {}
        if hasattr(engine, "cultural_engine") and engine.cultural_engine:
            aid = str(agent.agent_id)
            rep = engine.cultural_engine.get_repertoire(aid)
            weights = engine.cultural_engine.get_transmission_weights(aid)
            cultural_data = {
                "learning_style": weights.dominant_style.value if weights else "",
                "adopted_count": rep.adopted_count() if rep else 0,
                "repertoire_size": rep.variant_count() if rep else 0,
                "cultural_group": -1,
            }

        # Metacognitive data (Phase 3) — enriched with calibration curve + deliberation flag
        metacog_data = {}
        if hasattr(engine, "metacognition_engine") and engine.metacognition_engine:
            aid = str(agent.agent_id)
            mc_state = engine.metacognition_engine.get_agent_state(aid)
            if mc_state:
                # Calibration curve data
                calibration_curve_data = []
                if hasattr(mc_state.calibration, "calibration_curve"):
                    calibration_curve_data = [
                        {"bin_center": bin_center, "accuracy": accuracy, "count": count}
                        for bin_center, accuracy, count in mc_state.calibration.calibration_curve()
                    ]

                # Deliberation flag (check if System 2 was invoked this tick)
                deliberation_invoked = False
                if loop and hasattr(loop, "_last_intention") and loop._last_intention:
                    # Check if deliberation_used flag exists in the last intention
                    deliberation_invoked = getattr(loop._last_intention, "deliberation_used", False)

                # Get capability summary with proper type handling
                self_model_summary = {}
                if mc_state.self_model and hasattr(mc_state.self_model, "capability_summary"):
                    self_model_summary = mc_state.self_model.capability_summary()

                metacog_data = {
                    "active_strategy": mc_state.current_strategy_name,
                    "calibration_score": round(mc_state.calibration.calibration_score, 3),
                    "confidence_bias": round(mc_state.calibration.confidence_bias, 3),
                    "deliberation_threshold": round(mc_state.deliberation_threshold, 3),
                    "self_model": self_model_summary,
                    "self_awareness_score": round(
                        engine.metacognition_engine._compute_self_awareness_score(aid), 3
                    ),
                    "total_switches": len(mc_state.switch_history),
                    "calibration_curve": calibration_curve_data,
                    "deliberation_invoked": deliberation_invoked,
                }

        # Theory of Mind data (enriched with full model details)
        tom_data = {}
        if loop:
            strategy = loop.strategy
            mind_state = None
            for _ in range(5):
                if hasattr(strategy, "mind_state"):
                    mind_state = strategy.mind_state
                    break
                if hasattr(strategy, "inner_strategy"):
                    strategy = strategy.inner_strategy
                elif hasattr(strategy, "_inner"):
                    strategy = strategy._inner
                else:
                    break
            if mind_state:
                tom_data = {
                    "model_count": len(mind_state.models),
                    "models": {
                        other_id: {
                            "trust": getattr(m, "estimated_trust", 0.5),
                            "threat": getattr(m, "estimated_threat", 0.0),
                            "last_seen": m.last_observed_tick,
                            "estimated_disposition": m.estimated_disposition,
                            "prediction_accuracy": m.prediction_accuracy,
                            "ticks_observed": m.ticks_observed,
                            "times_helped_me": m.times_helped_me,
                            "times_attacked_me": m.times_attacked_me,
                        }
                        for other_id, m in mind_state.models.items()
                    },
                }

        # Coalition data
        coalition_data = {}
        if hasattr(engine, "coalition_manager") and engine.coalition_manager:
            aid = str(agent.agent_id)
            coalition = engine.coalition_manager.get_coalition(aid)
            if coalition:
                coalition_data = {
                    "coalition_id": coalition.id,
                    "role": "leader" if coalition.leader_id == aid else "member",
                    "goal": coalition.shared_goal or "",
                    "members": list(coalition.members),
                    "cohesion": round(coalition.cohesion, 2),
                }

        # Social relationships (Addition 1)
        social_relationships = {}
        memory_tuple = engine.registry.get_memory(agent.agent_id)
        if memory_tuple:
            episodic_memory, social_memory = memory_tuple
            if social_memory and hasattr(social_memory, "_relationships"):
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

        # Full SRIE cascade (Addition 3)
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

            reflection_dict = {}
            if loop._last_reflection:
                r = loop._last_reflection
                reflection_dict = {
                    "threat_level": r.threat_level,
                    "opportunity_score": r.opportunity_score,
                    "need_trends": dict(r.need_trends),
                    "last_action_succeeded": r.last_action_succeeded,
                    "interaction_count": len(r.recent_interaction_outcomes),
                }

            intention_dict = {}
            if loop._last_intention:
                i = loop._last_intention
                intention_dict = {
                    "primary_goal": i.primary_goal,
                    "confidence": i.confidence,
                    "target_position": i.target_position,
                    "target_agent_id": str(i.target_agent_id) if i.target_agent_id else None,
                }

            if sensation_summary or reflection_dict or intention_dict:
                srie_data = {
                    "sensation_summary": sensation_summary,
                    "reflection": reflection_dict,
                    "intention": intention_dict,
                }

        # Plan state (Addition 6)
        plan_data = {}
        if hasattr(engine.registry, "get_planner"):
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

        agent_dict = {
            "id": str(agent.agent_id),
            "name": agent.profile.name if agent.profile else str(agent.agent_id),
            "archetype": agent.profile.archetype if agent.profile else "unknown",
            "color": ARCHETYPE_COLORS.get(
                agent.profile.archetype if agent.profile else "survivalist",
                "#f8fafc",
            ),
            "x": agent.x,
            "y": agent.y,
            "hunger": round(agent.needs.hunger, 1),
            "thirst": round(agent.needs.thirst, 1),
            "energy": round(agent.needs.energy, 1),
            "health": round(agent.needs.health, 1),
            "alive": agent.alive,
            "action_type": action_type,
            "action_success": action_success,
            "intention": intention_goal,
            "confidence": intention_confidence,
            "monologue": record.internal_monologue,
            "inventory": dict(agent.inventory),
            "traits": traits_data,
            "cultural": cultural_data,
            "metacognition": metacog_data,
            "tom": tom_data,
            "coalition": coalition_data,
        }

        # Add optional fields only if they exist
        if social_relationships:
            agent_dict["social_relationships"] = social_relationships
        if srie_data:
            agent_dict["srie"] = srie_data
        if plan_data:
            agent_dict["plan"] = plan_data

        agents_data.append(agent_dict)

    # Global cultural data (Phase 2)
    cultural_global: dict[str, Any] = {}
    if hasattr(engine, "cultural_engine") and engine.cultural_engine:
        # Transmission events this tick
        tx_events = [
            {
                "observer": e.observer_id,
                "actor": e.actor_id,
                "variant": e.variant_id,
                "bias": e.bias_type,
                "adopted": e.adopted,
            }
            for e in engine.cultural_engine.transmission_events
            if e.tick == tick_record.tick
        ]

        # Cultural stats
        stats = engine.cultural_engine.get_cultural_stats()

        # Cultural groups (compute periodically)
        groups: list[list[str]] = []
        if (
            hasattr(engine, "cultural_analyzer")
            and engine.cultural_analyzer
            and engine.cultural_analyzer._history
        ):
            latest = engine.cultural_analyzer._history[-1]
            cultural_global["variant_frequencies"] = dict(latest.variant_frequencies)
            cultural_global["diversity"] = float(latest.cultural_diversity)

            # Assign group IDs to agents
            group_sets = engine.cultural_analyzer.detect_cultural_groups(engine.cultural_engine)
            groups = [list(g) for g in group_sets]

            # Update per-agent cultural_group
            for agent_data in agents_data:
                aid_str = str(agent_data["id"])  # type: ignore
                for idx, group in enumerate(group_sets):
                    if aid_str in group:
                        if "cultural" in agent_data and agent_data["cultural"]:
                            cultural_dict = agent_data["cultural"]
                            if isinstance(cultural_dict, dict):
                                cultural_dict["cultural_group"] = idx
                        break

        cultural_global["transmission_events"] = list(tx_events)
        cultural_global["cultural_groups"] = list(groups)
        cultural_global["total_adopted"] = stats.get("total_adopted", 0)

    # Global metacognitive data (Phase 3)
    metacog_global = {}
    if hasattr(engine, "metacognition_engine") and engine.metacognition_engine:
        mc_stats = engine.metacognition_engine.get_metacognitive_stats()
        metacog_global = {
            "total_agents_tracked": mc_stats["total_agents_tracked"],
            "avg_calibration_score": round(mc_stats["avg_calibration_score"], 3),
            "strategy_distribution": mc_stats["strategy_distribution"],
            "total_switches": mc_stats["total_switches"],
        }

    # Global language data (Phase 4) — enriched with lexicon similarity matrix (Addition 8)
    language_global = {}
    if hasattr(engine, "language_engine") and engine.language_engine:
        lang_stats = engine.language_engine.get_language_stats()
        language_global = {
            "total_vocabulary": lang_stats["total_vocabulary"],
            "unique_symbols": lang_stats["unique_symbols"],
            "agents_tracked": lang_stats["agents_tracked"],
            "established_conventions": lang_stats["established_conventions"],
            "messages_this_tick": lang_stats["messages_this_tick"],
            "innovations_this_tick": lang_stats["innovations_this_tick"],
            "conventions": lang_stats["convention_list"],
        }

        # Compute lexicon similarity matrix (Addition 8)
        living_agent_ids = [str(a.agent_id) for a in engine.registry.living_agents()]
        if living_agent_ids:
            language_global["lexicon_similarity"] = engine.language_engine.pairwise_similarity(
                living_agent_ids
            )

        # Per-agent language data (enriched with individual symbols - Addition 7)
        for agent_data in agents_data:
            aid_str: str = str(agent_data["id"])  # type: ignore
            lex = engine.language_engine.get_lexicon(aid_str)
            if lex:
                symbols_list = []
                # Extract individual symbol details
                if hasattr(lex, "_by_symbol"):
                    for sym_form, assocs in lex._by_symbol.items():
                        if assocs:
                            best_assoc = max(assocs, key=lambda a: a.strength)
                            if (
                                best_assoc.strength >= 0.2
                            ):  # Only include reasonably strong associations
                                symbols_list.append(
                                    {
                                        "form": sym_form,
                                        "meaning": (
                                            f"{best_assoc.meaning.meaning_type.value}:"
                                            f"{best_assoc.meaning.referent}"
                                        ),
                                        "success_rate": round(best_assoc.symbol.success_rate, 3),
                                        "times_used": best_assoc.symbol.times_used,
                                        "strength": round(best_assoc.strength, 3),
                                    }
                                )

                agent_data["language"] = {
                    "vocabulary_size": lex.vocabulary_size(),
                    "convention_count": lex.convention_count(),
                    "comm_success_rate": round(
                        engine.language_engine.get_communication_success_rate(aid_str), 2
                    ),
                    "symbols": symbols_list,
                }

    # Global coalition data
    coalition_global = {}
    if hasattr(engine, "coalition_manager") and engine.coalition_manager:
        all_c = engine.coalition_manager.all_coalitions()
        coalition_global = {
            "active_count": len(all_c),
            "coalitions": [
                {
                    "id": c.id,
                    "name": c.name,
                    "leader": c.leader_id,
                    "members": list(c.members),
                    "goal": c.shared_goal or "",
                    "cohesion": round(c.cohesion, 2),
                }
                for c in all_c
            ],
            "history_count": len(engine.coalition_manager.coalition_history()),
            "pending_count": len(engine.coalition_manager.pending_proposals()),
        }

    return {
        "tick": tick_record.tick,
        "day": engine.state.day,
        "time_of_day": engine.state.time_of_day,
        "agents": agents_data,
        "emergent_events": [str(e) for e in tick_record.emergent_events],
        "living_count": engine.registry.count_living,
        "dead_count": engine.registry.count_dead,
        "messages": recent_messages,
        "cultural": cultural_global,
        "metacognition": metacog_global,
        "language": language_global,
        "coalitions": coalition_global,
    }
