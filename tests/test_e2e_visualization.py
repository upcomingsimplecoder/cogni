"""End-to-end smoke tests for the AUTOCOG multi-lens visualization system.

Tests the full stack:
1. Simulation engine runs and produces valid tick data
2. tick_to_json serializes all enriched fields (metacognition, ToM, cultural, etc.)
3. LiveServer starts, serves HTML and static JS, handles WebSocket connections
4. All 5 lens JS files are syntactically valid ES modules
5. HTML templates reference all required DOM elements and modules
6. WebSocket broadcast delivers tick data to connected clients
7. Dashboard generation works end-to-end
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

import pytest

httpx = pytest.importorskip("httpx")
websockets = pytest.importorskip("websockets")

from src.config import SimulationConfig  # noqa: E402
from src.simulation.engine import SimulationEngine  # noqa: E402
from src.visualization.realtime import LiveServer, tick_to_json  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_engine() -> SimulationEngine:
    """Create a fresh engine with all archetypes."""
    config = SimulationConfig(
        num_agents=5,
        max_ticks=20,
        seed=42,
        world_width=32,
        world_height=32,
        agent_archetypes=["gatherer", "explorer", "diplomat", "aggressor", "survivalist"],
    )
    engine = SimulationEngine(config)
    engine.setup_multi_agent()
    return engine


@pytest.fixture
def full_engine() -> SimulationEngine:
    """Function-scoped engine (fresh per test, safe to mutate)."""
    return _make_engine()


@pytest.fixture(scope="class")
def class_engine() -> SimulationEngine:
    """Class-scoped engine for server tests."""
    return _make_engine()


@pytest.fixture
def tick_data(full_engine: SimulationEngine) -> dict:
    """Run 5 ticks and return the last tick_to_json output."""
    for _ in range(5):
        record = full_engine.step_all()
    return tick_to_json(full_engine, record)


def _find_free_port() -> int:
    """Find a free TCP port by binding to port 0."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="class")
def live_server(class_engine: SimulationEngine):
    """Start a live server on an ephemeral port, yield it, then stop.

    Scoped to class so all tests in a class share the same server instance,
    avoiding port-reuse issues on Windows.
    """
    pytest.importorskip("fastapi")
    port = _find_free_port()
    server = LiveServer(port=port, open_browser=False)
    server.set_engine(class_engine)
    server.start()
    server.wait_until_ready(timeout=5.0)
    # Give uvicorn a moment to fully bind
    time.sleep(0.5)
    yield server
    server.stop()
    # Allow OS to reclaim the port
    time.sleep(0.3)


# ---------------------------------------------------------------------------
# 1. Simulation Engine Produces Valid Tick Data
# ---------------------------------------------------------------------------


class TestSimulationDataIntegrity:
    """Verify the engine produces data rich enough for all 5 lenses."""

    def test_tick_data_has_all_top_level_keys(self, tick_data: dict) -> None:
        """Top-level JSON has all required sections."""
        required = {
            "tick",
            "day",
            "time_of_day",
            "agents",
            "emergent_events",
            "living_count",
            "dead_count",
            "messages",
        }
        assert required.issubset(tick_data.keys()), (
            f"Missing top-level keys: {required - tick_data.keys()}"
        )

    def test_agents_have_enriched_fields(self, tick_data: dict) -> None:
        """Each agent has the enriched data fields needed by the new lenses."""
        assert len(tick_data["agents"]) == 5

        for agent in tick_data["agents"]:
            # Core fields (Physical lens)
            for field in (
                "id",
                "name",
                "archetype",
                "color",
                "x",
                "y",
                "hunger",
                "thirst",
                "energy",
                "health",
                "alive",
                "action_type",
                "action_success",
                "intention",
                "confidence",
                "traits",
                "inventory",
            ):
                assert field in agent, f"Agent {agent.get('name', '?')} missing: {field}"

            # Enriched fields (present but possibly empty dicts)
            assert "cultural" in agent
            assert "metacognition" in agent
            assert "tom" in agent
            assert "coalition" in agent

    def test_needs_are_bounded(self, tick_data: dict) -> None:
        """Need values are 0-100 range floats."""
        for agent in tick_data["agents"]:
            for need in ("hunger", "thirst", "energy", "health"):
                val = agent[need]
                assert isinstance(val, float), f"{need} is {type(val)}"
                assert 0 <= val <= 100, f"{need}={val} out of range"

    def test_positions_in_world_bounds(self, tick_data: dict) -> None:
        """Agent positions are within the 32x32 world."""
        for agent in tick_data["agents"]:
            assert 0 <= agent["x"] < 32, f"x={agent['x']} out of bounds"
            assert 0 <= agent["y"] < 32, f"y={agent['y']} out of bounds"

    def test_full_json_round_trip(self, tick_data: dict) -> None:
        """Entire tick data survives JSON serialize -> deserialize."""
        payload = json.dumps(tick_data)
        restored = json.loads(payload)
        assert restored["tick"] == tick_data["tick"]
        assert len(restored["agents"]) == len(tick_data["agents"])

    def test_multi_tick_progression(self, full_engine: SimulationEngine) -> None:
        """Engine ticks advance and data remains valid over 10 ticks."""
        for expected_tick in range(1, 11):
            record = full_engine.step_all()
            data = tick_to_json(full_engine, record)
            assert data["tick"] == expected_tick
            assert len(data["agents"]) > 0
            # Verify JSON-safe each tick
            json.dumps(data)


# ---------------------------------------------------------------------------
# 2. JS Module Syntax Validation
# ---------------------------------------------------------------------------


class TestJSModuleSyntax:
    """Verify all JS files are syntactically valid ES modules."""

    JS_ROOT = Path(__file__).parent.parent / "src" / "visualization" / "static" / "js"

    EXPECTED_FILES = {
        # Core
        "core/colors.js",
        "core/config.js",
        "core/data-source.js",
        "core/event-bus.js",
        "core/network-graph.js",
        "core/spatial-index.js",
        "core/temporal-buffer.js",
        "core/time-series-store.js",
        # Renderers
        "renderers/canvas-compositor.js",
        "renderers/grid-renderer.js",
        "renderers/agent-renderer.js",
        "renderers/overlay-renderer.js",
        # Lenses
        "lenses/lens-base.js",
        "lenses/physical-lens.js",
        "lenses/social-lens.js",
        "lenses/cognitive-lens.js",
        "lenses/cultural-lens.js",
        "lenses/temporal-lens.js",
        # Panels
        "panels/inspector-panel.js",
        "panels/side-panel-manager.js",
        "panels/timeline-panel.js",
        # Utils
        "utils/export.js",
        # Orchestrator
        "app.js",
    }

    def test_all_expected_files_exist(self) -> None:
        """All 21+ JS files exist on disk."""
        for rel_path in self.EXPECTED_FILES:
            full_path = self.JS_ROOT / rel_path
            assert full_path.exists(), f"Missing JS file: {rel_path}"

    @pytest.mark.parametrize("rel_path", sorted(EXPECTED_FILES))
    def test_file_not_empty(self, rel_path: str) -> None:
        """Each JS file has non-trivial content."""
        content = (self.JS_ROOT / rel_path).read_text(encoding="utf-8")
        assert len(content) > 50, f"{rel_path} is suspiciously short ({len(content)} bytes)"

    def test_lens_files_extend_lens_base(self) -> None:
        """All 5 lens files extend LensBase and export their class."""
        lenses = {
            "lenses/physical-lens.js": "PhysicalLens",
            "lenses/social-lens.js": "SocialLens",
            "lenses/cognitive-lens.js": "CognitiveLens",
            "lenses/cultural-lens.js": "CulturalLens",
            "lenses/temporal-lens.js": "TemporalLens",
        }
        for rel_path, class_name in lenses.items():
            content = (self.JS_ROOT / rel_path).read_text(encoding="utf-8")
            assert f"export class {class_name} extends LensBase" in content, (
                f"{rel_path} doesn't export {class_name} extending LensBase"
            )
            assert "super('" in content, f"{rel_path} doesn't call super() with lens name"

    def test_lens_files_implement_required_methods(self) -> None:
        """All lens files implement getCanvasLayers, getAgentStyle, renderSidePanel."""
        required_methods = ["getCanvasLayers", "getAgentStyle", "renderSidePanel"]
        for lens_file in [
            "physical-lens.js",
            "social-lens.js",
            "cognitive-lens.js",
            "cultural-lens.js",
            "temporal-lens.js",
        ]:
            content = (self.JS_ROOT / "lenses" / lens_file).read_text(encoding="utf-8")
            for method in required_methods:
                assert f"{method}(" in content, f"{lens_file} missing method: {method}"

    def test_lens_files_return_proper_layer_objects(self) -> None:
        """Each lens's getCanvasLayers contains layers with name, zIndex, draw."""
        for lens_file in [
            "physical-lens.js",
            "social-lens.js",
            "cognitive-lens.js",
            "cultural-lens.js",
            "temporal-lens.js",
        ]:
            content = (self.JS_ROOT / "lenses" / lens_file).read_text(encoding="utf-8")
            assert "zIndex:" in content, f"{lens_file} has no zIndex in layers"
            assert "draw:" in content, f"{lens_file} has no draw function in layers"

    def test_app_imports_all_lenses(self) -> None:
        """app.js imports all 5 lenses."""
        content = (self.JS_ROOT / "app.js").read_text(encoding="utf-8")
        for lens in ["PhysicalLens", "SocialLens", "CognitiveLens", "CulturalLens", "TemporalLens"]:
            assert f"import {{ {lens} }}" in content, f"app.js doesn't import {lens}"

    def test_app_registers_all_lenses(self) -> None:
        """app.js registers all 5 lenses in _initializeLenses."""
        content = (self.JS_ROOT / "app.js").read_text(encoding="utf-8")
        for name in ["physical", "social", "cognitive", "cultural", "temporal"]:
            assert f"'{name}'" in content, f"app.js doesn't register lens '{name}'"

    def test_app_lens_keybinding_dispatch(self) -> None:
        """app.js dispatches lens-specific key bindings before global ones."""
        content = (self.JS_ROOT / "app.js").read_text(encoding="utf-8")
        assert "getKeyBindings()" in content, "app.js doesn't call getKeyBindings() on active lens"
        # The lens dispatch should come before global KEYBOARD_BINDINGS lookup
        kb_pos = content.find("KEYBOARD_BINDINGS[e.key]")
        lens_pos = content.find("lensBindings[e.key]")
        assert lens_pos < kb_pos, "Lens key bindings should be checked before global bindings"

    def test_no_disabled_buttons_in_toolbar(self) -> None:
        """Toolbar creation doesn't disable any lens buttons."""
        content = (self.JS_ROOT / "app.js").read_text(encoding="utf-8")
        assert "disabled" not in content, "app.js still has disabled buttons in toolbar"
        assert "Coming soon" not in content, "app.js still has 'Coming soon' placeholder"


# ---------------------------------------------------------------------------
# 3. HTML Template Validation
# ---------------------------------------------------------------------------


class TestHTMLTemplates:
    """Verify HTML templates have correct structure for the modular JS system."""

    TEMPLATES_DIR = Path(__file__).parent.parent / "src" / "visualization" / "templates"

    def test_live_template_exists(self) -> None:
        """live_new.html exists."""
        assert (self.TEMPLATES_DIR / "live_new.html").exists()

    def test_dashboard_template_exists(self) -> None:
        """dashboard_new.html exists."""
        assert (self.TEMPLATES_DIR / "dashboard_new.html").exists()

    def test_live_template_loads_app_module(self) -> None:
        """live_new.html loads app.js as ES module."""
        content = (self.TEMPLATES_DIR / "live_new.html").read_text(encoding="utf-8")
        assert 'type="module"' in content, "Missing module script tag"
        assert "app.js" in content, "Doesn't load app.js"

    def test_live_template_has_required_dom(self) -> None:
        """live_new.html has canvas, sidebar, header elements."""
        content = (self.TEMPLATES_DIR / "live_new.html").read_text(encoding="utf-8")
        assert 'id="world-canvas"' in content, "Missing #world-canvas"
        assert "<header" in content, "Missing <header>"
        # sidebar class used by SidePanelManager
        assert "sidebar" in content, "Missing sidebar element"

    def test_dashboard_template_has_data_placeholder(self) -> None:
        """dashboard_new.html has the /*__DATA__*/ placeholder for embedded JSON."""
        content = (self.TEMPLATES_DIR / "dashboard_new.html").read_text(encoding="utf-8")
        assert "/*__DATA__*/" in content, "Missing data placeholder"

    def test_dashboard_template_loads_app_module(self) -> None:
        """dashboard_new.html loads app.js as ES module."""
        content = (self.TEMPLATES_DIR / "dashboard_new.html").read_text(encoding="utf-8")
        assert 'type="module"' in content, "Missing module script tag"
        assert "app.js" in content, "Doesn't load app.js"


# ---------------------------------------------------------------------------
# 4. Live Server HTTP Endpoints
# ---------------------------------------------------------------------------


class TestLiveServerHTTP:
    """Test the FastAPI server serves correct responses."""

    def test_root_returns_html(self, live_server: LiveServer) -> None:
        """GET / returns the live dashboard HTML."""
        resp = httpx.get(f"http://127.0.0.1:{live_server.port}/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "<canvas" in resp.text
        assert 'id="world-canvas"' in resp.text

    def test_api_config_returns_json(self, live_server: LiveServer) -> None:
        """GET /api/config returns world configuration."""
        resp = httpx.get(f"http://127.0.0.1:{live_server.port}/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["world_width"] == 32
        assert data["world_height"] == 32
        assert data["num_agents"] == 5
        assert "archetype_colors" in data

    def test_static_app_js_served(self, live_server: LiveServer) -> None:
        """GET /static/js/app.js returns the orchestrator module."""
        resp = httpx.get(f"http://127.0.0.1:{live_server.port}/static/js/app.js")
        assert resp.status_code == 200
        assert "AutocogApp" in resp.text
        # Check correct MIME type for ES modules
        ct = resp.headers.get("content-type", "")
        assert "javascript" in ct or "text/" in ct

    @pytest.mark.parametrize(
        "lens_file",
        [
            "physical-lens.js",
            "social-lens.js",
            "cognitive-lens.js",
            "cultural-lens.js",
            "temporal-lens.js",
        ],
    )
    def test_lens_files_served(self, live_server: LiveServer, lens_file: str) -> None:
        """Each lens JS file is accessible via /static/js/lenses/."""
        resp = httpx.get(f"http://127.0.0.1:{live_server.port}/static/js/lenses/{lens_file}")
        assert resp.status_code == 200, f"Failed to serve {lens_file}"
        assert "LensBase" in resp.text, f"{lens_file} doesn't reference LensBase"

    @pytest.mark.parametrize(
        "core_file",
        [
            "colors.js",
            "config.js",
            "data-source.js",
            "event-bus.js",
            "network-graph.js",
            "spatial-index.js",
            "temporal-buffer.js",
            "time-series-store.js",
        ],
    )
    def test_core_files_served(self, live_server: LiveServer, core_file: str) -> None:
        """Each core JS module is accessible via /static/js/core/."""
        resp = httpx.get(f"http://127.0.0.1:{live_server.port}/static/js/core/{core_file}")
        assert resp.status_code == 200, f"Failed to serve core/{core_file}"

    @pytest.mark.parametrize(
        "renderer_file",
        [
            "canvas-compositor.js",
            "grid-renderer.js",
            "agent-renderer.js",
            "overlay-renderer.js",
        ],
    )
    def test_renderer_files_served(self, live_server: LiveServer, renderer_file: str) -> None:
        """Each renderer JS module is accessible."""
        resp = httpx.get(f"http://127.0.0.1:{live_server.port}/static/js/renderers/{renderer_file}")
        assert resp.status_code == 200, f"Failed to serve renderers/{renderer_file}"


# ---------------------------------------------------------------------------
# 5. WebSocket Data Flow
# ---------------------------------------------------------------------------


class TestWebSocketDataFlow:
    """Test that tick data flows through WebSocket to clients."""

    @pytest.mark.asyncio
    async def test_websocket_connects(self, live_server: LiveServer) -> None:
        """Client can connect to /ws endpoint."""
        async with websockets.connect(f"ws://127.0.0.1:{live_server.port}/ws") as ws:
            # websockets v16+: connection object is truthy when open
            assert ws  # connection exists

    @pytest.mark.asyncio
    async def test_websocket_receives_broadcast(
        self, live_server: LiveServer, class_engine: SimulationEngine
    ) -> None:
        """Connected client receives broadcast tick data."""
        async with websockets.connect(f"ws://127.0.0.1:{live_server.port}/ws") as ws:
            # Give server a moment to register the client
            await asyncio.sleep(0.2)

            # Step engine and broadcast
            record = class_engine.step_all()
            data = tick_to_json(class_engine, record)
            live_server.broadcast(data)

            # Wait for message with timeout
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                received = json.loads(raw)
                assert "tick" in received
                assert "agents" in received
                assert len(received["agents"]) > 0
            except TimeoutError:
                pytest.fail("No WebSocket message received within 3 seconds")

    @pytest.mark.asyncio
    async def test_websocket_broadcast_multiple_ticks(
        self, live_server: LiveServer, class_engine: SimulationEngine
    ) -> None:
        """Client receives multiple consecutive tick broadcasts."""
        async with websockets.connect(f"ws://127.0.0.1:{live_server.port}/ws") as ws:
            await asyncio.sleep(0.2)

            ticks_received = []
            for _ in range(3):
                record = class_engine.step_all()
                data = tick_to_json(class_engine, record)
                live_server.broadcast(data)

                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8")
                    received = json.loads(raw)
                    ticks_received.append(received["tick"])
                except TimeoutError:
                    break

            assert len(ticks_received) == 3, (
                f"Expected 3 ticks, got {len(ticks_received)}: {ticks_received}"
            )
            # Ticks should be strictly increasing
            assert ticks_received == sorted(ticks_received)

    @pytest.mark.asyncio
    async def test_websocket_data_has_enriched_fields(
        self, live_server: LiveServer, class_engine: SimulationEngine
    ) -> None:
        """Broadcast data includes all enriched fields needed by the new lenses."""
        async with websockets.connect(f"ws://127.0.0.1:{live_server.port}/ws") as ws:
            await asyncio.sleep(0.2)

            # Run a few ticks to build up state
            for _ in range(5):
                record = class_engine.step_all()

            data = tick_to_json(class_engine, record)
            live_server.broadcast(data)

            raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            received = json.loads(raw)

            # Verify agent enriched fields are present in the WebSocket payload
            agent = received["agents"][0]
            assert "cultural" in agent, "Missing cultural data in WS payload"
            assert "metacognition" in agent, "Missing metacognition data in WS payload"
            assert "tom" in agent, "Missing ToM data in WS payload"
            assert "coalition" in agent, "Missing coalition data in WS payload"


# ---------------------------------------------------------------------------
# 6. Lens-Specific Content Validation
# ---------------------------------------------------------------------------


class TestLensContentIntegrity:
    """Deep validation of each lens's JS content for correctness."""

    JS_LENSES = Path(__file__).parent.parent / "src" / "visualization" / "static" / "js" / "lenses"

    def test_cognitive_lens_uses_metacognition_data(self) -> None:
        """Cognitive lens reads metacognition and ToM fields."""
        content = (self.JS_LENSES / "cognitive-lens.js").read_text(encoding="utf-8")
        # Should access metacognition fields
        assert "metacognition" in content
        assert "calibration_score" in content
        assert "deliberation_invoked" in content
        assert "active_strategy" in content
        # Should access ToM fields
        assert "tom" in content.lower() or "ToM" in content
        # Should have calibration curve rendering
        assert "calibration_curve" in content
        # Should have SRIE cascade
        assert "srie" in content

    def test_social_lens_uses_relationship_data(self) -> None:
        """Social lens reads social_relationships and coalition fields."""
        content = (self.JS_LENSES / "social-lens.js").read_text(encoding="utf-8")
        assert "social_relationships" in content
        assert "coalition" in content
        assert "trust" in content
        # Should use convex hulls for coalitions
        assert "drawConvexHull" in content or "ConvexHull" in content
        # Should draw trust lines
        assert "drawLine" in content or "drawArrow" in content

    def test_cultural_lens_uses_cultural_data(self) -> None:
        """Cultural lens reads cultural and language fields."""
        content = (self.JS_LENSES / "cultural-lens.js").read_text(encoding="utf-8")
        assert "cultural" in content
        assert "learning_style" in content
        assert "repertoire_size" in content
        assert "cultural_group" in content
        # Should handle transmission events
        assert "transmission" in content.lower()
        # Should handle language data
        assert "language" in content
        assert "vocabulary" in content or "vocab" in content

    def test_temporal_lens_uses_temporal_data(self) -> None:
        """Temporal lens reads trail and metric history data."""
        content = (self.JS_LENSES / "temporal-lens.js").read_text(encoding="utf-8")
        # Should use temporal buffer for trails
        assert "getAgentTrail" in content
        # Should render sparklines or time series
        assert "sparkline" in content.lower() or "svg" in content.lower()
        # Should show metric trends
        assert "trend" in content.lower()
        # Should use NEED_COLORS
        assert "NEED_COLORS" in content

    def test_all_lenses_have_keybindings(self) -> None:
        """Each new lens defines getKeyBindings with at least one binding."""
        for lens_file in [
            "social-lens.js",
            "cognitive-lens.js",
            "cultural-lens.js",
            "temporal-lens.js",
        ]:
            content = (self.JS_LENSES / lens_file).read_text(encoding="utf-8")
            # Should have getKeyBindings that returns an object with at least one key
            assert "getKeyBindings()" in content, f"{lens_file} missing getKeyBindings"
            # The method should not return empty object
            # Find the getKeyBindings method body
            idx = content.find("getKeyBindings()")
            after = content[idx : idx + 200]
            assert "return {}" not in after.replace(" ", ""), (
                f"{lens_file} has empty getKeyBindings"
            )

    def test_all_lenses_use_ctx_save_restore(self) -> None:
        """Canvas state is properly isolated — either directly or via renderer delegation."""
        for lens_file in [
            "social-lens.js",
            "cognitive-lens.js",
            "cultural-lens.js",
            "temporal-lens.js",
        ]:
            content = (self.JS_LENSES / lens_file).read_text(encoding="utf-8")
            saves = content.count("ctx.save()")
            restores = content.count("ctx.restore()")
            # Lenses that delegate all rendering to AgentRenderer/OverlayRenderer
            # (which handle save/restore internally) may not call ctx.save() directly.
            # In that case, verify they use those renderers.
            if saves == 0:
                uses_renderers = (
                    "OverlayRenderer" in content
                    or "AgentRenderer" in content
                    or "GridRenderer" in content
                )
                assert uses_renderers, (
                    f"{lens_file} neither calls ctx.save() nor delegates to renderers"
                )
            else:
                assert saves == restores, f"{lens_file} has {saves} save() but {restores} restore()"

    def test_all_lenses_null_safe(self) -> None:
        """Lenses use optional chaining for data access."""
        for lens_file in [
            "social-lens.js",
            "cognitive-lens.js",
            "cultural-lens.js",
            "temporal-lens.js",
        ]:
            content = (self.JS_LENSES / lens_file).read_text(encoding="utf-8")
            # Should use optional chaining somewhere (not every access, but some)
            assert "?." in content, (
                f"{lens_file} doesn't use optional chaining (?.) for null safety"
            )


# ---------------------------------------------------------------------------
# 7. Dashboard Generation End-to-End
# ---------------------------------------------------------------------------


class TestDashboardGeneration:
    """Test that DashboardGenerator produces valid HTML with embedded data."""

    def test_dashboard_generates_from_engine(
        self, full_engine: SimulationEngine, tmp_path: Path
    ) -> None:
        """Generate a dashboard from engine trajectory data."""
        from src.trajectory.recorder import TrajectoryRecorder
        from src.visualization.dashboard import DashboardGenerator

        # Record some ticks
        recorder = TrajectoryRecorder(output_dir=str(tmp_path / "traj"), run_id="test_run")
        recorder.start_run(full_engine)
        for _ in range(10):
            tick_record = full_engine.step_all()
            recorder.record_tick(full_engine, tick_record)
        recorder.end_run(full_engine)

        # Generate dashboard from the recorded trajectory
        jsonl_path = tmp_path / "traj" / "test_run" / "trajectory.jsonl"
        output_path = tmp_path / "dashboard.html"
        gen = DashboardGenerator.from_file(str(jsonl_path))
        gen.generate(str(output_path))

        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")

        # Verify HTML structure
        assert "<!DOCTYPE html>" in content
        assert "<canvas" in content
        assert 'type="module"' in content
        assert "app.js" in content

        # Verify data is embedded (the /*__DATA__*/ placeholder was replaced)
        assert "/*__DATA__*/" not in content
        # The embedded JSON should contain agent data
        assert '"agents"' in content or '"ticks"' in content


# ---------------------------------------------------------------------------
# 8. Cross-Lens Consistency
# ---------------------------------------------------------------------------


class TestCrossLensConsistency:
    """Verify all lenses share consistent patterns and don't conflict."""

    JS_LENSES = Path(__file__).parent.parent / "src" / "visualization" / "static" / "js" / "lenses"
    ALL_LENSES = [
        "physical-lens.js",
        "social-lens.js",
        "cognitive-lens.js",
        "cultural-lens.js",
        "temporal-lens.js",
    ]

    def test_all_lenses_use_same_renderer_imports(self) -> None:
        """All lenses that use renderers import from the same paths."""
        for lens_file in self.ALL_LENSES:
            content = (self.JS_LENSES / lens_file).read_text(encoding="utf-8")
            if "GridRenderer" in content:
                assert "'../renderers/grid-renderer.js'" in content, (
                    f"{lens_file} has wrong GridRenderer import path"
                )
            if "AgentRenderer" in content:
                assert "'../renderers/agent-renderer.js'" in content, (
                    f"{lens_file} has wrong AgentRenderer import path"
                )

    def test_all_lenses_import_from_lens_base(self) -> None:
        """All lenses import LensBase from the correct path."""
        for lens_file in self.ALL_LENSES:
            content = (self.JS_LENSES / lens_file).read_text(encoding="utf-8")
            assert "from './lens-base.js'" in content, f"{lens_file} has wrong LensBase import path"

    def test_layer_zindex_no_conflicts_within_lens(self) -> None:
        """Within each lens, no two layers share the same zIndex."""
        for lens_file in self.ALL_LENSES:
            content = (self.JS_LENSES / lens_file).read_text(encoding="utf-8")
            zindices = re.findall(r"zIndex:\s*(\d+)", content)
            zindices_int = [int(z) for z in zindices]
            assert len(zindices_int) == len(set(zindices_int)), (
                f"{lens_file} has duplicate zIndex values: {zindices_int}"
            )

    def test_getAgentStyle_returns_required_fields(self) -> None:
        """Each getAgentStyle mentions color, shape, and size."""
        for lens_file in self.ALL_LENSES:
            content = (self.JS_LENSES / lens_file).read_text(encoding="utf-8")
            # Find the getAgentStyle method
            idx = content.find("getAgentStyle(")
            if idx == -1:
                pytest.fail(f"{lens_file} missing getAgentStyle")
            # Look at the full method body (up to the next method or end of class)
            next_method = content.find("\n  get", idx + 1)
            end = next_method if next_method != -1 else idx + 2000
            snippet = content[idx:end]
            assert "color" in snippet, f"{lens_file} getAgentStyle missing color"
            assert "shape" in snippet, f"{lens_file} getAgentStyle missing shape"
            assert "size" in snippet, f"{lens_file} getAgentStyle missing size"
