"""Tests for the real-time visualization server and data serialization."""

from __future__ import annotations

import json

import pytest

from src.config import SimulationConfig
from src.simulation.engine import SimulationEngine
from src.visualization.realtime import (
    ARCHETYPE_COLORS,
    LiveServer,
    tick_to_json,
)


@pytest.fixture
def engine() -> SimulationEngine:
    """Create an engine with agents for testing."""
    config = SimulationConfig()
    config.num_agents = 3
    config.max_ticks = 10
    config.seed = 42
    engine = SimulationEngine(config)
    engine.setup_multi_agent()
    return engine


class TestTickToJson:
    """Tests for the tick_to_json serialization function."""

    def test_basic_structure(self, engine: SimulationEngine) -> None:
        """tick_to_json returns dict with all required top-level keys."""
        tick_record = engine.step_all()
        data = tick_to_json(engine, tick_record)

        assert "tick" in data
        assert "day" in data
        assert "time_of_day" in data
        assert "agents" in data
        assert "emergent_events" in data
        assert "living_count" in data
        assert "dead_count" in data
        assert "messages" in data

    def test_tick_number_matches(self, engine: SimulationEngine) -> None:
        """Tick number in output matches the tick record."""
        tick_record = engine.step_all()
        data = tick_to_json(engine, tick_record)
        assert data["tick"] == tick_record.tick

    def test_agent_count(self, engine: SimulationEngine) -> None:
        """All agents appear in the output."""
        tick_record = engine.step_all()
        data = tick_to_json(engine, tick_record)
        assert len(data["agents"]) == 3
        assert data["living_count"] == 3
        assert data["dead_count"] == 0

    def test_agent_fields(self, engine: SimulationEngine) -> None:
        """Each agent has all required fields."""
        tick_record = engine.step_all()
        data = tick_to_json(engine, tick_record)
        agent = data["agents"][0]

        required_fields = [
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
            "monologue",
            "inventory",
            "traits",
        ]
        for field in required_fields:
            assert field in agent, f"Missing field: {field}"

    def test_agent_color_from_archetype(self, engine: SimulationEngine) -> None:
        """Agent colors match their archetype."""
        tick_record = engine.step_all()
        data = tick_to_json(engine, tick_record)

        for agent in data["agents"]:
            expected_color = ARCHETYPE_COLORS.get(agent["archetype"], "#f8fafc")
            assert agent["color"] == expected_color

    def test_needs_are_numeric(self, engine: SimulationEngine) -> None:
        """Need values are rounded floats."""
        tick_record = engine.step_all()
        data = tick_to_json(engine, tick_record)

        for agent in data["agents"]:
            for need in ("hunger", "thirst", "energy", "health"):
                assert isinstance(agent[need], float)
                # Verify rounded to 1 decimal
                assert agent[need] == round(agent[need], 1)

    def test_traits_dict(self, engine: SimulationEngine) -> None:
        """Traits are serialized as a dict of floats."""
        tick_record = engine.step_all()
        data = tick_to_json(engine, tick_record)

        for agent in data["agents"]:
            assert isinstance(agent["traits"], dict)
            if agent["traits"]:
                for key, value in agent["traits"].items():
                    assert isinstance(key, str)
                    assert isinstance(value, (int, float))

    def test_action_type_is_string(self, engine: SimulationEngine) -> None:
        """Action type is serialized as a string value."""
        tick_record = engine.step_all()
        data = tick_to_json(engine, tick_record)

        for agent in data["agents"]:
            assert isinstance(agent["action_type"], str)
            assert agent["action_type"] != ""

    def test_intention_is_string(self, engine: SimulationEngine) -> None:
        """Intention primary_goal is a string."""
        tick_record = engine.step_all()
        data = tick_to_json(engine, tick_record)

        for agent in data["agents"]:
            assert isinstance(agent["intention"], str)

    def test_json_serializable(self, engine: SimulationEngine) -> None:
        """Output is fully JSON-serializable (no dataclasses, no custom objects)."""
        tick_record = engine.step_all()
        data = tick_to_json(engine, tick_record)

        # This will raise TypeError if anything isn't serializable
        json_str = json.dumps(data)
        assert len(json_str) > 0

        # Round-trip: deserialize and verify structure
        parsed = json.loads(json_str)
        assert parsed["tick"] == data["tick"]
        assert len(parsed["agents"]) == len(data["agents"])

    def test_multiple_ticks(self, engine: SimulationEngine) -> None:
        """Serialization works across multiple ticks."""
        for _ in range(5):
            tick_record = engine.step_all()
            data = tick_to_json(engine, tick_record)
            # Should be valid each tick
            json.dumps(data)

        # Last tick should be 5
        assert data["tick"] == 5

    def test_messages_format(self, engine: SimulationEngine) -> None:
        """Messages list contains properly structured dicts (or is empty)."""
        # Run a few ticks — messages may or may not appear
        for _ in range(5):
            tick_record = engine.step_all()
        data = tick_to_json(engine, tick_record)

        assert isinstance(data["messages"], list)
        for msg in data["messages"]:
            assert "id" in msg
            assert "sender" in msg
            assert "type" in msg
            assert "content" in msg

    def test_emergent_events_are_strings(self, engine: SimulationEngine) -> None:
        """Emergent events are a list of strings."""
        tick_record = engine.step_all()
        data = tick_to_json(engine, tick_record)

        assert isinstance(data["emergent_events"], list)
        for event in data["emergent_events"]:
            assert isinstance(event, str)


class TestLiveServer:
    """Tests for the LiveServer class."""

    @pytest.fixture(autouse=True)
    def _require_fastapi(self) -> None:
        pytest.importorskip("fastapi")

    def test_server_creation(self) -> None:
        """LiveServer can be instantiated."""
        server = LiveServer(port=9999)
        assert server.port == 9999
        assert server.engine is None

    def test_set_engine(self) -> None:
        """set_engine stores the engine reference."""
        server = LiveServer(port=9999)
        engine = SimulationEngine()
        server.set_engine(engine)
        assert server.engine is engine

    def test_broadcast_no_clients(self) -> None:
        """broadcast() doesn't error when no clients connected."""
        server = LiveServer(port=9999)
        # Should not raise even with no clients
        server.broadcast({"tick": 1, "agents": []})

    def test_archetype_colors_complete(self) -> None:
        """All archetypes have color mappings."""
        expected = {"gatherer", "explorer", "diplomat", "aggressor", "survivalist"}
        assert set(ARCHETYPE_COLORS.keys()) == expected

    def test_archetype_colors_are_hex(self) -> None:
        """All colors are valid hex strings."""
        for color in ARCHETYPE_COLORS.values():
            assert color.startswith("#")
            assert len(color) == 7  # #RRGGBB


class TestLiveHtmlTemplate:
    """Tests that the live dashboard HTML template exists and is valid."""

    def test_template_exists(self) -> None:
        """live.html template file exists."""
        from pathlib import Path

        template_path = (
            Path(__file__).parent.parent / "src" / "visualization" / "templates" / "live.html"
        )
        assert template_path.exists(), f"Template not found at {template_path}"

    def test_template_has_websocket(self) -> None:
        """Template contains WebSocket client code."""
        from pathlib import Path

        template_path = (
            Path(__file__).parent.parent / "src" / "visualization" / "templates" / "live.html"
        )
        content = template_path.read_text(encoding="utf-8")

        assert "WebSocket" in content
        assert "/ws" in content
        assert "onmessage" in content

    def test_template_has_canvas(self) -> None:
        """Template contains canvas rendering code."""
        from pathlib import Path

        template_path = (
            Path(__file__).parent.parent / "src" / "visualization" / "templates" / "live.html"
        )
        content = template_path.read_text(encoding="utf-8")

        assert "<canvas" in content
        assert "getContext" in content

    def test_template_has_inspector(self) -> None:
        """Template contains agent inspector UI."""
        from pathlib import Path

        template_path = (
            Path(__file__).parent.parent / "src" / "visualization" / "templates" / "live.html"
        )
        content = template_path.read_text(encoding="utf-8")

        assert "inspector" in content.lower() or "Inspector" in content

    def test_template_has_auto_reconnect(self) -> None:
        """Template contains auto-reconnect logic."""
        from pathlib import Path

        template_path = (
            Path(__file__).parent.parent / "src" / "visualization" / "templates" / "live.html"
        )
        content = template_path.read_text(encoding="utf-8")

        assert "reconnect" in content.lower() or "setTimeout" in content


class TestMainCLIIntegration:
    """Tests that --live flag is properly handled in main.py."""

    def test_live_flag_parsed(self) -> None:
        """--live flag sets live_mode in main."""
        # We test by importing and checking the arg parsing logic
        # The flag parsing is in main(), which reads sys.argv.
        # We can verify the help text includes --live
        import inspect

        import src.main as main_module

        source = inspect.getsource(main_module.main)
        assert "--live" in source

    def test_live_port_flag_parsed(self) -> None:
        """--live-port flag is handled."""
        import inspect

        import src.main as main_module

        source = inspect.getsource(main_module.main)
        assert "--live-port=" in source
