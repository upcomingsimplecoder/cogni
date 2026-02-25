"""Tests for plugin system (loader and registry)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.awareness.types import Expression, Intention, Reflection
from src.plugins.loader import PluginLoader
from src.plugins.registry import PluginRegistry
from src.simulation.actions import Action, ActionResult, ActionType


# Test fixtures
@pytest.fixture(autouse=True)
def clear_registry():
    """Clear plugin registry before and after each test."""
    PluginRegistry.clear()
    yield
    PluginRegistry.clear()


@pytest.fixture
def temp_plugin_dir(tmp_path):
    """Create temporary plugin directory."""
    plugin_dir = tmp_path / "test_plugins"
    plugin_dir.mkdir()
    return plugin_dir


# Registry Tests
def test_register_strategy():
    """Test strategy registration decorator."""

    @PluginRegistry.register_strategy("test_strategy")
    class TestStrategy:
        def form_intention(self, sensation, reflection):
            return Intention(primary_goal="test")

        def express(self, sensation, reflection, intention):
            return Expression(action=Action(type=ActionType.WAIT))

    assert PluginRegistry.get_strategy("test_strategy") == TestStrategy
    assert "test_strategy" in PluginRegistry.all_strategies()


def test_register_action():
    """Test action handler registration."""

    @PluginRegistry.register_action("test_action")
    def test_action_handler(action, agent, world, registry):
        return ActionResult(action=action, success=True, message="Test")

    handler = PluginRegistry.get_action_handler("test_action")
    assert handler is not None
    assert callable(handler)


def test_register_archetype():
    """Test archetype registration."""
    PluginRegistry.register_archetype(
        name="test_archetype",
        traits={"cooperation": 0.5, "curiosity": 0.7},
        color="blue",
        symbol="T",
    )

    archetype = PluginRegistry.get_archetype("test_archetype")
    assert archetype is not None
    assert archetype["traits"]["cooperation"] == 0.5
    assert archetype["color"] == "blue"
    assert archetype["symbol"] == "T"


def test_register_evaluation():
    """Test evaluation strategy registration."""

    @PluginRegistry.register_evaluation("test_eval")
    class TestEvaluation:
        def evaluate(self, agent, engine, sensation):
            return Reflection(
                last_action_succeeded=True,
                need_trends={},
                threat_level=0.0,
                opportunity_score=0.5,
            )

    eval_cls = PluginRegistry.get_evaluation("test_eval")
    assert eval_cls == TestEvaluation


def test_register_metric():
    """Test metric registration."""

    @PluginRegistry.register_metric("test_metric")
    def test_metric(engine):
        return 42.0

    metric = PluginRegistry.get_metric("test_metric")
    assert metric is not None
    assert metric(None) == 42.0


def test_registry_override_warning():
    """Test that re-registering logs warning but works."""

    @PluginRegistry.register_strategy("override_test")
    class Strategy1:
        pass

    @PluginRegistry.register_strategy("override_test")
    class Strategy2:
        pass

    # Second registration should override
    assert PluginRegistry.get_strategy("override_test") == Strategy2


def test_registry_get_nonexistent():
    """Test getting non-existent registrations returns None."""
    assert PluginRegistry.get_strategy("nonexistent") is None
    assert PluginRegistry.get_action_handler("nonexistent") is None
    assert PluginRegistry.get_archetype("nonexistent") is None
    assert PluginRegistry.get_evaluation("nonexistent") is None
    assert PluginRegistry.get_metric("nonexistent") is None


def test_registry_clear():
    """Test clearing all registrations."""

    @PluginRegistry.register_strategy("test")
    class TestStrategy:
        pass

    assert len(PluginRegistry.all_strategies()) > 0

    PluginRegistry.clear()

    assert len(PluginRegistry.all_strategies()) == 0
    assert len(PluginRegistry._actions) == 0
    assert len(PluginRegistry._archetypes) == 0


# Loader Tests
def test_load_plugin_file_simple(temp_plugin_dir):
    """Test loading a simple plugin file."""
    plugin_file = temp_plugin_dir / "simple_plugin.py"
    plugin_file.write_text(
        """
from src.plugins import PluginRegistry

@PluginRegistry.register_strategy("simple_strategy")
class SimpleStrategy:
    def form_intention(self, sensation, reflection):
        pass
    def express(self, sensation, reflection, intention):
        pass
"""
    )

    PluginLoader.load_plugin_file(str(plugin_file))

    assert PluginRegistry.get_strategy("simple_strategy") is not None


def test_load_plugin_file_not_found():
    """Test loading non-existent file raises error."""
    with pytest.raises(FileNotFoundError):
        PluginLoader.load_plugin_file("/nonexistent/plugin.py")


def test_load_plugin_file_invalid_extension(temp_plugin_dir):
    """Test loading non-Python file raises error."""
    txt_file = temp_plugin_dir / "not_python.txt"
    txt_file.write_text("not python code")

    with pytest.raises(ValueError, match="Not a Python file"):
        PluginLoader.load_plugin_file(str(txt_file))


def test_load_from_directory(temp_plugin_dir):
    """Test loading all plugins from directory."""
    # Create multiple plugin files
    (temp_plugin_dir / "plugin1.py").write_text(
        """
from src.plugins import PluginRegistry

@PluginRegistry.register_strategy("strategy1")
class Strategy1:
    def form_intention(self, s, r): pass
    def express(self, s, r, i): pass
"""
    )

    (temp_plugin_dir / "plugin2.py").write_text(
        """
from src.plugins import PluginRegistry

@PluginRegistry.register_strategy("strategy2")
class Strategy2:
    def form_intention(self, s, r): pass
    def express(self, s, r, i): pass

@PluginRegistry.register_action("action2")
def action2_handler(action, agent, world, registry):
    pass
"""
    )

    # Create a file that should be ignored
    (temp_plugin_dir / "__init__.py").write_text("")
    (temp_plugin_dir / "_private.py").write_text("")

    stats = PluginLoader.load_from_directory(str(temp_plugin_dir))

    assert stats["strategies"] == 2
    assert stats["actions"] == 1
    assert len(stats["errors"]) == 0
    assert PluginRegistry.get_strategy("strategy1") is not None
    assert PluginRegistry.get_strategy("strategy2") is not None
    assert PluginRegistry.get_action_handler("action2") is not None


def test_load_from_directory_with_errors(temp_plugin_dir):
    """Test that errors in one plugin don't prevent loading others."""
    # Good plugin
    (temp_plugin_dir / "good.py").write_text(
        """
from src.plugins import PluginRegistry

@PluginRegistry.register_strategy("good")
class GoodStrategy:
    def form_intention(self, s, r): pass
    def express(self, s, r, i): pass
"""
    )

    # Bad plugin with syntax error
    (temp_plugin_dir / "bad.py").write_text(
        """
from src.plugins import PluginRegistry

@PluginRegistry.register_strategy("bad")
class BadStrategy:
    def form_intention(self, s, r)  # Missing colon - syntax error
        pass
"""
    )

    stats = PluginLoader.load_from_directory(str(temp_plugin_dir))

    # Good plugin should load
    assert stats["strategies"] >= 1
    assert PluginRegistry.get_strategy("good") is not None

    # Error should be recorded
    assert len(stats["errors"]) == 1
    assert "bad.py" in stats["errors"][0]


def test_load_all_default_directory():
    """Test load_all with default plugins directory."""
    # This test assumes the plugins/ directory exists
    # It should not fail even if directory is empty or doesn't exist
    stats = PluginLoader.load_all()

    assert isinstance(stats, dict)
    assert "strategies" in stats
    assert "actions" in stats
    assert "errors" in stats


def test_load_all_custom_directories(temp_plugin_dir):
    """Test load_all with custom directory list."""
    plugin_dir1 = temp_plugin_dir / "dir1"
    plugin_dir2 = temp_plugin_dir / "dir2"
    plugin_dir1.mkdir()
    plugin_dir2.mkdir()

    (plugin_dir1 / "plugin1.py").write_text(
        """
from src.plugins import PluginRegistry

@PluginRegistry.register_strategy("strategy_dir1")
class Strategy1:
    def form_intention(self, s, r): pass
    def express(self, s, r, i): pass
"""
    )

    (plugin_dir2 / "plugin2.py").write_text(
        """
from src.plugins import PluginRegistry

@PluginRegistry.register_strategy("strategy_dir2")
class Strategy2:
    def form_intention(self, s, r): pass
    def express(self, s, r, i): pass
"""
    )

    stats = PluginLoader.load_all([str(plugin_dir1), str(plugin_dir2)])

    assert stats["strategies"] == 2
    assert PluginRegistry.get_strategy("strategy_dir1") is not None
    assert PluginRegistry.get_strategy("strategy_dir2") is not None


def test_load_all_nonexistent_directory():
    """Test load_all gracefully handles non-existent directories."""
    stats = PluginLoader.load_all(["/nonexistent/directory"])

    assert stats["strategies"] == 0
    assert stats["actions"] == 0
    # Should not crash, just log warning


def test_example_plugin_loads():
    """Test that the example plugin can be loaded."""
    # Try to load the actual example plugin if it exists
    example_path = Path("plugins/example_strategy.py")

    if example_path.exists():
        PluginRegistry.clear()
        PluginLoader.load_plugin_file(str(example_path))

        strategy = PluginRegistry.get_strategy("cautious_gatherer")
        assert strategy is not None

        # Test instantiation
        instance = strategy()
        assert hasattr(instance, "form_intention")
        assert hasattr(instance, "express")


def test_plugin_with_multiple_registrations(temp_plugin_dir):
    """Test plugin that registers multiple components."""
    plugin_file = temp_plugin_dir / "multi.py"
    plugin_file.write_text(
        """
from src.plugins import PluginRegistry

@PluginRegistry.register_strategy("multi_strategy")
class MultiStrategy:
    def form_intention(self, s, r): pass
    def express(self, s, r, i): pass

@PluginRegistry.register_action("multi_action")
def multi_action_handler(action, agent, world, registry):
    pass

PluginRegistry.register_archetype(
    name="multi_arch",
    traits={"test": 1.0},
    color="red",
    symbol="M"
)

@PluginRegistry.register_metric("multi_metric")
def multi_metric(engine):
    return 1.0
"""
    )

    stats = PluginLoader.load_from_directory(str(temp_plugin_dir))

    assert stats["strategies"] == 1
    assert stats["actions"] == 1
    assert stats["archetypes"] == 1
    assert stats["metrics"] == 1
    assert PluginRegistry.get_strategy("multi_strategy") is not None
    assert PluginRegistry.get_action_handler("multi_action") is not None
    assert PluginRegistry.get_archetype("multi_arch") is not None
    assert PluginRegistry.get_metric("multi_metric") is not None


def test_loader_stats_accumulation(temp_plugin_dir):
    """Test that loader correctly accumulates statistics."""
    (temp_plugin_dir / "p1.py").write_text(
        """
from src.plugins import PluginRegistry

@PluginRegistry.register_strategy("s1")
class S1:
    def form_intention(self, s, r): pass
    def express(self, s, r, i): pass
"""
    )

    (temp_plugin_dir / "p2.py").write_text(
        """
from src.plugins import PluginRegistry

@PluginRegistry.register_strategy("s2")
class S2:
    def form_intention(self, s, r): pass
    def express(self, s, r, i): pass

@PluginRegistry.register_action("a1")
def a1_handler(action, agent, world, registry): pass
"""
    )

    stats = PluginLoader.load_from_directory(str(temp_plugin_dir))

    assert stats["strategies"] == 2
    assert stats["actions"] == 1
    assert stats["archetypes"] == 0
    assert stats["evaluations"] == 0
    assert stats["metrics"] == 0
