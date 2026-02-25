#!/usr/bin/env python3
"""Test script to verify Phase 1.1 setup."""

import sys
from pathlib import Path


def test_js_modules():
    """Test that all JS modules exist."""
    js_root = Path("src/visualization/static/js")

    expected_files = [
        "app.js",
        "core/colors.js",
        "core/config.js",
        "core/data-source.js",
        "core/event-bus.js",
        "core/network-graph.js",
        "core/spatial-index.js",
        "core/temporal-buffer.js",
        "core/time-series-store.js",
        "lenses/lens-base.js",
        "lenses/physical-lens.js",
        "panels/inspector-panel.js",
        "panels/side-panel-manager.js",
        "panels/timeline-panel.js",
        "renderers/agent-renderer.js",
        "renderers/canvas-compositor.js",
        "renderers/grid-renderer.js",
        "renderers/overlay-renderer.js",
        "utils/export.js",
    ]

    missing = []
    for file in expected_files:
        path = js_root / file
        if not path.exists():
            missing.append(str(path))

    if missing:
        print(f"❌ Missing {len(missing)} JS modules:")
        for m in missing:
            print(f"   - {m}")
        return False

    print(f"✅ All {len(expected_files)} JS modules present")
    return True


def test_templates():
    """Test that new templates exist."""
    templates = Path("src/visualization/templates")

    expected = ["live_new.html", "dashboard_new.html"]
    missing = []

    for template in expected:
        path = templates / template
        if not path.exists():
            missing.append(str(path))

    if missing:
        print(f"❌ Missing {len(missing)} templates:")
        for m in missing:
            print(f"   - {m}")
        return False

    print(f"✅ All {len(expected)} templates present")
    return True


def test_python_imports():
    """Test that Python modules import correctly."""
    try:
        # Test imports (no fastapi required)
        import src.visualization.dashboard  # noqa: F401
        import src.visualization.server  # noqa: F401

        print("✅ Python modules import correctly")
        return True
    except ImportError as e:
        print(f"❌ Python import failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing Phase 1.1 Setup...\n")

    tests = [
        ("JS Modules", test_js_modules),
        ("Templates", test_templates),
        ("Python Imports", test_python_imports),
    ]

    results = []
    for name, test_func in tests:
        print(f"\n{name}:")
        result = test_func()
        results.append(result)

    print("\n" + "=" * 50)
    if all(results):
        print("✅ All tests passed!")
        print("\nNext steps:")
        print("1. Run a live simulation to test WebSocket mode")
        print("2. Generate a dashboard to test replay mode")
        print("3. Verify Physical lens rendering matches existing behavior")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
