"""Minimal validation that CLI structure is correct."""

import sys
from io import StringIO

# Test that help can be generated without errors
try:
    from src.experiments.__main__ import main

    # Capture help output
    old_argv = sys.argv
    old_stdout = sys.stdout

    try:
        sys.argv = ["experiments", "benchmark", "--help"]
        sys.stdout = StringIO()

        try:
            main()
        except SystemExit as e:
            # --help causes sys.exit(0), which is expected
            if e.code != 0:
                raise

        help_output = sys.stdout.getvalue()

        # Check that key options are in help
        assert "--suite" in help_output, "Missing --suite option in help"
        assert "--scenario" in help_output, "Missing --scenario option in help"
        assert "--architectures" in help_output, "Missing --architectures option in help"
        assert "--report" in help_output, "Missing --report option in help"
        assert "--push-hf" in help_output, "Missing --push-hf option in help"

        print("✓ CLI help output is correct")
        print("\nHelp output:")
        print(help_output)

    finally:
        sys.argv = old_argv
        sys.stdout = old_stdout

except Exception as e:
    print(f"✗ CLI help generation failed: {e}")
    sys.exit(1)
