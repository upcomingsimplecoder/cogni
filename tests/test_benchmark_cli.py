"""Quick validation test for benchmark CLI."""

import sys
from pathlib import Path

# Test imports
try:
    from src.experiments.analysis import ResultAnalyzer  # noqa: F401
    from src.experiments.benchmark import BenchmarkRunner
    from src.experiments.hf_push import HuggingFacePublisher
    from src.experiments.report import ReportGenerator

    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test BenchmarkRunner initialization
try:
    runner = BenchmarkRunner(suite_dir="experiments/benchmarks")
    print("✓ BenchmarkRunner initialized")
except Exception as e:
    print(f"✗ BenchmarkRunner initialization failed: {e}")
    sys.exit(1)

# Test that benchmark YAML files exist
suite_dir = Path("experiments/benchmarks")
yaml_files = list(suite_dir.glob("*.yaml"))
print(f"✓ Found {len(yaml_files)} benchmark scenarios")

# Test ReportGenerator has new method
if hasattr(ReportGenerator, "to_comparison_markdown"):
    print("✓ ReportGenerator.to_comparison_markdown exists")
else:
    print("✗ ReportGenerator.to_comparison_markdown not found")
    sys.exit(1)

# Test HuggingFacePublisher (without actual HF hub)
try:
    # This will fail if huggingface_hub not installed, but that's expected
    publisher = HuggingFacePublisher(repo_id="test/test")
    print("✓ HuggingFacePublisher initialized (HF hub available)")
except ImportError as e:
    if "huggingface_hub" in str(e):
        print("✓ HuggingFacePublisher raises correct ImportError (HF hub not installed)")
    else:
        print(f"✗ Unexpected import error: {e}")
        sys.exit(1)

print("\n" + "=" * 70)
print("All validation checks passed!")
print("=" * 70)
