# Changelog

All notable changes to cogniarch will be documented in this file.

## [0.1.1] - 2025-02-25

### Added
- `BENCHMARKS.md` with curated architecture rankings and key findings across 10 scenarios
- `examples/reproduce_paper.ipynb` tutorial notebook -- replicate ALIFE 2026 findings in 5 minutes
- `scripts/push_to_hf.py` for publishing datasets to HuggingFace
- Live dashboard documentation in README with all 5 lens descriptions and key bindings
- HuggingFace dataset link in README and BENCHMARKS.md

### Changed
- README updated with dashboard lenses section, tutorial link, benchmark link, HF dataset link
- Project structure in README updated to include visualization and Parquet export

## [0.1.0] - 2025-02-24

### Added
- Initial public release
- SRIE pipeline (Sensation, Reflection, Intention, Expression) with pluggable components
- 7 cognitive architectures: reactive, cautious, optimistic, social, dual_process, planning, metacognitive
- 7 decision strategies: Hardcoded, Personality, LLM, Planning, Theory of Mind, Metacognitive, Culturally Modulated
- 5 agent archetypes: Gatherer, Explorer, Diplomat, Aggressor, Survivalist
- Advanced subsystems: Theory of Mind, emergent language, cultural transmission, coalition formation, trait evolution
- Batch experiment runner with YAML-defined specs
- 25 pre-configured experiments (15 alignment studies + 10 benchmarks)
- Live WebSocket dashboard with 5 visualization lenses
- Trajectory recording and export (JSONL, CSV, Parquet)
- CLI entry point with comprehensive flag support
- 1,161 tests passing
