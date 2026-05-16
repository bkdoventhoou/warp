# Commit Classification Reference

This document defines how to classify commits when performing a release audit.

## Classification Categories

### `feat` — New Features
Commits that introduce new user-facing functionality.

**Indicators:**
- Adds a new API, function, or class
- Introduces a new kernel primitive or tile operation
- Adds support for a new hardware target or backend
- New example scripts or tutorials

**Examples:**
- `feat: add warp.sparse.bsr_matrix_from_triplets`
- `feat: support CUDA graphs in launch context`
- `add tile_reduce with custom operator support`

---

### `fix` — Bug Fixes
Commits that correct incorrect behavior without changing the public API.

**Indicators:**
- Resolves a crash, assertion failure, or incorrect output
- Fixes a regression introduced in a prior commit
- Corrects a type mismatch or memory error
- Addresses a reported GitHub issue

**Examples:**
- `fix: correct gradient accumulation in wp.Tape`
- `fix off-by-one in tile_load bounds check`
- `resolve segfault when freeing pinned memory on Windows`

---

### `perf` — Performance Improvements
Commits that improve speed, memory usage, or compilation time without changing behavior.

**Indicators:**
- Reduces kernel launch overhead
- Improves memory allocation strategy
- Speeds up Python-side graph construction
- Reduces JIT compilation time

**Examples:**
- `perf: cache module hash to avoid redundant recompilation`
- `reduce allocations in ScopedMemoryTracker hot path`

---

### `refactor` — Internal Refactoring
Commits that restructure code without changing external behavior.

**Indicators:**
- Renames internal symbols or reorganizes modules
- Extracts helper functions
- Consolidates duplicated logic
- No user-visible change

**Examples:**
- `refactor: split codegen.py into codegen/ package`
- `move Allocator base class to warp/memory.py`

---

### `docs` — Documentation
Commits that only affect documentation, docstrings, or comments.

**Indicators:**
- Updates README, CHANGELOG, or RST docs
- Adds or corrects docstrings
- Fixes typos in comments

---

### `test` — Tests
Commits that add or modify tests without affecting production code.

**Indicators:**
- Adds new test cases to `warp/tests/`
- Fixes a flaky or broken test
- Adds regression tests for a fixed bug

---

### `build` / `ci` — Build & CI
Commits that affect build scripts, CI pipelines, or packaging.

**Indicators:**
- Changes to `setup.py`, `CMakeLists.txt`, or GitHub Actions workflows
- Updates dependency versions
- Modifies Docker or conda environment files

---

### `chore` — Maintenance
Routine maintenance tasks that don't fit other categories.

**Indicators:**
- Version bumps
- License header updates
- Removing dead code or stale TODOs

---

## Ambiguous Cases

| Situation | Recommended Classification |
|-----------|---------------------------|
| Bug fix that also improves perf | `fix` (primary intent wins) |
| New test added alongside a fix | `fix` (tests are secondary) |
| Refactor that accidentally changes behavior | Escalate — may need `fix` |
| Doc update that also fixes a code example | `docs` if example was non-functional, `fix` if it was shipped |
| Feature flagged behind env var | `feat` (still user-facing) |

---

## Changelog Inclusion Rules

Only the following categories should appear in the public CHANGELOG:

- `feat` — always include
- `fix` — always include
- `perf` — include if user-observable
- `refactor` — include only if it affects public API surface
- `docs` — omit unless it corrects a significant error
- `test`, `build`, `ci`, `chore` — omit
