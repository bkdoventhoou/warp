# Commit Classification Examples

This document provides concrete examples of commits and their correct classifications
for use during release audits.

## Bug Fix Examples

### Kernel / Compilation
```
fix: correct gradient computation for wp.matmul with transposed inputs
fix: resolve segfault in CUDA kernel when array stride is non-contiguous
fix: handle edge case in tile_load when tile size exceeds array bounds
```
**Classification:** `Fixed` — correctness issues in core runtime or kernels.

### Python API
```
fix: wp.from_torch() now correctly handles bool tensors
fix: resolve AttributeError when calling array.numpy() on CPU arrays with zero elements
```
**Classification:** `Fixed` — user-facing API returning wrong results or raising unexpected errors.

### Build / Install
```
fix: add missing __init__.py to warp.stubs package
fix: correct wheel metadata for Python 3.12 compatibility
```
**Classification:** `Fixed` — packaging or install-time issues.

---

## New Feature Examples

### Language / Kernels
```
feat: add wp.tile_matmul() for cooperative matrix multiply in tile programs
feat: support dynamic array slicing inside kernels
feat: add wp.rand_truncnorm() for truncated normal sampling
```
**Classification:** `Added` — net-new language or kernel capabilities.

### Python API
```
feat: add wp.ScopedDevice context manager for temporary device switching
feat: introduce wp.Stream.record_event() and wp.Stream.wait_event()
```
**Classification:** `Added` — new public Python symbols or methods.

### Integrations
```
feat: add JAX interop via wp.from_jax() and wp.to_jax()
feat: support Torch compile backend for warp kernels
```
**Classification:** `Added` — new framework integrations.

---

## Improvement / Enhancement Examples

```
perf: reduce kernel launch overhead by caching module hash
perf: vectorize adjoint computation for wp.sin / wp.cos
refactor: consolidate CUDA context management into CudaContextGuard
improve: better error message when kernel argument dtype mismatches
```
**Classification:** `Improved` — existing functionality made faster, cleaner, or more robust
without changing the public contract.

---

## Breaking Change Examples

```
refactor!: rename wp.context.runtime -> wp.context.Runtime (capitalized)
feat!: wp.launch() now requires explicit `dim` keyword argument
remove: drop support for Python 3.8
```
**Classification:** `Breaking Changes` — must appear in their own section at the top of
the release notes, before all other entries.

---

## Documentation-Only Examples

```
docs: fix typo in wp.Mesh docstring
docs: add missing return type annotation to wp.array.__getitem__
docs: expand tile programming guide with matmul example
```
**Classification:** Typically omitted from user-facing changelog unless the fix resolves
a significantly misleading doc. Use judgment.

---

## Chore / CI Examples (Usually Omitted)

```
ci: pin numpy<2.0 in test requirements
chore: bump version to 1.4.0
chore: update copyright year
ci: add Windows ARM64 build job
```
**Classification:** Do **not** include in user-facing release notes unless the change
directly affects end-user workflow (e.g., dropping a supported OS).

---

## Ambiguous Cases — Guidance

| Commit message | Recommended classification |
|---|---|
| `refactor: rewrite tape internals` | `Improved` if no API change; `Breaking Changes` if tape interface changed |
| `test: add grad check for wp.quat_rpy` | Omit |
| `fix: update stubs for new overloads` | `Fixed` — stubs are user-facing |
| `feat: add experimental wp.fft (undocumented)` | `Added` with `(experimental)` tag |
| `revert: revert wp.tile_load stride fix` | `Fixed` if the revert restores correct behavior |
