# PR Classification Examples

This document provides concrete examples of how to classify pull requests
for the Warp release audit process. Use these examples as reference when
determining whether a PR should be included in release notes and how to
categorize it.

---

## Category: Bug Fix

### Example 1 — Kernel compilation error on Windows
**PR Title:** Fix kernel compilation failure with MSVC when using `wp.func`
**Labels:** `bug`, `windows`
**Classification:** Bug Fix ✅
**Release Note:**
> Fixed a kernel compilation failure on Windows when using `@wp.func` decorated
> functions with MSVC toolchain due to incorrect name mangling.

### Example 2 — Incorrect gradient computation
**PR Title:** Fix gradient of `wp.log` for negative inputs
**Labels:** `bug`, `autodiff`
**Classification:** Bug Fix ✅
**Release Note:**
> Fixed incorrect gradient computation for `wp.log` when inputs are near zero
> or negative, which previously produced `NaN` values during backpropagation.

---

## Category: New Feature

### Example 1 — New built-in function
**PR Title:** Add `wp.smoothstep` built-in function
**Labels:** `enhancement`, `builtins`
**Classification:** New Feature ✅
**Release Note:**
> Added `wp.smoothstep(edge0, edge1, x)` built-in function for smooth
> Hermite interpolation between two values.

### Example 2 — New device support
**PR Title:** Add support for CUDA graph capture in `wp.Stream`
**Labels:** `enhancement`, `cuda`
**Classification:** New Feature ✅
**Release Note:**
> Added CUDA graph capture support via `wp.Stream.begin_capture()` and
> `wp.Stream.end_capture()`, enabling replay of recorded kernel launches
> with minimal CPU overhead.

---

## Category: Performance Improvement

### Example 1 — Faster array operations
**PR Title:** Optimize `wp.array` fill operation using CUDA memset
**Labels:** `performance`, `cuda`
**Classification:** Performance Improvement ✅
**Release Note:**
> Improved performance of `wp.array.fill()` on CUDA devices by using
> native `cudaMemset` for scalar types, reducing overhead by up to 40%.

---

## Category: Documentation

### Example 1 — API docs update
**PR Title:** Add docstrings to `wp.Mesh` methods
**Labels:** `documentation`
**Classification:** Documentation — include only if significant ⚠️
**Notes:** Include if this fills a notable gap in public API docs.

---

## Category: Internal / Skip

### Example 1 — CI configuration
**PR Title:** Update GitHub Actions runner to ubuntu-22.04
**Labels:** `ci`, `infrastructure`
**Classification:** Internal — skip ❌
**Reason:** Infrastructure change with no user-facing impact.

### Example 2 — Code style
**PR Title:** Apply clang-format to warp/native/crt.h
**Labels:** `cleanup`
**Classification:** Internal — skip ❌
**Reason:** Formatting-only change, no behavioral impact.

### Example 3 — Dependency bump
**PR Title:** Bump numpy from 1.24.0 to 1.26.4
**Labels:** `dependencies`
**Classification:** Internal — skip unless breaking ⚠️
**Notes:** Include only if the version change affects public API or minimum
supported versions.

---

## Ambiguous Cases

### Ambiguous 1 — Refactor with side effects
**PR Title:** Refactor `wp.context` module to improve startup time
**Labels:** `refactor`, `performance`
**Classification:** Performance Improvement ✅ (if measurable improvement is documented)
**Notes:** Check PR body for benchmark data. If startup improvement is
quantified, include as a performance note.

### Ambiguous 2 — Deprecation notice
**PR Title:** Deprecate `wp.config.verify_fp` in favor of `wp.config.verify_grad`
**Labels:** `deprecation`, `breaking-change`
**Classification:** Deprecation Warning ✅ — always include
**Release Note:**
> `wp.config.verify_fp` is deprecated and will be removed in a future release.
> Use `wp.config.verify_grad` instead.
