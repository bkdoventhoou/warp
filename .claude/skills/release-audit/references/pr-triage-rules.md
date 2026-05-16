# PR Triage Rules

This document defines rules for triaging pull requests during release audits.
Use these rules to determine how PRs should be categorized and prioritized.

---

## Priority Levels

### P0 — Critical (Must include in release notes)
- Fixes a regression introduced in the current release cycle
- Resolves a crash, data corruption, or silent incorrect behavior
- Security-related fix
- Breaks backward compatibility in a public API

### P1 — High (Should include in release notes)
- New user-facing feature or capability
- Performance improvement with measurable impact (>10% on a standard benchmark)
- Deprecation of a public API or behavior
- Bug fix affecting a documented workflow

### P2 — Medium (Include if space permits)
- Internal refactor with no user-visible behavior change
- Test coverage improvements
- Documentation corrections or additions
- Minor performance improvements

### P3 — Low (Omit from release notes)
- CI/CD pipeline changes
- Dependency bumps with no functional impact
- Code style or formatting changes
- Typo fixes in non-user-facing strings

---

## Label Mapping

When GitHub labels are present, apply the following mappings before manual review:

| Label                  | Default Priority |
|------------------------|------------------|
| `bug`                  | P1               |
| `critical` / `hotfix`  | P0               |
| `enhancement`          | P1               |
| `performance`          | P1               |
| `refactor`             | P2               |
| `documentation`        | P2               |
| `test`                 | P2               |
| `ci` / `infra`         | P3               |
| `chore`                | P3               |
| `security`             | P0               |
| `deprecation`          | P1               |

> Labels are a starting point only. Always verify by reading the PR description
> and linked issues before finalizing priority.

---

## Merge Status Rules

- Only **merged** PRs should appear in release notes.
- **Closed (unmerged)** PRs must be excluded entirely.
- **Draft** PRs must be excluded unless explicitly flagged by a maintainer.
- If a PR was merged and then reverted within the same release cycle, **exclude both**.

---

## Duplicate Detection

- If two PRs address the same issue, keep only the one that was merged last.
- If a PR is a follow-up fix to another PR in the same cycle, group them under
  a single changelog entry referencing both PR numbers.
- Cherry-picks from a previous release branch should be noted as backports,
  not as new features.

---

## Special Cases

### Warp-Specific Considerations
- PRs touching `warp/stubs/` or `warp/codegen/` should be reviewed for
  user-visible type annotation or code generation changes before downgrading
  to P2/P3.
- PRs modifying `warp/native/` (C++/CUDA kernels) require checking whether
  the change affects kernel correctness, performance, or the public Python API.
- PRs that only update `CHANGELOG.md` itself are P3 and should not generate
  a new changelog entry.

### Version Bumps
- PRs that solely bump `VERSION` or `pyproject.toml` version strings are P3.
- If a version bump PR also includes release preparation changes (e.g., updating
  deprecation warnings), elevate to P2.

---

## Output Format for Triage Table

When producing a triage summary, use the following Markdown table format:

```
| PR # | Title (truncated to 60 chars) | Author | Priority | Category | Notes |
|------|-------------------------------|--------|----------|----------|-------|
| 1234 | Fix kernel launch on CUDA 12.x | @user | P0       | Bug Fix  |       |
```

Sort the table by Priority (P0 first), then by PR number descending.
