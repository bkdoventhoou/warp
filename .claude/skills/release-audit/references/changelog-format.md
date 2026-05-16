# Changelog Format Reference

This document defines the expected format and structure for changelog entries in the warp project.

## File Location

The changelog is maintained at `CHANGELOG.md` in the project root.

## Version Header Format

```
## [X.Y.Z] - YYYY-MM-DD
```

- Version numbers follow [Semantic Versioning](https://semver.org/)
- Dates use ISO 8601 format (YYYY-MM-DD)
- Unreleased changes use `## [Unreleased]`

## Section Order

Within each version block, sections must appear in this order:

1. `### Added` — new features and capabilities
2. `### Changed` — changes to existing functionality
3. `### Deprecated` — features marked for future removal
4. `### Removed` — features removed in this release
5. `### Fixed` — bug fixes
6. `### Security` — security-related fixes or improvements
7. `### Performance` — performance improvements without API changes

Omit sections that have no entries.

## Entry Format

Each entry is a bullet point under its section:

```
- Short imperative description of the change ([#PR_NUMBER](link))
```

### Rules

- Use imperative mood: "Add", "Fix", "Remove", not "Added", "Fixed", "Removed"
- Start with a capital letter
- No trailing period
- Keep entries concise (one line preferred, two lines max)
- Reference the PR or issue number when available
- Group related entries together within a section

## Example Block

```markdown
## [1.4.0] - 2024-03-15

### Added

- Add support for `wp.fabricarray` type in kernel arguments ([#312](https://github.com/nvidia/warp/pull/312))
- Add `warp.sim.ModelBuilder.add_cloth_grid` helper for cloth simulation setup

### Changed

- Upgrade minimum CUDA toolkit requirement from 11.5 to 11.8
- `wp.load_module()` now raises `RuntimeError` instead of returning `None` on failure

### Fixed

- Fix incorrect gradient computation in `wp.quat_rotate` for batched inputs ([#298](https://github.com/nvidia/warp/issues/298))
- Fix memory leak in CUDA graph capture when using `wp.Tape`

### Performance

- Improve codegen throughput for modules with large numbers of structs
```

## What NOT to Include

- Internal refactors with no user-visible effect (use `### Changed` only if behavior changes)
- CI/CD pipeline changes
- Dependency bumps with no functional impact
- Test-only changes
- Typo fixes in non-public-facing code

## Linking Conventions

PR links use the GitHub pull request URL:
```
([#NUMBER](https://github.com/nvidia/warp/pull/NUMBER))
```

Issue links:
```
([#NUMBER](https://github.com/nvidia/warp/issues/NUMBER))
```

## Release Checklist

Before tagging a release, verify:

1. `[Unreleased]` section is renamed to the new version with today's date
2. A new empty `[Unreleased]` section is added above it
3. All entries follow the format rules above
4. No duplicate entries exist across sections
5. The version in `warp/version.py` matches the changelog header
