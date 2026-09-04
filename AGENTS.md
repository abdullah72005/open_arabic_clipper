# Repository Guidelines

## Project Structure & Module Organization

This repository is currently a minimal scaffold. `README.md` introduces the project and `LICENSE` contains its license. Keep implementation code in a top-level directory that matches the selected stack (for example, `src/`), with automated tests in `tests/` or colocated as `*.test.*`. Put static, non-code files in `assets/`. Update `README.md` whenever a new build tool, entry point, or required service is introduced.

Repository-local AI workflows live in `.agents/skills/`; do not edit installed skill files unless intentionally maintaining them. `skills-lock.json` records their sources and should be committed with skill changes.

## Build, Test, and Development Commands

No application runtime, package manifest, or test suite exists yet. Do not document or rely on imaginary commands. Once tooling is added, expose the standard development, lint, test, and production-build commands in `README.md` and keep this section synchronized. Useful repository checks today are:

```bash
git status --short       # show pending changes
npx skills list --json   # inspect project AI skills
```

## Coding Style & Naming Conventions

Follow the formatter, linter, and conventions of the language selected for the project; add their configuration at the repository root. Use descriptive, lowercase, hyphenated names for documentation and assets (`api-reference.md`, `logo-mark.svg`). Use the language’s conventional source-file naming and avoid unrelated refactors in focused changes. Format and lint modified files before requesting review.

## Testing Guidelines

Add tests alongside each new feature or bug fix. Name tests after observable behavior, such as `clips_selected_text_when_triggered`. Keep test inputs deterministic and avoid network-dependent tests unless they are explicitly integration tests. When a test runner is introduced, document the exact local command and any coverage threshold in `README.md`.

## Commit & Pull Request Guidelines

The current history contains only `Initial commit`, so no established convention exists. Use concise, imperative subjects: `Add clipboard parser` or `Fix empty selection handling`. Keep commits narrowly scoped. Pull requests should explain the change and test evidence, link related issues, and include screenshots or recordings for visible UI changes. Call out configuration, migration, or security implications explicitly.

## AI-Assisted Work

Superpowers and Caveman are installed locally for Codex-compatible agents. Use Superpowers skills only when task scope matches their trigger conditions; do not invoke them merely because they are installed. Use Caveman by default for agent responses, except where clarity or safety requires normal prose. Update skills with `npx skills update -y` and review the resulting diff before committing.
