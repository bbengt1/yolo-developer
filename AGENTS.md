# Repository Guidelines

## Project Structure & Module Organization
- `src/yolo_developer/` holds the library and CLI implementation; key domains are `agents/`, `orchestrator/`, `memory/`, `gates/`, `seed/`, `audit/`, `config/`, `cli/`, `sdk/`, and `mcp/`.
- `tests/` is split into `unit/`, `integration/`, `e2e/`, plus `fixtures/` for shared data.
- `docs/` contains GitHub Pages documentation; `_bmad/` and `_bmad-output/` store BMad workflow assets and artifacts.

## Build, Test, and Development Commands
- `uv sync --all-extras` installs dev dependencies.
- `uv run yolo --help` verifies the CLI entry point.
- `uv run pytest` runs the full test suite.
- `uv run pytest --cov=src/yolo_developer --cov-report=term-missing` reports coverage.
- `uv run ruff check src tests` and `uv run ruff format src tests` lint/format.
- `uv run mypy src/yolo_developer` runs strict type checks.

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indentation, double quotes enforced by Ruff.
- Add `from __future__ import annotations` to new Python files.
- Public APIs should be exported in package `__init__.py` files.
- Use `snake_case` for functions/vars, `PascalCase` for classes, `UPPER_CASE` for constants.
- Prefer explicit error messages with full paths; use `ConfigurationError` for config-related failures.

## Testing Guidelines
- Frameworks: `pytest` + `pytest-asyncio`.
- Naming: files `test_*.py`, functions `test_*`, classes `Test*`.
- Unit tests live in `tests/unit/`; integration tests in `tests/integration/`.
- Example: `uv run pytest tests/unit/config/test_schema.py::TestYoloConfig::test_method_name -v`.

## Commit & Pull Request Guidelines
- Commit messages follow conventional style: `feat:`, `fix:`, `docs:`, with optional scopes like `fix(docs):` and story/issue refs in parentheses.
- PRs should include a clear description, linked issues, and test commands run; add screenshots/log snippets for CLI or UX changes.

## Security & Configuration Tips
- Configuration defaults to `yolo.yaml` with environment overrides using `YOLO_` and `__` for nesting (e.g., `YOLO_LLM__OPENAI_API_KEY`).
- Keep secrets in env vars only; do not commit keys or `.yolo/` artifacts.

## Agent-Specific Instructions
- This repo follows BMad workflows in `_bmad/bmm/workflows/`; sprint status is tracked in `_bmad-output/implementation-artifacts/sprint-status.yaml`.
- After completing any user story or issue, update all documentation to reflect the change, including the root `README.md` and GitHub Pages docs in `docs/`.
- After completing any issue, close it out with comments describing what was done.
