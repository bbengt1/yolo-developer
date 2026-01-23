# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YOLO Developer is an autonomous multi-agent AI development system using the BMad Method. It orchestrates specialized AI agents (Analyst, PM, Architect, Dev, SM, TEA) through a LangGraph-based orchestration engine to autonomously handle software development tasks.

**Current Status:** Epic 1 (Project Initialization & Configuration) is complete. Epic 2 (Memory & Context Layer) is next.

## Build, Test, and Lint Commands

```bash
# Install dependencies
uv sync --all-extras

# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src/yolo_developer --cov-report=term-missing

# Run a single test file
uv run pytest tests/unit/config/test_schema.py

# Run a specific test
uv run pytest tests/unit/config/test_schema.py::TestYoloConfig::test_method_name -v

# Type checking
uv run mypy src/yolo_developer

# Linting and formatting
uv run ruff check src tests
uv run ruff format src tests

# CLI entry point
uv run yolo --help
```

## Architecture

### Core Components

The system uses a **LangGraph-based multi-agent orchestration** pattern:

```
src/yolo_developer/
├── agents/         # Individual agent modules (analyst, pm, architect, dev, sm, tea)
├── orchestrator/   # LangGraph graph definition, state schema, node functions
├── memory/         # Vector store (ChromaDB) and graph store integration
├── gates/          # Quality gate framework with blocking mechanism
├── audit/          # Decision logging and traceability
├── config/         # Pydantic configuration with YAML + env var support
├── cli/            # Typer CLI interface
├── sdk/            # Python SDK for programmatic access
└── mcp/            # FastMCP server for external integration
```

### Configuration System (Implemented)

Configuration uses a three-layer priority system: **defaults → YAML → environment variables**

```python
from yolo_developer.config import load_config, YoloConfig
config = load_config()  # Loads from ./yolo.yaml with env var overrides
```

Environment variables use `YOLO_` prefix with `__` as nested delimiter:
- `YOLO_PROJECT_NAME` - Project name (required)
- `YOLO_LLM__CHEAP_MODEL` - Model for routine tasks
- `YOLO_LLM__OPENAI_API_KEY` - API key (secrets only via env vars)
- `YOLO_QUALITY__TEST_COVERAGE_THRESHOLD` - Quality threshold (0.0-1.0)

API keys (`openai_api_key`, `anthropic_api_key`) are stored as `SecretStr` and **never** included in config exports.

### External CLI Tools Integration

YOLO Developer can delegate tasks to external CLI tools like Claude Code and Aider. Tools are configured in `yolo.yaml`:

```yaml
tools:
  claude_code:
    enabled: true
    timeout: 300  # seconds
    output_format: json  # json or text
    extra_args: []  # additional CLI args
  aider:
    enabled: false
```

Or via environment variables:
- `YOLO_TOOLS__CLAUDE_CODE__ENABLED=true`
- `YOLO_TOOLS__CLAUDE_CODE__TIMEOUT=600`
- `YOLO_TOOLS__AIDER__ENABLED=true`

Check tool status with:
```bash
uv run yolo tools           # Show tool availability
uv run yolo tools status    # Same as above
uv run yolo tools --json    # JSON output
```

Tool integration provides:
- **Security**: Delegated authentication (tools use their own credentials)
- **Features**: Access to tool-specific capabilities (Claude Code plan mode, MCP client, web search)
- **Flexibility**: Configure tools independently per project

### Test Organization

```
tests/
├── unit/           # Unit tests (config/, agents/, gates/, etc.)
├── integration/    # Integration tests
├── e2e/            # End-to-end tests
└── fixtures/       # Shared test fixtures and mocks
```

## BMad Workflow System

This project uses the BMad Method for AI-assisted development. Workflows are in `_bmad/bmm/workflows/`:

- `/bmad:bmm:workflows:create-story` - Create next story from backlog
- `/bmad:bmm:workflows:dev-story` - Implement a story using red-green-refactor TDD
- `/bmad:bmm:workflows:code-review` - Adversarial code review
- `/bmad:bmm:workflows:sprint-planning` - Sprint status tracking

Sprint status is tracked in `_bmad-output/implementation-artifacts/sprint-status.yaml`.

## Code Style Requirements

- Use `from __future__ import annotations` in all Python files
- Export public API from `__init__.py` files
- Use `ConfigurationError` for all config-related errors
- Include full paths in error messages for debugging
- Run `ruff check`, `ruff format`, and `mypy` before committing
- mypy is configured with `strict = true`

## GitHub Issue Workflow

When implementing work from a GitHub issue, follow these practices:

### Starting Work
1. **Check current branch** - If on `main`, create a feature branch before starting work
   - Branch naming: `feat/issue-{number}-short-description` or `fix/issue-{number}-short-description`
   - Example: `feat/issue-17-cli-tool-integration`
2. **Follow issue tasks explicitly** - Use the actionable steps/checklist in the issue as your implementation guide
3. **Reference the issue** - Include `#issue-number` in commit messages

### During Implementation
- Complete tasks in the order specified in the issue when dependencies exist
- Mark checkboxes in the issue as tasks are completed (if you have write access)
- If scope changes or blockers arise, comment on the issue before deviating

### Completing Work
1. **Run all checks** before committing: `uv run ruff check src tests && uv run ruff format src tests && uv run mypy src/yolo_developer && uv run pytest`
2. **Create PR** referencing the issue with `Closes #issue-number` or `Fixes #issue-number` in the PR description
3. **After merge** - The issue will auto-close if using closing keywords; otherwise close manually
4. **Delete feature branch** after successful merge to keep the repository clean
