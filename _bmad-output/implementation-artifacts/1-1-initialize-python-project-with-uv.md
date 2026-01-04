# Story 1.1: Initialize Python Project with uv

Status: done

## Story

As a developer,
I want to initialize a new YOLO Developer project using uv package manager,
So that I have a properly configured Python environment with all dependencies.

## Acceptance Criteria

1. **AC1: Project Creation**
   - **Given** I am in an empty directory
   - **When** I run `yolo init`
   - **Then** a new Python project is created with pyproject.toml

2. **AC2: Core Dependencies Installed**
   - **Given** the project is initialized
   - **When** the initialization completes
   - **Then** all core dependencies are installed:
     - langgraph>=1.0.5
     - langchain-core
     - langchain-anthropic
     - langchain-openai
     - chromadb>=1.2.0
     - typer
     - rich
     - pydantic>=2.0.0
     - pydantic-settings
     - litellm
     - tenacity
     - structlog
     - pyyaml
     - python-dotenv

3. **AC3: Development Dependencies Installed**
   - **Given** the project is initialized
   - **When** the initialization completes
   - **Then** all development dependencies are installed:
     - pytest
     - pytest-asyncio
     - pytest-cov
     - ruff
     - mypy
     - langsmith (or langfuse)

4. **AC4: PEP 621 Compliance**
   - **Given** the project is initialized
   - **When** I inspect pyproject.toml
   - **Then** the file follows PEP 621 standards
   - **And** includes proper metadata (name, version, description, authors)
   - **And** includes Python version constraint (>=3.10)

## Tasks / Subtasks

- [x] Task 1: Create CLI Entry Point (AC: 1)
  - [x] Create src/yolo_developer/cli/__init__.py
  - [x] Create src/yolo_developer/cli/main.py with Typer app
  - [x] Register `init` command in CLI
  - [x] Set up entry point in pyproject.toml: `yolo = "yolo_developer.cli.main:app"`

- [x] Task 2: Implement `yolo init` Command (AC: 1, 2, 3, 4)
  - [x] Create init command handler in cli/commands/init.py
  - [x] Generate pyproject.toml from template with all dependencies
  - [x] Run `uv sync` to install all dependencies
  - [x] Validate Python version >= 3.10

- [x] Task 3: Create pyproject.toml Template (AC: 2, 3, 4)
  - [x] Define PEP 621 compliant structure
  - [x] Include all core dependencies with version constraints
  - [x] Include all dev dependencies
  - [x] Set up CLI entry point configuration

- [x] Task 4: Write Unit Tests (AC: all)
  - [x] Test pyproject.toml generation is valid TOML
  - [x] Test all required dependencies are included
  - [x] Test PEP 621 metadata is present
  - [x] Test entry point is correctly defined

## Dev Notes

### Critical Architecture Requirements

**From ADR-008 (Configuration Pattern):**
- Use Pydantic Settings with YAML override for configuration
- Environment variables via python-dotenv
- Schema validation on all config

**From Project Structure (ARCH-STRUCT):**
```
yolo-developer/
├── src/
│   └── yolo_developer/
│       ├── __init__.py
│       ├── cli/
│       │   ├── __init__.py
│       │   ├── main.py
│       │   └── commands/
│       ├── sdk/
│       ├── mcp/
│       ├── agents/
│       ├── orchestrator/
│       ├── memory/
│       ├── gates/
│       ├── audit/
│       ├── config/
│       └── utils/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── pyproject.toml
└── README.md
```

**From ARCH-QUALITY:**
- Ruff for linting and formatting (replaces black, isort, flake8)
- mypy for static type checking
- pytest with pytest-asyncio for testing
- snake_case for all identifiers
- Async/await for all I/O operations
- Full type annotations on all functions

### Project Structure Notes

This story creates the FOUNDATION - all other stories depend on this. The directory structure must be created per architecture specification even though Story 1.2 focuses on it specifically. This story should:
1. Create minimal structure needed for CLI to work
2. Ensure pyproject.toml is complete and valid
3. Leave detailed directory structure to Story 1.2

### Package Versions (Latest as of 2026-01-04)

| Package | Version | Notes |
|---------|---------|-------|
| langgraph | 1.0.5+ | Required for orchestration |
| chromadb | 1.2.x | Rust-core rewrite, embedded mode |
| litellm | latest | Multi-provider LLM abstraction |
| typer | 0.12+ | CLI framework with rich support |
| rich | 13.x | Terminal formatting |
| pydantic | 2.x | Validation and settings |
| tenacity | 8.x | Retry logic |
| structlog | 24.x | Structured logging |

### pyproject.toml Template

```toml
[project]
name = "yolo-developer"
version = "0.1.0"
description = "Autonomous multi-agent AI development system using BMad Method"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "you@example.com"}
]
keywords = ["ai", "agents", "development", "automation"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

dependencies = [
    "langgraph>=1.0.5",
    "langchain-core",
    "langchain-anthropic",
    "langchain-openai",
    "chromadb>=1.2.0",
    "typer",
    "rich",
    "pydantic>=2.0.0",
    "pydantic-settings",
    "litellm",
    "tenacity",
    "structlog",
    "pyyaml",
    "python-dotenv",
    "fastmcp>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-asyncio",
    "pytest-cov",
    "ruff",
    "mypy",
    "langsmith",
]

[project.scripts]
yolo = "yolo_developer.cli.main:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/yolo_developer"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
python_version = "3.10"
strict = true
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

### Implementation Approach

1. **Bootstrap Problem:** This story creates the CLI that runs `yolo init`, but we need a CLI to run `yolo init`. Solution:
   - First implementation: Manual project setup following the template
   - The `yolo init` command will be used for NEW projects after the tool is installed

2. **For Initial Development:**
   ```bash
   # Manual bootstrap (one-time for this project)
   mkdir yolo-developer && cd yolo-developer
   uv init --lib
   # Copy pyproject.toml content from template above
   uv sync
   ```

3. **Then implement the `yolo init` command** that automates this for users

### References

- [Source: architecture.md#Starter Template & Technology Foundation]
- [Source: architecture.md#Project Structure]
- [Source: architecture.md#ADR-008: Configuration Pattern]
- [Source: epics.md#Story 1.1]
- [LangGraph Application Structure](https://docs.langchain.com/langgraph-platform/application-structure)
- [uv Documentation](https://docs.astral.sh/uv/)
- [PEP 621](https://peps.python.org/pep-0621/)

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

- All 40 unit tests pass (34 original + 6 from code review)
- CLI verified working: `yolo --help`, `yolo version`, `yolo init --help`
- Ruff: All checks passed
- mypy: Success, no issues found

### Completion Notes List

1. Created CLI entry point with Typer framework
2. Implemented `yolo init` command with pyproject.toml template generation
3. Fixed `typer[all]` to `typer` (extra no longer exists in typer 0.21.0)
4. Fixed `create_directory_structure` to create `__init__.py` in intermediate directories
5. All acceptance criteria verified through unit tests

**Code Review Fixes Applied:**
6. Created `py.typed` marker for PEP 561 compliance
7. Created `.gitignore` with standard Python exclusions
8. Removed unused variables (`result`, `pyproject_path`)
9. Updated type annotations from `Optional[X]` to `X | None`
10. Removed unused `pytest` import from test file
11. Added tests for `run_uv_sync` function (3 tests)
12. Added integration tests for `init_command` function (3 tests)

### Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-01-04 | Story created - ready-for-dev | SM Agent |
| 2026-01-04 | Story implemented - all tasks complete, 34 tests pass | Dev Agent |
| 2026-01-04 | Code review: Fixed 8 issues, added 6 tests (40 total) | Code Review Agent |

### File List

- `src/yolo_developer/__init__.py` - Package init with version
- `src/yolo_developer/cli/__init__.py` - CLI module init
- `src/yolo_developer/cli/main.py` - Main CLI entry point with Typer app
- `src/yolo_developer/cli/commands/__init__.py` - Commands module init
- `src/yolo_developer/cli/commands/init.py` - Init command implementation
- `src/yolo_developer/py.typed` - PEP 561 type marker
- `pyproject.toml` - Project configuration (PEP 621 compliant)
- `README.md` - Project readme
- `.gitignore` - Git ignore patterns
- `uv.lock` - uv dependency lock file
- `tests/__init__.py` - Tests package init
- `tests/unit/__init__.py` - Unit tests package init
- `tests/unit/test_init.py` - Unit tests for init command (40 tests)
