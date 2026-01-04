# Story 1.2: Create Project Directory Structure

Status: done

## Story

As a developer,
I want the project to have a standardized directory structure,
So that all code is organized following architectural best practices.

## Acceptance Criteria

1. **AC1: Complete Directory Structure**
   - **Given** I have initialized a YOLO Developer project (Story 1.1 complete)
   - **When** the `yolo init` command completes
   - **Then** the directory structure matches the architecture specification exactly

2. **AC2: Source Modules Created**
   - **Given** the project structure is created
   - **When** I inspect src/yolo_developer/
   - **Then** all required modules exist:
     - cli/ (with commands/ subdirectory)
     - sdk/
     - mcp/
     - agents/ (with prompts/ subdirectory)
     - orchestrator/
     - memory/
     - gates/ (with gates/ subdirectory)
     - seed/
     - llm/
     - audit/
     - config/
     - utils/

3. **AC3: Test Directories Created**
   - **Given** the project structure is created
   - **When** I inspect tests/
   - **Then** all required test directories exist:
     - unit/ (with subdirectories mirroring source: agents/, gates/, memory/, seed/, config/)
     - integration/
     - e2e/
     - fixtures/ (with seeds/, states/ subdirectories)

4. **AC4: Init Files Created**
   - **Given** the project structure is created
   - **When** I inspect any Python package directory
   - **Then** an __init__.py file exists
   - **And** the file is valid Python (may be empty or contain __all__)

5. **AC5: PEP 561 Compliance**
   - **Given** the project structure is created
   - **When** I inspect src/yolo_developer/
   - **Then** a py.typed marker file exists
   - **And** the package is type-check ready

## Tasks / Subtasks

- [x] Task 1: Update `yolo init` to Create Full Directory Structure (AC: 1, 2, 3, 4)
  - [x] Update cli/commands/init.py to call extended create_directory_structure
  - [x] Implement complete source module creation (12 modules per architecture)
  - [x] Implement complete test directory structure
  - [x] Ensure all __init__.py files are created including intermediate directories

- [x] Task 2: Create Source Module Directories (AC: 2, 4)
  - [x] Create sdk/ with __init__.py
  - [x] Create mcp/ with __init__.py
  - [x] Create agents/ with __init__.py and prompts/ subdirectory
  - [x] Create orchestrator/ with __init__.py
  - [x] Create memory/ with __init__.py
  - [x] Create gates/ with __init__.py and gates/ subdirectory
  - [x] Create seed/ with __init__.py
  - [x] Create llm/ with __init__.py
  - [x] Create audit/ with __init__.py
  - [x] Create config/ with __init__.py
  - [x] Create utils/ with __init__.py

- [x] Task 3: Create Test Directory Structure (AC: 3, 4)
  - [x] Create tests/unit/agents/ with __init__.py
  - [x] Create tests/unit/gates/ with __init__.py
  - [x] Create tests/unit/memory/ with __init__.py
  - [x] Create tests/unit/seed/ with __init__.py
  - [x] Create tests/unit/config/ with __init__.py
  - [x] Create tests/integration/ with __init__.py
  - [x] Create tests/e2e/ with __init__.py
  - [x] Create tests/fixtures/ with seeds/ and states/ subdirectories
  - [x] Create tests/fixtures/mocks.py stub file

- [x] Task 4: Verify py.typed Compliance (AC: 5)
  - [x] Confirm py.typed exists in src/yolo_developer/ (created in Story 1.1)
  - [x] Verify mypy can process the package structure
  - [x] Add test to verify py.typed marker is present

- [x] Task 5: Write Unit Tests (AC: all)
  - [x] Test all 12 source modules are created
  - [x] Test all test directories are created correctly
  - [x] Test all __init__.py files exist
  - [x] Test py.typed marker exists
  - [x] Test directory structure matches architecture specification exactly

## Dev Notes

### Critical Architecture Requirements

**From ARCH-STRUCT (Project Structure):**
The complete directory structure as defined in architecture.md:

```
yolo-developer/
├── src/
│   └── yolo_developer/
│       ├── __init__.py                 # Package version, public API
│       ├── py.typed                    # PEP 561 marker (ALREADY EXISTS from 1.1)
│       │
│       ├── cli/                        # CLI Interface (FR98-105) - ALREADY EXISTS
│       │   ├── __init__.py
│       │   ├── main.py                 # Typer app, entry point
│       │   └── commands/
│       │       ├── __init__.py
│       │       └── init.py             # yolo init
│       │
│       ├── sdk/                        # Python SDK (FR106-111) - TO CREATE
│       │   └── __init__.py
│       │
│       ├── mcp/                        # MCP Server (FR112-117) - TO CREATE
│       │   └── __init__.py
│       │
│       ├── agents/                     # Agent Implementations - TO CREATE
│       │   ├── __init__.py
│       │   └── prompts/
│       │       └── __init__.py
│       │
│       ├── orchestrator/               # LangGraph Orchestration - TO CREATE
│       │   └── __init__.py
│       │
│       ├── memory/                     # Memory Layer - TO CREATE
│       │   └── __init__.py
│       │
│       ├── gates/                      # Quality Gate Framework - TO CREATE
│       │   ├── __init__.py
│       │   └── gates/
│       │       └── __init__.py
│       │
│       ├── seed/                       # Seed Processing - TO CREATE
│       │   └── __init__.py
│       │
│       ├── llm/                        # LLM Abstraction - TO CREATE
│       │   └── __init__.py
│       │
│       ├── audit/                      # Audit Trail - TO CREATE
│       │   └── __init__.py
│       │
│       ├── config/                     # Configuration - TO CREATE
│       │   └── __init__.py
│       │
│       └── utils/                      # Shared Utilities - TO CREATE
│           └── __init__.py
│
├── tests/
│   ├── __init__.py                     # ALREADY EXISTS
│   ├── conftest.py                     # TO CREATE (can be empty initially)
│   ├── fixtures/                       # TO CREATE
│   │   ├── __init__.py
│   │   ├── seeds/                      # Sample seed documents
│   │   ├── states/                     # Sample state snapshots
│   │   └── mocks.py                    # LLM mocks stub
│   ├── unit/                           # ALREADY EXISTS
│   │   ├── __init__.py
│   │   ├── test_init.py                # ALREADY EXISTS
│   │   ├── agents/
│   │   │   └── __init__.py
│   │   ├── gates/
│   │   │   └── __init__.py
│   │   ├── memory/
│   │   │   └── __init__.py
│   │   ├── seed/
│   │   │   └── __init__.py
│   │   └── config/
│   │       └── __init__.py
│   ├── integration/                    # TO CREATE
│   │   └── __init__.py
│   └── e2e/                            # TO CREATE
│       └── __init__.py
│
├── pyproject.toml                      # ALREADY EXISTS
├── README.md                           # ALREADY EXISTS
├── .gitignore                          # ALREADY EXISTS
└── uv.lock                             # ALREADY EXISTS
```

### What Already Exists (from Story 1.1)

The following was created in Story 1.1:
- src/yolo_developer/__init__.py
- src/yolo_developer/py.typed
- src/yolo_developer/cli/ (with main.py and commands/init.py)
- tests/__init__.py
- tests/unit/__init__.py
- tests/unit/test_init.py
- pyproject.toml
- README.md
- .gitignore
- uv.lock

### Implementation Approach

1. **Extend the `create_directory_structure` function** in cli/commands/init.py:
   - Currently creates basic structure
   - Need to add all 12 source modules
   - Need to add complete test structure

2. **Module Creation Pattern:**
   ```python
   SOURCE_MODULES = [
       "sdk",
       "mcp",
       "agents",
       "agents/prompts",
       "orchestrator",
       "memory",
       "gates",
       "gates/gates",  # Nested for gate implementations
       "seed",
       "llm",
       "audit",
       "config",
       "utils",
   ]

   TEST_DIRECTORIES = [
       "unit/agents",
       "unit/gates",
       "unit/memory",
       "unit/seed",
       "unit/config",
       "integration",
       "e2e",
       "fixtures",
       "fixtures/seeds",
       "fixtures/states",
   ]
   ```

3. **Verify All __init__.py Files:**
   - Use pathlib for cross-platform compatibility
   - Create empty __init__.py in each package directory
   - Already handled in Story 1.1 for intermediate directories

### conftest.py Stub

Create an initial conftest.py with minimal content:

```python
"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import pytest


# Placeholder for shared fixtures
# Add fixtures here as needed during implementation
```

### mocks.py Stub

Create tests/fixtures/mocks.py with initial structure:

```python
"""Mock objects for testing LLM and external services."""

from __future__ import annotations

from typing import Any


class MockLLMResponse:
    """Mock response from LLM calls."""

    def __init__(self, content: str) -> None:
        self.content = content


# Add more mocks as needed during implementation
```

### Testing Requirements

Tests should verify:
1. **Structure Completeness:** All 12 source modules exist
2. **Test Directories:** All test subdirectories exist
3. **Init Files:** Every directory has __init__.py
4. **py.typed:** PEP 561 marker exists
5. **No Extras:** No unexpected directories created

Example test structure:

```python
class TestDirectoryStructure:
    def test_all_source_modules_created(self, project_path: Path) -> None:
        """Verify all 12 source modules exist."""
        expected_modules = [
            "cli", "sdk", "mcp", "agents", "orchestrator",
            "memory", "gates", "seed", "llm", "audit", "config", "utils"
        ]
        src_dir = project_path / "src" / "yolo_developer"
        for module in expected_modules:
            assert (src_dir / module).is_dir()
            assert (src_dir / module / "__init__.py").is_file()

    def test_agents_prompts_subdirectory(self, project_path: Path) -> None:
        """Verify agents/prompts subdirectory exists."""
        prompts_dir = project_path / "src" / "yolo_developer" / "agents" / "prompts"
        assert prompts_dir.is_dir()
        assert (prompts_dir / "__init__.py").is_file()

    def test_gates_gates_subdirectory(self, project_path: Path) -> None:
        """Verify gates/gates subdirectory exists."""
        gates_gates = project_path / "src" / "yolo_developer" / "gates" / "gates"
        assert gates_gates.is_dir()
        assert (gates_gates / "__init__.py").is_file()

    def test_test_directories_created(self, project_path: Path) -> None:
        """Verify all test directories exist."""
        tests_dir = project_path / "tests"
        expected_dirs = [
            "unit/agents", "unit/gates", "unit/memory",
            "unit/seed", "unit/config",
            "integration", "e2e",
            "fixtures", "fixtures/seeds", "fixtures/states"
        ]
        for dir_path in expected_dirs:
            assert (tests_dir / dir_path).is_dir()

    def test_py_typed_exists(self, project_path: Path) -> None:
        """Verify PEP 561 py.typed marker exists."""
        py_typed = project_path / "src" / "yolo_developer" / "py.typed"
        assert py_typed.is_file()
```

### Dependencies

**Depends On:**
- Story 1.1: Initialize Python Project with uv (DONE)

**Blocks:**
- Story 1.3: Set Up Code Quality Tooling
- Story 1.4: Implement Configuration Schema with Pydantic
- All subsequent Epic 1 stories

### References

- [Source: architecture.md#Project Structure]
- [Source: architecture.md#ARCH-STRUCT]
- [Source: epics.md#Story 1.2]
- [PEP 561 - Distributing and Packaging Type Information](https://peps.python.org/pep-0561/)

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

- All 62 unit tests pass (40 from Story 1.1 + 22 new directory structure tests)
- mypy: Success, no issues found in 18 source files
- Ruff: All checks passed

### Completion Notes List

1. Extended `create_directory_structure` function in cli/commands/init.py to create complete architecture
2. Added all 12 source modules per architecture: cli, sdk, mcp, agents, orchestrator, memory, gates, seed, llm, audit, config, utils
3. Added nested subdirectories: agents/prompts/, gates/gates/
4. Created test directory structure: unit/{agents,gates,memory,seed,config}, integration, e2e, fixtures/{seeds,states}
5. Added `create_conftest` function for pytest configuration stub
6. Added `create_mocks_stub` function for LLM mock objects
7. Fixed __init__.py creation logic to include tests/ root directory
8. Created comprehensive test file tests/unit/test_directory_structure.py with 22 tests

### Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-01-04 | Story created - ready-for-dev | SM Agent |
| 2026-01-04 | Story implemented - all tasks complete, 62 tests pass | Dev Agent |
| 2026-01-04 | Code review complete - 3 issues fixed (pytest import, typing import, .gitkeep files) | Dev Agent |

### File List

- `src/yolo_developer/cli/commands/init.py` - Extended create_directory_structure, added create_conftest, create_mocks_stub
- `src/yolo_developer/sdk/__init__.py` - New module
- `src/yolo_developer/mcp/__init__.py` - New module
- `src/yolo_developer/agents/__init__.py` - New module
- `src/yolo_developer/agents/prompts/__init__.py` - New subdirectory
- `src/yolo_developer/orchestrator/__init__.py` - New module
- `src/yolo_developer/memory/__init__.py` - New module
- `src/yolo_developer/gates/__init__.py` - New module
- `src/yolo_developer/gates/gates/__init__.py` - New subdirectory
- `src/yolo_developer/seed/__init__.py` - New module
- `src/yolo_developer/llm/__init__.py` - New module
- `src/yolo_developer/audit/__init__.py` - New module
- `src/yolo_developer/config/__init__.py` - New module
- `src/yolo_developer/utils/__init__.py` - New module
- `tests/conftest.py` - New pytest configuration
- `tests/fixtures/__init__.py` - New test package
- `tests/fixtures/mocks.py` - New LLM mocks stub
- `tests/fixtures/seeds/` - New directory for sample seeds
- `tests/fixtures/states/` - New directory for state snapshots
- `tests/unit/agents/__init__.py` - New test subdirectory
- `tests/unit/gates/__init__.py` - New test subdirectory
- `tests/unit/memory/__init__.py` - New test subdirectory
- `tests/unit/seed/__init__.py` - New test subdirectory
- `tests/unit/config/__init__.py` - New test subdirectory
- `tests/unit/test_directory_structure.py` - New test file (22 tests)
- `tests/integration/__init__.py` - New test package
- `tests/e2e/__init__.py` - New test package
