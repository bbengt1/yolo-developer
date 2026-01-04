# Story 1.3: Set Up Code Quality Tooling

Status: done

## Story

As a developer,
I want code quality tools (ruff, mypy, pre-commit) configured,
So that code quality is enforced automatically from the start.

## Acceptance Criteria

1. **AC1: Ruff Linting Configured**
   - **Given** I have the project initialized with directory structure (Story 1.2 complete)
   - **When** I run `uv run ruff check src/ tests/`
   - **Then** ruff executes without configuration errors
   - **And** the configuration is defined in pyproject.toml

2. **AC2: Ruff Formatting Configured**
   - **Given** ruff is configured for the project
   - **When** I run `uv run ruff format src/ tests/`
   - **Then** code is formatted according to project standards
   - **And** the configuration matches pyproject.toml settings

3. **AC3: Mypy Type Checking Configured**
   - **Given** the project has type annotations
   - **When** I run `uv run mypy src/`
   - **Then** mypy executes with strict mode enabled
   - **And** the py.typed marker is recognized
   - **And** configuration is defined in pyproject.toml

4. **AC4: Pre-commit Hooks Installed**
   - **Given** the project has .pre-commit-config.yaml
   - **When** I run `pre-commit install`
   - **Then** hooks are installed successfully
   - **And** hooks run on git commit

5. **AC5: All Tools Pass on Clean Codebase**
   - **Given** all quality tools are configured
   - **When** I run all quality checks on the existing codebase
   - **Then** ruff check passes with no errors
   - **And** ruff format reports no changes needed
   - **And** mypy reports no type errors
   - **And** the codebase is considered "clean"

## Tasks / Subtasks

- [x] Task 1: Update Ruff Configuration in pyproject.toml (AC: 1, 2)
  - [x] Configure ruff line-length to 100 characters
  - [x] Configure target-version to py310
  - [x] Enable lint rule sets: E (errors), F (pyflakes), I (isort), N (naming), W (warnings), UP (pyupgrade)
  - [x] Configure format settings for consistency

- [x] Task 2: Update Mypy Configuration in pyproject.toml (AC: 3)
  - [x] Set python_version = "3.10"
  - [x] Enable strict = true
  - [x] Enable warn_return_any = true
  - [x] Enable warn_unused_configs = true
  - [x] Add ignore_missing_imports = true for third-party packages

- [x] Task 3: Create Pre-commit Configuration (AC: 4)
  - [x] Create .pre-commit-config.yaml in project root
  - [x] Add ruff hook for linting
  - [x] Add ruff-format hook for formatting
  - [x] Add mypy hook for type checking
  - [x] Add trailing-whitespace hook
  - [x] Add end-of-file-fixer hook
  - [x] Add check-yaml hook

- [x] Task 4: Fix Any Existing Code Quality Issues (AC: 5)
  - [x] Run ruff check and fix any linting issues
  - [x] Run ruff format to format all files
  - [x] Run mypy and fix any type errors
  - [x] Ensure all existing code passes quality checks

- [x] Task 5: Write Unit Tests for Quality Tooling (AC: all)
  - [x] Test that ruff configuration exists in pyproject.toml
  - [x] Test that mypy configuration exists in pyproject.toml
  - [x] Test that .pre-commit-config.yaml exists
  - [x] Test that pre-commit config has required hooks
  - [x] Test that ruff check passes on src/
  - [x] Test that mypy passes on src/

## Dev Notes

### Critical Architecture Requirements

**From architecture.md (Code Quality Section):**

The architecture specifies:
- **Ruff** for linting and formatting (replaces black, isort, flake8)
- **mypy** for static type checking
- **pytest** for testing with async support
- All code must use `from __future__ import annotations`

**From Implementation Patterns (architecture.md lines 1517-1523):**

Pattern Enforcement via tooling:
| Mechanism | Tool | When |
|-----------|------|------|
| Linting | Ruff | Pre-commit, CI |
| Type checking | mypy | Pre-commit, CI |
| Import sorting | Ruff (isort) | Pre-commit |
| Test naming | pytest-naming | CI |

### Ruff Configuration (From pyproject.toml template in Story 1.1)

The current pyproject.toml already has basic ruff config:
```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]
```

This should be verified and potentially expanded:

```toml
[tool.ruff]
line-length = 100
target-version = "py310"
exclude = [
    ".git",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "*.egg-info",
]

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "F",    # pyflakes
    "I",    # isort
    "N",    # pep8-naming
    "W",    # pycodestyle warnings
    "UP",   # pyupgrade
    "B",    # flake8-bugbear
    "C4",   # flake8-comprehensions
    "DTZ",  # flake8-datetimez
    "RUF",  # ruff-specific rules
]
ignore = [
    "E501",  # line too long (handled by formatter)
]

[tool.ruff.lint.isort]
known-first-party = ["yolo_developer"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false
line-ending = "auto"
```

### Mypy Configuration

The current pyproject.toml has:
```toml
[tool.mypy]
python_version = "3.10"
strict = true
warn_return_any = true
warn_unused_configs = true
```

Should be expanded to:
```toml
[tool.mypy]
python_version = "3.10"
strict = true
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
show_error_codes = true

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false
```

### Pre-commit Configuration

Create `.pre-commit-config.yaml`:

```yaml
# Pre-commit hooks for yolo-developer
# Install with: pre-commit install
# Run manually: pre-commit run --all-files

repos:
  # General file quality
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: debug-statements

  # Ruff linting and formatting
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.4
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format

  # Type checking with mypy
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.13.0
    hooks:
      - id: mypy
        additional_dependencies:
          - typer
          - rich
          - pydantic
          - pydantic-settings
        args: [--ignore-missing-imports]
        pass_filenames: false
        entry: mypy src/
```

### What Already Exists (from Story 1.1 and 1.2)

From pyproject.toml:
- Basic ruff config (line-length, target-version, select rules)
- Basic mypy config (python_version, strict, warn settings)
- Dev dependencies already include: pytest, pytest-asyncio, pytest-cov, ruff, mypy

From project structure:
- src/yolo_developer/ with all modules having __init__.py
- tests/ with conftest.py
- All Python files use `from __future__ import annotations`

### Implementation Approach

1. **Expand Ruff Configuration:**
   - Add more lint rules (B, C4, DTZ, RUF)
   - Add exclude patterns
   - Configure isort settings
   - Configure format settings

2. **Expand Mypy Configuration:**
   - Add ignore_missing_imports for third-party
   - Add show_error_codes for better debugging
   - Add test file overrides

3. **Create Pre-commit Config:**
   - Create .pre-commit-config.yaml
   - Add pre-commit to dev dependencies
   - Document installation in README

4. **Fix Any Issues:**
   - Run ruff check --fix
   - Run ruff format
   - Run mypy and fix any type issues
   - All existing code should already be clean from Stories 1.1/1.2

### Testing Requirements

Tests should verify:
1. **Configuration Existence:** pyproject.toml has ruff and mypy sections
2. **Pre-commit Existence:** .pre-commit-config.yaml exists with required hooks
3. **Tool Execution:** All tools can run without errors
4. **Codebase Cleanliness:** No lint/type/format issues in src/

Example test:
```python
import subprocess
from pathlib import Path

class TestCodeQualityTooling:
    def test_ruff_config_exists(self, project_root: Path) -> None:
        """Verify ruff configuration in pyproject.toml."""
        pyproject = project_root / "pyproject.toml"
        content = pyproject.read_text()
        assert "[tool.ruff]" in content
        assert "[tool.ruff.lint]" in content

    def test_mypy_config_exists(self, project_root: Path) -> None:
        """Verify mypy configuration in pyproject.toml."""
        pyproject = project_root / "pyproject.toml"
        content = pyproject.read_text()
        assert "[tool.mypy]" in content
        assert "strict = true" in content

    def test_precommit_config_exists(self, project_root: Path) -> None:
        """Verify .pre-commit-config.yaml exists."""
        precommit = project_root / ".pre-commit-config.yaml"
        assert precommit.is_file()

    def test_ruff_check_passes(self) -> None:
        """Verify ruff check passes on codebase."""
        result = subprocess.run(
            ["uv", "run", "ruff", "check", "src/", "tests/"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Ruff check failed: {result.stdout}"

    def test_mypy_passes(self) -> None:
        """Verify mypy passes on codebase."""
        result = subprocess.run(
            ["uv", "run", "mypy", "src/"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Mypy failed: {result.stdout}"
```

### Dependencies

**Depends On:**
- Story 1.1: Initialize Python Project with uv (DONE)
- Story 1.2: Create Project Directory Structure (DONE)

**Blocks:**
- Story 1.4: Implement Configuration Schema with Pydantic
- All subsequent Epic 1 stories

### References

- [Source: architecture.md#Code Quality]
- [Source: architecture.md#Pattern Enforcement]
- [Source: epics.md#Story 1.3]
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Mypy Documentation](https://mypy.readthedocs.io/)
- [Pre-commit Documentation](https://pre-commit.com/)

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

- All 86 unit tests pass (62 from Stories 1.1/1.2 + 24 code quality tests)
- mypy: Success, no issues found in 18 source files
- ruff check: All checks passed
- ruff format: All 33 files formatted correctly
- pre-commit validate-config: Configuration valid

### Completion Notes List

1. Expanded ruff configuration with additional lint rules (B, C4, DTZ, RUF), exclude patterns, isort settings, and format settings
2. Expanded mypy configuration with ignore_missing_imports, show_error_codes, and test file overrides
3. Created .pre-commit-config.yaml with hooks for trailing-whitespace, end-of-file-fixer, check-yaml, ruff, ruff-format, and mypy
4. Added pre-commit to dev dependencies in pyproject.toml
5. Fixed RUF012 violations (mutable class attributes) by adding ClassVar annotations to test files
6. Formatted all files with ruff format
7. Created comprehensive test file tests/unit/test_code_quality_tooling.py with 24 tests covering all ACs

**Code Review Fixes (2026-01-04):**
8. Added `from __future__ import annotations` to all 17 source files (architecture requirement)
9. Added DTZ rule to ruff lint select (was in story notes but missing from implementation)
10. Added `from __future__ import annotations` to test files (test_init.py, test_directory_structure.py)
11. Added missing mypy dependencies to pre-commit config (langgraph, langchain-core, chromadb, structlog, tenacity)
12. Added test_ruff_has_all_expected_lint_rules test to validate all expected rules are present

### Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-01-04 | Story created - ready-for-dev | SM Agent |
| 2026-01-04 | Story implemented - all tasks complete, 85 tests pass | Dev Agent |
| 2026-01-04 | Code review - found 6 issues, all fixed, 86 tests pass | Code Review |

### File List

- `pyproject.toml` - Expanded ruff and mypy configuration, added pre-commit to dev dependencies, added DTZ rule
- `.pre-commit-config.yaml` - New pre-commit configuration file with expanded mypy deps
- `tests/unit/test_code_quality_tooling.py` - New test file with 24 tests (added lint rules validation test)
- `tests/unit/test_directory_structure.py` - Fixed RUF012 violations, added future annotations
- `tests/unit/test_init.py` - Fixed RUF012 violations, added future annotations
- `src/yolo_developer/__init__.py` - Added future annotations
- `src/yolo_developer/agents/__init__.py` - Added future annotations
- `src/yolo_developer/agents/prompts/__init__.py` - Added future annotations
- `src/yolo_developer/audit/__init__.py` - Added future annotations
- `src/yolo_developer/cli/__init__.py` - Added future annotations
- `src/yolo_developer/cli/commands/__init__.py` - Added future annotations
- `src/yolo_developer/cli/main.py` - Added future annotations
- `src/yolo_developer/config/__init__.py` - Added future annotations
- `src/yolo_developer/gates/__init__.py` - Added future annotations
- `src/yolo_developer/gates/gates/__init__.py` - Added future annotations
- `src/yolo_developer/llm/__init__.py` - Added future annotations
- `src/yolo_developer/mcp/__init__.py` - Added future annotations
- `src/yolo_developer/memory/__init__.py` - Added future annotations
- `src/yolo_developer/orchestrator/__init__.py` - Added future annotations
- `src/yolo_developer/sdk/__init__.py` - Added future annotations
- `src/yolo_developer/seed/__init__.py` - Added future annotations
- `src/yolo_developer/utils/__init__.py` - Added future annotations
