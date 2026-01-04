"""Initialize a new YOLO Developer project."""

import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

console = Console()

# PEP 621 compliant pyproject.toml template
PYPROJECT_TEMPLATE = '''[project]
name = "{name}"
version = "0.1.0"
description = "Autonomous multi-agent AI development system using BMad Method"
readme = "README.md"
requires-python = ">=3.10"
license = {{text = "MIT"}}
authors = [
    {{name = "{author}", email = "{email}"}}
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
'''

README_TEMPLATE = '''# {name}

Autonomous multi-agent AI development system using BMad Method.

## Installation

```bash
uv sync
```

## Usage

```bash
yolo --help
```

## Development

```bash
uv sync --all-extras
uv run pytest
```
'''


def validate_python_version() -> bool:
    """Validate Python version is >= 3.10."""
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 10):
        console.print(
            f"[red]Error: Python 3.10+ required. Found {major}.{minor}[/red]"
        )
        return False
    return True


def create_directory_structure(project_path: Path) -> None:
    """Create the complete directory structure for YOLO Developer per architecture spec."""
    # Source modules per architecture specification
    source_directories = [
        "src/yolo_developer/cli/commands",
        "src/yolo_developer/sdk",
        "src/yolo_developer/mcp",
        "src/yolo_developer/agents/prompts",  # agents with prompts subdirectory
        "src/yolo_developer/orchestrator",
        "src/yolo_developer/memory",
        "src/yolo_developer/gates/gates",  # gates with gates subdirectory for implementations
        "src/yolo_developer/seed",
        "src/yolo_developer/llm",
        "src/yolo_developer/audit",
        "src/yolo_developer/config",
        "src/yolo_developer/utils",
    ]

    # Test directories per architecture specification
    test_directories = [
        "tests/unit/agents",
        "tests/unit/gates",
        "tests/unit/memory",
        "tests/unit/seed",
        "tests/unit/config",
        "tests/integration",
        "tests/e2e",
        "tests/fixtures/seeds",
        "tests/fixtures/states",
    ]

    directories = source_directories + test_directories

    # Collect all Python package directories
    python_package_dirs: set[Path] = set()

    for dir_path in directories:
        full_path = project_path / dir_path
        full_path.mkdir(parents=True, exist_ok=True)

        # Collect all parent directories that should be Python packages
        if dir_path.startswith("src/") or dir_path.startswith("tests/"):
            # Walk from the deepest directory up to src/ or tests/
            current = full_path
            while current != project_path:
                rel_path = current.relative_to(project_path)
                rel_str = str(rel_path)
                # Include tests/ directory itself and all src/yolo_developer/... directories
                if rel_str == "tests" or rel_str.startswith("tests/") or rel_str.startswith("src/"):
                    python_package_dirs.add(current)
                current = current.parent

    # Create __init__.py in all Python package directories
    for pkg_dir in python_package_dirs:
        init_file = pkg_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text('"""Package."""\n')

    # Create .gitkeep files in empty fixture directories to preserve them in git
    gitkeep_dirs = [
        project_path / "tests" / "fixtures" / "seeds",
        project_path / "tests" / "fixtures" / "states",
    ]
    for gitkeep_dir in gitkeep_dirs:
        gitkeep_file = gitkeep_dir / ".gitkeep"
        if not gitkeep_file.exists():
            gitkeep_file.touch()


def create_pyproject_toml(
    project_path: Path,
    name: str,
    author: str,
    email: str,
) -> Path:
    """Create pyproject.toml from template."""
    content = PYPROJECT_TEMPLATE.format(
        name=name,
        author=author,
        email=email,
    )
    pyproject_path = project_path / "pyproject.toml"
    pyproject_path.write_text(content)
    return pyproject_path


def create_readme(project_path: Path, name: str) -> None:
    """Create README.md."""
    content = README_TEMPLATE.format(name=name)
    readme_path = project_path / "README.md"
    readme_path.write_text(content)


def create_py_typed(project_path: Path) -> None:
    """Create py.typed marker for PEP 561 compliance."""
    py_typed_path = project_path / "src" / "yolo_developer" / "py.typed"
    py_typed_path.touch()


def create_conftest(project_path: Path) -> None:
    """Create pytest conftest.py with initial configuration."""
    conftest_content = '''"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import pytest  # noqa: F401


# Placeholder for shared fixtures
# Add fixtures here as needed during implementation
'''
    conftest_path = project_path / "tests" / "conftest.py"
    if not conftest_path.exists():
        conftest_path.write_text(conftest_content)


def create_mocks_stub(project_path: Path) -> None:
    """Create tests/fixtures/mocks.py stub file for LLM mocking."""
    mocks_content = '''"""Mock objects for testing LLM and external services."""

from __future__ import annotations

from typing import Any  # noqa: F401


class MockLLMResponse:
    """Mock response from LLM calls."""

    def __init__(self, content: str) -> None:
        self.content = content


# Add more mocks as needed during implementation
'''
    mocks_path = project_path / "tests" / "fixtures" / "mocks.py"
    if not mocks_path.exists():
        mocks_path.write_text(mocks_content)


def run_uv_sync(project_path: Path) -> bool:
    """Run uv sync to install dependencies."""
    try:
        subprocess.run(
            ["uv", "sync", "--all-extras"],
            cwd=project_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error running uv sync: {e.stderr}[/red]")
        return False
    except FileNotFoundError:
        console.print(
            "[yellow]Warning: uv not found. Run 'uv sync' manually after installation.[/yellow]"
        )
        return True  # Don't fail if uv is not installed


def init_command(
    path: str | None = None,
    name: str | None = None,
    author: str | None = None,
    email: str | None = None,
) -> None:
    """Initialize a new YOLO Developer project."""
    # Validate Python version
    if not validate_python_version():
        raise SystemExit(1)

    # Resolve project path
    project_path = Path(path) if path else Path.cwd()
    project_path = project_path.resolve()

    # Create directory if it doesn't exist
    project_path.mkdir(parents=True, exist_ok=True)

    # Derive project name
    project_name = name or project_path.name

    # Default author info
    author_name = author or "YOLO Developer"
    author_email = email or "dev@example.com"

    console.print(
        Panel(
            f"Initializing YOLO Developer project: [bold]{project_name}[/bold]",
            title="YOLO Init",
            border_style="blue",
        )
    )

    # Create directory structure
    console.print("[blue]Creating directory structure...[/blue]")
    create_directory_structure(project_path)

    # Create pyproject.toml
    console.print("[blue]Creating pyproject.toml...[/blue]")
    create_pyproject_toml(project_path, project_name, author_name, author_email)

    # Create README
    console.print("[blue]Creating README.md...[/blue]")
    create_readme(project_path, project_name)

    # Create py.typed marker
    console.print("[blue]Creating py.typed marker...[/blue]")
    create_py_typed(project_path)

    # Create pytest conftest.py
    console.print("[blue]Creating pytest configuration...[/blue]")
    create_conftest(project_path)

    # Create mocks stub
    console.print("[blue]Creating test fixtures...[/blue]")
    create_mocks_stub(project_path)

    # Run uv sync
    console.print("[blue]Installing dependencies with uv...[/blue]")
    if run_uv_sync(project_path):
        console.print("[green]Dependencies installed successfully![/green]")
    else:
        console.print("[yellow]Please run 'uv sync' manually.[/yellow]")

    console.print(
        Panel(
            f"[green]Project initialized successfully at:[/green]\n{project_path}\n\n"
            f"[blue]Next steps:[/blue]\n"
            f"  cd {project_path}\n"
            f"  yolo --help",
            title="Success",
            border_style="green",
        )
    )
