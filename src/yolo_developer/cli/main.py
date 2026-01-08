"""Main CLI entry point for YOLO Developer."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from yolo_developer.cli.commands.init import init_command
from yolo_developer.cli.commands.seed import seed_command

app = typer.Typer(
    name="yolo",
    help="YOLO Developer - Autonomous multi-agent AI development system",
    no_args_is_help=True,
)
console = Console()


@app.command("init")
def init(
    path: str | None = typer.Argument(
        None,
        help="Directory to initialize the project in. Defaults to current directory.",
    ),
    name: str | None = typer.Option(
        None,
        "--name",
        "-n",
        help="Project name. Defaults to directory name.",
    ),
    author: str | None = typer.Option(
        None,
        "--author",
        "-a",
        help="Author name for pyproject.toml.",
    ),
    email: str | None = typer.Option(
        None,
        "--email",
        "-e",
        help="Author email for pyproject.toml.",
    ),
) -> None:
    """Initialize a new YOLO Developer project.

    Creates a new Python project with all required dependencies for
    autonomous multi-agent development using the BMad Method.
    """
    init_command(path=path, name=name, author=author, email=email)


@app.command("version")
def version() -> None:
    """Show YOLO Developer version."""
    from yolo_developer import __version__

    console.print(f"YOLO Developer v{__version__}")


@app.command("seed")
def seed(
    file_path: Path = typer.Argument(  # noqa: B008
        ...,
        help="Path to the seed document file to parse.",
        exists=False,  # We handle existence check in command for better error messages
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output including confidence scores and metadata.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output results as JSON instead of formatted tables.",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Detect ambiguities and prompt for clarification before parsing.",
    ),
    validate_sop: bool = typer.Option(
        False,
        "--validate-sop",
        help="Validate seed against SOP constraints.",
    ),
    sop_store: Path | None = typer.Option(  # noqa: B008
        None,
        "--sop-store",
        help="Path to SOP store JSON file with constraints.",
    ),
    override_soft: bool = typer.Option(
        False,
        "--override-soft",
        help="Auto-override all SOFT SOP conflicts without prompting.",
    ),
    report_format: str | None = typer.Option(
        None,
        "--report-format",
        "-r",
        help="Generate validation report in specified format (json, markdown, rich).",
    ),
    report_output: Path | None = typer.Option(  # noqa: B008
        None,
        "--report-output",
        "-o",
        help="File path to write the validation report to.",
    ),
) -> None:
    """Parse a seed document into structured components.

    Reads a natural language requirements document and extracts:
    - Goals: High-level project objectives
    - Features: Discrete functional capabilities
    - Constraints: Technical, business, or other limitations

    In interactive mode (--interactive), the command will:
    1. Detect ambiguities in the seed document
    2. Display ambiguities with resolution prompts
    3. Allow you to clarify each ambiguity
    4. Re-parse with your clarifications applied

    With SOP validation (--validate-sop), the command will:
    1. Load constraints from the SOP store
    2. Validate seed against constraints
    3. Report HARD conflicts (block processing)
    4. Prompt for SOFT conflict overrides (or auto-override with --override-soft)

    With validation reports (--report-format), the command will:
    1. Parse the seed document
    2. Calculate quality metrics (ambiguity, SOP, extraction scores)
    3. Generate a comprehensive validation report
    4. Output in the specified format (json, markdown, or rich console)

    Examples:
        yolo seed requirements.md
        yolo seed requirements.md --verbose
        yolo seed requirements.md --json
        yolo seed requirements.md --interactive
        yolo seed requirements.md --validate-sop --sop-store constraints.json
        yolo seed requirements.md --validate-sop --override-soft
        yolo seed requirements.md --report-format json
        yolo seed requirements.md --report-format markdown -o report.md
        yolo seed requirements.md --report-format rich
    """
    seed_command(
        file_path=file_path,
        verbose=verbose,
        json_output=json_output,
        interactive=interactive,
        validate_sop=validate_sop,
        sop_store_path=sop_store,
        override_soft=override_soft,
        report_format=report_format,
        report_output=report_output,
    )


if __name__ == "__main__":
    app()
