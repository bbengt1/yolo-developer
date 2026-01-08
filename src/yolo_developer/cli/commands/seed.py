"""CLI command for parsing seed documents (Story 4.2).

This module implements the `yolo seed` command that parses natural language
seed documents into structured components (goals, features, constraints).

Example:
    $ yolo seed requirements.md
    $ yolo seed requirements.md --verbose
    $ yolo seed requirements.md --json
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import structlog
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from yolo_developer.seed import SeedParseResult, parse_seed

logger = structlog.get_logger(__name__)
console = Console()


def _display_parse_results(result: SeedParseResult, verbose: bool = False) -> None:
    """Display parse results in formatted Rich output.

    Args:
        result: The parsed seed result to display.
        verbose: If True, show additional details like confidence scores.
    """
    # Summary panel
    summary_text = (
        f"[bold green]Goals:[/bold green] {result.goal_count}\n"
        f"[bold blue]Features:[/bold blue] {result.feature_count}\n"
        f"[bold yellow]Constraints:[/bold yellow] {result.constraint_count}"
    )

    if verbose:
        summary_text += f"\n\n[dim]Source:[/dim] {result.source.value}"
        summary_text += f"\n[dim]Content length:[/dim] {len(result.raw_content)} characters"
        if result.metadata:
            summary_text += (
                f"\n[dim]Metadata keys:[/dim] {', '.join(k for k, _ in result.metadata)}"
            )

    console.print(Panel(summary_text, title="Seed Parse Summary", border_style="cyan"))

    # Goals table
    if result.goals:
        goals_table = Table(title="Goals", show_header=True, header_style="bold green")
        goals_table.add_column("Title", style="green")
        goals_table.add_column("Priority", justify="center")
        goals_table.add_column("Rationale", style="dim")

        for goal in result.goals:
            goals_table.add_row(
                goal.title,
                str(goal.priority),
                goal.rationale or "-",
            )
        console.print(goals_table)
        console.print()

    # Features table
    if result.features:
        features_table = Table(title="Features", show_header=True, header_style="bold blue")
        features_table.add_column("Name", style="blue")
        features_table.add_column("Description")
        features_table.add_column("User Value", style="dim")

        for feature in result.features:
            features_table.add_row(
                feature.name,
                feature.description[:80] + "..."
                if len(feature.description) > 80
                else feature.description,
                feature.user_value or "-",
            )
        console.print(features_table)
        console.print()

    # Constraints table
    if result.constraints:
        constraints_table = Table(title="Constraints", show_header=True, header_style="bold yellow")
        constraints_table.add_column("Category", style="yellow")
        constraints_table.add_column("Description")
        constraints_table.add_column("Impact", style="dim")

        for constraint in result.constraints:
            constraints_table.add_row(
                constraint.category.value.title(),
                constraint.description[:80] + "..."
                if len(constraint.description) > 80
                else constraint.description,
                constraint.impact or "-",
            )
        console.print(constraints_table)

    # Verbose mode: show additional metadata
    if verbose and result.goals:
        console.print()
        console.print("[dim]Goal Details (Verbose):[/dim]")
        for i, goal in enumerate(result.goals, 1):
            console.print(f"  {i}. [bold]{goal.title}[/bold]")
            console.print(f"     Description: {goal.description}")

    if verbose and result.features:
        console.print()
        console.print("[dim]Feature Details (Verbose):[/dim]")
        for i, feature in enumerate(result.features, 1):
            console.print(f"  {i}. [bold]{feature.name}[/bold]")
            console.print(f"     Description: {feature.description}")
            if feature.related_goals:
                console.print(f"     Related Goals: {', '.join(feature.related_goals)}")


def _output_json(result: SeedParseResult) -> None:
    """Output parse result as formatted JSON.

    Args:
        result: The parsed seed result to output as JSON.
    """
    result_dict = result.to_dict()
    console.print_json(json.dumps(result_dict, indent=2))


def _read_seed_file(file_path: Path) -> str:
    """Read and validate seed file content.

    Args:
        file_path: Path to the seed file.

    Returns:
        The file content as string.

    Raises:
        typer.Exit: If file cannot be read.
    """
    logger.info("reading_seed_file", file_path=str(file_path))

    # Validate file exists
    if not file_path.exists():
        console.print(
            f"[red]Error:[/red] File not found: [bold]{file_path}[/bold]\n\n"
            f"[dim]Please check that the file path is correct and the file exists.[/dim]"
        )
        raise typer.Exit(code=1)

    # Validate it's a file (not a directory)
    if not file_path.is_file():
        console.print(
            f"[red]Error:[/red] Path is not a file: [bold]{file_path}[/bold]\n\n"
            f"[dim]Expected a file, but found a directory. Please provide a file path.[/dim]"
        )
        raise typer.Exit(code=1)

    # Read file content
    try:
        content = file_path.read_text(encoding="utf-8")
        logger.info(
            "seed_file_read_success",
            file_path=str(file_path),
            content_length=len(content),
        )
        return content
    except PermissionError as e:
        console.print(
            f"[red]Error:[/red] Permission denied reading: [bold]{file_path}[/bold]\n\n"
            f"[dim]Check file permissions and try again.[/dim]"
        )
        raise typer.Exit(code=1) from e
    except UnicodeDecodeError as e:
        console.print(
            f"[red]Error:[/red] Cannot decode file: [bold]{file_path}[/bold]\n\n"
            f"[dim]The file appears to use a non-UTF-8 encoding. "
            f"Please convert it to UTF-8 and try again.[/dim]"
        )
        raise typer.Exit(code=1) from e


async def _parse_seed_async(content: str, filename: str) -> SeedParseResult:
    """Async wrapper for parse_seed.

    Args:
        content: The seed document content.
        filename: The original filename for format detection.

    Returns:
        The parsed seed result.
    """
    return await parse_seed(content, filename=filename)


def seed_command(
    file_path: Path,
    verbose: bool = False,
    json_output: bool = False,
) -> None:
    """Parse a seed document and display structured results.

    Reads a natural language seed document, parses it into structured
    components (goals, features, constraints), and displays the results.

    Args:
        file_path: Path to the seed document file.
        verbose: If True, show additional details in output.
        json_output: If True, output results as JSON instead of tables.
    """
    logger.info(
        "seed_command_started",
        file_path=str(file_path),
        verbose=verbose,
        json_output=json_output,
    )

    # Read the seed file
    content = _read_seed_file(file_path)

    # Show progress (only in non-JSON mode)
    if not json_output:
        console.print(
            f"[blue]Parsing seed file:[/blue] [bold]{file_path.name}[/bold] "
            f"[dim]({len(content)} characters)[/dim]"
        )
        console.print()

    # Parse the seed content
    try:
        result = asyncio.run(_parse_seed_async(content, file_path.name))
        logger.info(
            "seed_parsing_success",
            goals=result.goal_count,
            features=result.feature_count,
            constraints=result.constraint_count,
        )
    except Exception as e:
        logger.error("seed_parsing_failed", error=str(e))
        console.print(
            f"[red]Error:[/red] Failed to parse seed document\n\n"
            f"[dim]Details: {e!s}[/dim]\n\n"
            f"[dim]This may be due to LLM API issues or malformed content. "
            f"Check your API configuration and try again.[/dim]"
        )
        raise typer.Exit(code=1) from e

    # Output results
    if json_output:
        _output_json(result)
    else:
        _display_parse_results(result, verbose=verbose)
        console.print()
        console.print(
            "[green]Seed parsing complete![/green] "
            "[dim]Use --json for machine-readable output.[/dim]"
        )

    logger.info("seed_command_completed", file_path=str(file_path))
