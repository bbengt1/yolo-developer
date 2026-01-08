"""CLI command for parsing seed documents (Story 4.2, 4.3).

This module implements the `yolo seed` command that parses natural language
seed documents into structured components (goals, features, constraints).
It also supports interactive ambiguity resolution via --interactive flag.

Example:
    $ yolo seed requirements.md
    $ yolo seed requirements.md --verbose
    $ yolo seed requirements.md --json
    $ yolo seed requirements.md --interactive
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import structlog
import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from yolo_developer.seed import SeedParseResult, parse_seed
from yolo_developer.seed.ambiguity import (
    Ambiguity,
    AmbiguityResult,
    AnswerFormat,
    Resolution,
    ResolutionPrompt,
    calculate_question_priority,
    detect_ambiguities,
    prioritize_questions,
)

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


def _display_ambiguities(ambiguity_result: AmbiguityResult, verbose: bool = False) -> None:
    """Display detected ambiguities in a Rich table.

    Args:
        ambiguity_result: The ambiguity detection result to display.
        verbose: If True, show resolution prompts and format hints.
    """
    if not ambiguity_result.has_ambiguities:
        console.print(
            "[green]No ambiguities detected![/green] "
            "[dim]The seed document appears clear.[/dim]"
        )
        return

    # Ambiguities table with priority column (Story 4.4)
    table = Table(title="Ambiguities Detected", show_header=True, header_style="bold red")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Priority", justify="center")
    table.add_column("Type", style="cyan")
    table.add_column("Severity", justify="center")
    table.add_column("Description")

    # Sort by priority for display (Story 4.4)
    sorted_ambiguities = prioritize_questions(list(ambiguity_result.ambiguities))

    for i, amb in enumerate(sorted_ambiguities, 1):
        severity_style = {
            "low": "green",
            "medium": "yellow",
            "high": "red",
        }.get(amb.severity.value, "white")

        # Calculate priority score (Story 4.4)
        priority_score = calculate_question_priority(amb)
        # Check if blocking type (UNDEFINED, SCOPE per AC4)
        is_blocking = amb.ambiguity_type.value in ("undefined", "scope")
        # Blocking types with severity >= MEDIUM should show as HIGH priority
        if is_blocking and priority_score >= 40:
            priority_label = "[red bold]HIGH[/red bold]"
        elif priority_score >= 50:
            priority_label = "[red bold]HIGH[/red bold]"
        elif priority_score >= 35:
            priority_label = "[yellow]MED[/yellow]"
        else:
            priority_label = "[dim]LOW[/dim]"

        table.add_row(
            str(i),
            priority_label,
            amb.ambiguity_type.value.title(),
            f"[{severity_style}]{amb.severity.value.upper()}[/{severity_style}]",
            amb.description[:50] + "..." if len(amb.description) > 50 else amb.description,
        )

    console.print(table)

    # Confidence summary
    console.print()
    confidence_style = (
        "green" if ambiguity_result.overall_confidence >= 0.8 else
        "yellow" if ambiguity_result.overall_confidence >= 0.5 else
        "red"
    )
    console.print(
        f"[bold]Ambiguity Confidence:[/bold] "
        f"[{confidence_style}]{ambiguity_result.overall_confidence:.0%}[/{confidence_style}]"
    )

    # Verbose: show resolution prompts with format hints (Story 4.4)
    if verbose and ambiguity_result.resolution_prompts:
        console.print()
        console.print("[dim]Resolution Prompts (Verbose):[/dim]")
        for i, (amb, prompt) in enumerate(
            zip(
                sorted_ambiguities,
                ambiguity_result.resolution_prompts,
                strict=False,
            ),
            1,
        ):
            console.print(f"  {i}. [bold]{amb.source_text}[/bold]")
            console.print(f"     Question: {prompt.question}")
            if prompt.suggestions:
                console.print(f"     Suggestions: {', '.join(prompt.suggestions)}")
            # Show format hint in verbose mode (Story 4.4)
            if prompt.format_hint:
                console.print(f"     [cyan]Format:[/cyan] {prompt.format_hint}")


def _prompt_for_resolution(
    amb: Ambiguity,
    prompt: ResolutionPrompt,
    index: int,
    priority_score: int | None = None,
) -> Resolution | None:
    """Prompt user for resolution of an ambiguity.

    Args:
        amb: The ambiguity to resolve.
        prompt: The resolution prompt with question and suggestions.
        index: The ambiguity index (1-based).
        priority_score: Optional priority score for display (Story 4.4).

    Returns:
        Resolution if user provided input, None if skipped.
    """
    # Calculate priority if not provided (Story 4.4)
    if priority_score is None:
        priority_score = calculate_question_priority(amb)

    # Build priority indicator (Story 4.4)
    # Check if blocking type (UNDEFINED, SCOPE per AC4)
    is_blocking = amb.ambiguity_type.value in ("undefined", "scope")
    priority_indicator = ""
    # Blocking types with severity >= MEDIUM should show as HIGH priority
    if is_blocking and priority_score >= 40:
        priority_indicator = " [red bold][HIGH PRIORITY][/red bold]"
    elif priority_score >= 50:
        priority_indicator = " [red bold][HIGH PRIORITY][/red bold]"
    elif priority_score >= 35:
        priority_indicator = " [yellow][MEDIUM PRIORITY][/yellow]"

    # Display ambiguity context with priority (Story 4.4)
    panel_content = (
        f"[bold]{amb.ambiguity_type.value.title()}[/bold] "
        f"([{amb.severity.value}] severity){priority_indicator}\n\n"
        f"[yellow]Source text:[/yellow] \"{amb.source_text}\"\n"
        f"[yellow]Location:[/yellow] {amb.location}\n"
        f"[yellow]Issue:[/yellow] {amb.description}"
    )
    console.print(Panel(panel_content, title=f"Ambiguity #{index}", border_style="yellow"))

    # Show question
    console.print(f"\n[bold]{prompt.question}[/bold]")

    # Show format hint if available (Story 4.4)
    if prompt.format_hint:
        console.print(f"[cyan]Expected format:[/cyan] {prompt.format_hint}")

    # Build choices list
    choices = list(prompt.suggestions) if prompt.suggestions else []
    choices.append("skip")

    # Show suggestions
    if prompt.suggestions:
        console.print("[dim]Suggestions:[/dim]")
        for i, suggestion in enumerate(prompt.suggestions, 1):
            console.print(f"  [{i}] {suggestion}")
        console.print("  [s] Skip this ambiguity")

    # Get user input
    response = Prompt.ask(
        "\n[bold]Your answer (number, 's' to skip, or custom text)[/bold]",
        default="s",
    )

    # Handle skip
    if response.lower() in ("s", "skip", ""):
        console.print("[dim]Skipped[/dim]")
        return None

    # Handle numbered choice
    if response.isdigit() and prompt.suggestions:
        idx = int(response) - 1
        if 0 <= idx < len(prompt.suggestions):
            response = prompt.suggestions[idx]
            console.print(f"[green]Selected:[/green] {response}")

    # Format-specific validation (Story 4.4)
    validated_response = _validate_format_response(response, prompt)
    if validated_response is None:
        return None

    # Create resolution
    return Resolution(
        ambiguity_id=f"amb-{index}",
        user_response=validated_response,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _validate_format_response(response: str, prompt: ResolutionPrompt) -> str | None:
    """Validate user response based on answer format (Story 4.4).

    Args:
        response: The user's response text.
        prompt: The resolution prompt with format info.

    Returns:
        Validated response (possibly normalized), or None if invalid and user skips.
    """
    import re

    # Validation pattern takes precedence if defined
    if prompt.validation_pattern:
        if not re.match(prompt.validation_pattern, response):
            console.print(
                f"[yellow]Warning:[/yellow] Response doesn't match expected pattern. "
                f"[dim]({prompt.format_hint or 'see format hint'})[/dim]"
            )
            retry = Prompt.ask(
                "[bold]Keep this response anyway?[/bold] (y/n)",
                default="y",
            )
            if retry.lower() != "y":
                console.print("[dim]Skipped[/dim]")
                return None

    # Format-specific validation
    if prompt.answer_format == AnswerFormat.BOOLEAN:
        normalized = response.lower().strip()
        if normalized in ("yes", "y", "true", "1"):
            return "yes"
        elif normalized in ("no", "n", "false", "0"):
            return "no"
        else:
            console.print(
                "[yellow]Note:[/yellow] Expected yes/no answer. "
                f"Keeping original: '{response}'"
            )

    elif prompt.answer_format == AnswerFormat.NUMERIC:
        # Try to extract a number
        try:
            # Handle numbers with commas like "1,000"
            cleaned = response.replace(",", "")
            float(cleaned)  # Just validate it's a number
        except ValueError:
            console.print(
                "[yellow]Note:[/yellow] Expected numeric answer. "
                f"Keeping original: '{response}'"
            )

    elif prompt.answer_format == AnswerFormat.DATE:
        # Basic date format check (YYYY-MM-DD)
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", response):
            console.print(
                "[yellow]Note:[/yellow] Expected date format (YYYY-MM-DD). "
                f"Keeping original: '{response}'"
            )

    return response


async def _detect_ambiguities_async(content: str) -> AmbiguityResult:
    """Async wrapper for detect_ambiguities.

    Args:
        content: The seed document content.

    Returns:
        The ambiguity detection result.
    """
    return await detect_ambiguities(content)


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


async def _parse_seed_async(
    content: str,
    filename: str,
    detect_ambiguities_flag: bool = False,
) -> SeedParseResult:
    """Async wrapper for parse_seed.

    Args:
        content: The seed document content.
        filename: The original filename for format detection.
        detect_ambiguities_flag: Whether to run ambiguity detection.

    Returns:
        The parsed seed result.
    """
    return await parse_seed(
        content,
        filename=filename,
        detect_ambiguities=detect_ambiguities_flag,
    )


def _apply_resolutions_to_content(
    content: str,
    ambiguity_result: AmbiguityResult,
    resolutions: list[Resolution],
) -> str:
    """Apply user resolutions to seed content.

    Appends clarifications to the end of the content as a "Clarifications" section.

    Args:
        content: Original seed content.
        ambiguity_result: The detected ambiguities.
        resolutions: User-provided resolutions.

    Returns:
        Modified content with clarifications appended.
    """
    if not resolutions:
        return content

    # Build clarification section
    clarification_lines = ["\n\n## Clarifications (User-Provided)\n"]

    for resolution in resolutions:
        # Find matching ambiguity
        amb_idx = int(resolution.ambiguity_id.split("-")[1]) - 1
        if 0 <= amb_idx < len(ambiguity_result.ambiguities):
            amb = ambiguity_result.ambiguities[amb_idx]
            clarification_lines.append(
                f"- **{amb.source_text}**: {resolution.user_response}\n"
            )

    return content + "".join(clarification_lines)


def seed_command(
    file_path: Path,
    verbose: bool = False,
    json_output: bool = False,
    interactive: bool = False,
) -> None:
    """Parse a seed document and display structured results.

    Reads a natural language seed document, parses it into structured
    components (goals, features, constraints), and displays the results.
    In interactive mode, detects ambiguities and prompts for resolution.

    Args:
        file_path: Path to the seed document file.
        verbose: If True, show additional details in output.
        json_output: If True, output results as JSON instead of tables.
        interactive: If True, detect ambiguities and prompt for resolution.
    """
    logger.info(
        "seed_command_started",
        file_path=str(file_path),
        verbose=verbose,
        json_output=json_output,
        interactive=interactive,
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

    # Interactive mode: detect ambiguities first
    resolutions: list[Resolution] = []
    if interactive:
        console.print("[blue]Detecting ambiguities...[/blue]")
        console.print()

        try:
            ambiguity_result = asyncio.run(_detect_ambiguities_async(content))
            logger.info(
                "ambiguity_detection_complete",
                ambiguity_count=len(ambiguity_result.ambiguities),
            )

            # Display ambiguities
            _display_ambiguities(ambiguity_result, verbose=verbose)

            # Prompt for resolutions if ambiguities found
            if ambiguity_result.has_ambiguities:
                console.print()
                console.print(
                    "[bold]Would you like to clarify these ambiguities?[/bold] "
                    "[dim](This will re-parse with your clarifications)[/dim]"
                )
                console.print()

                # Sort ambiguities by priority before prompting (Story 4.4)
                sorted_ambiguities = prioritize_questions(list(ambiguity_result.ambiguities))

                # Build a mapping from sorted ambiguities to their original prompts
                amb_to_prompt = dict(
                    zip(ambiguity_result.ambiguities, ambiguity_result.resolution_prompts, strict=False)
                )

                for i, amb in enumerate(sorted_ambiguities, 1):
                    prompt = amb_to_prompt.get(amb)
                    if prompt is None:
                        logger.warning(
                            "ambiguity_missing_prompt",
                            ambiguity_source=amb.source_text[:50],
                            index=i,
                        )
                        continue

                    # Calculate priority score for display (Story 4.4)
                    priority_score = calculate_question_priority(amb)

                    console.print()
                    resolution = _prompt_for_resolution(amb, prompt, i, priority_score)
                    if resolution:
                        resolutions.append(resolution)

                # Apply resolutions to content
                if resolutions:
                    console.print()
                    console.print(
                        f"[green]Applied {len(resolutions)} clarification(s)[/green]"
                    )
                    content = _apply_resolutions_to_content(
                        content, ambiguity_result, resolutions
                    )
                    console.print()
                    console.print("[blue]Re-parsing with clarifications...[/blue]")
                    console.print()

                # Show unresolved ambiguities count
                unresolved = len(ambiguity_result.ambiguities) - len(resolutions)
                if unresolved > 0:
                    console.print(
                        f"[yellow]Note:[/yellow] {unresolved} ambiguity(ies) "
                        f"were not resolved."
                    )
                    console.print()

        except Exception as e:
            logger.error("ambiguity_detection_failed", error=str(e))
            console.print(
                f"[yellow]Warning:[/yellow] Ambiguity detection failed: {e!s}\n"
                f"[dim]Continuing with normal parsing...[/dim]"
            )
            console.print()

    # Parse the seed content (with ambiguity detection if not interactive)
    try:
        # In interactive mode, we already did ambiguity detection separately
        # In normal mode with verbose, we can show ambiguities inline
        detect_flag = verbose and not interactive
        result = asyncio.run(
            _parse_seed_async(content, file_path.name, detect_flag)
        )
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

        # Show ambiguity summary in verbose mode (non-interactive)
        if verbose and not interactive and result.has_ambiguities:
            console.print()
            console.print(
                f"[yellow]Ambiguities:[/yellow] {result.ambiguity_count} detected "
                f"(confidence: {result.ambiguity_confidence:.0%})"
            )
            console.print("[dim]Use --interactive mode to resolve ambiguities.[/dim]")

        console.print()
        console.print(
            "[green]Seed parsing complete![/green] "
            "[dim]Use --json for machine-readable output.[/dim]"
        )

    logger.info("seed_command_completed", file_path=str(file_path))
