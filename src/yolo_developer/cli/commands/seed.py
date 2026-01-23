"""CLI command for parsing seed documents (Story 4.2, 4.3, 4.5, 4.6, 4.7).

This module implements the `yolo seed` command that parses natural language
seed documents into structured components (goals, features, constraints).
It also supports interactive ambiguity resolution via --interactive flag,
SOP constraint validation via --validate-sop flag, semantic validation
reports via --report-format flag (Story 4.6), and quality threshold rejection
via automatic quality checks (Story 4.7).

Example:
    $ yolo seed requirements.md
    $ yolo seed requirements.md --verbose
    $ yolo seed requirements.md --json
    $ yolo seed requirements.md --interactive
    $ yolo seed requirements.md --validate-sop
    $ yolo seed requirements.md --validate-sop --override-soft
    $ yolo seed requirements.md --report-format json
    $ yolo seed requirements.md --report-format markdown --report-output report.md
    $ yolo seed requirements.md --report-format rich --force  # bypass threshold rejection
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from yolo_developer.seed import SeedParseResult, parse_seed
from yolo_developer.seed.utils import get_api_key_for_model
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
from yolo_developer.seed.rejection import (
    QualityThreshold,
    RejectionResult,
    create_rejection_with_remediation,
)
from yolo_developer.seed.report import (
    format_report_json,
    format_report_markdown,
    format_report_rich,
    generate_validation_report,
)
from yolo_developer.seed.sop import (
    ConflictSeverity,
    InMemorySOPStore,
    SOPConflict,
    SOPConstraint,
    SOPValidationResult,
)

if TYPE_CHECKING:
    from yolo_developer.seed.sop import SOPStore

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
            "[green]No ambiguities detected![/green] [dim]The seed document appears clear.[/dim]"
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
        "green"
        if ambiguity_result.overall_confidence >= 0.8
        else "yellow"
        if ambiguity_result.overall_confidence >= 0.5
        else "red"
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


def _display_sop_conflicts(
    sop_result: SOPValidationResult,
    verbose: bool = False,
) -> None:
    """Display SOP validation conflicts in a Rich table (Story 4.5).

    Args:
        sop_result: The SOP validation result to display.
        verbose: If True, show additional details like resolution options.
    """
    if not sop_result.has_conflicts:
        console.print(
            "[green]No SOP conflicts detected![/green] "
            "[dim]The seed document is compatible with all constraints.[/dim]"
        )
        return

    # SOP Conflicts table
    table = Table(
        title="SOP Conflicts Detected",
        show_header=True,
        header_style="bold red",
    )
    table.add_column("#", justify="right", style="dim")
    table.add_column("Severity", justify="center")
    table.add_column("Category", style="cyan")
    table.add_column("Rule", width=30)
    table.add_column("Conflict", width=30)

    for i, conflict in enumerate(sop_result.conflicts, 1):
        # Style based on severity
        severity_label = (
            "[red bold]HARD[/red bold]"
            if conflict.severity == ConflictSeverity.HARD
            else "[yellow]SOFT[/yellow]"
        )

        table.add_row(
            str(i),
            severity_label,
            conflict.constraint.category.value.title(),
            (
                conflict.constraint.rule_text[:30] + "..."
                if len(conflict.constraint.rule_text) > 30
                else conflict.constraint.rule_text
            ),
            (
                conflict.description[:30] + "..."
                if len(conflict.description) > 30
                else conflict.description
            ),
        )

    console.print(table)

    # Summary
    console.print()
    console.print(
        f"[bold]HARD conflicts:[/bold] [red]{sop_result.hard_conflict_count}[/red] "
        f"(blocks processing)"
    )
    console.print(
        f"[bold]SOFT conflicts:[/bold] [yellow]{sop_result.soft_conflict_count}[/yellow] "
        f"(can override)"
    )

    # Verbose: show detailed conflict info
    if verbose:
        console.print()
        console.print("[dim]Conflict Details (Verbose):[/dim]")
        for i, conflict in enumerate(sop_result.conflicts, 1):
            severity_color = "red" if conflict.severity == ConflictSeverity.HARD else "yellow"
            console.print(
                f"\n  [{severity_color}]{i}. {conflict.severity.value.upper()}[/{severity_color}]"
            )
            console.print(f"     [bold]Rule:[/bold] {conflict.constraint.rule_text}")
            console.print(f"     [bold]Source:[/bold] {conflict.constraint.source}")
            console.print(f'     [bold]Seed text:[/bold] "{conflict.seed_text}"')
            console.print(f"     [bold]Description:[/bold] {conflict.description}")
            if conflict.resolution_options:
                console.print("     [bold]Resolution options:[/bold]")
                for opt in conflict.resolution_options:
                    console.print(f"       - {opt}")


def _display_rejection(rejection_result: RejectionResult) -> None:
    """Display rejection panel with failed thresholds and remediation (Story 4.7).

    Args:
        rejection_result: The RejectionResult containing failure details.
    """
    # Build panel content with failed thresholds
    content_lines = ["[red bold]Seed Rejected - Quality Below Threshold[/red bold]\n"]

    # Show failed thresholds
    content_lines.append("[red]Failed Thresholds:[/red]")
    for reason in rejection_result.reasons:
        content_lines.append(
            f"  [red]x[/red] {reason.threshold_name.title()} Score: "
            f"{reason.actual_score:.2f} < {reason.required_score:.2f} required"
        )

    # Show remediation recommendations
    if rejection_result.recommendations:
        content_lines.append("\n[yellow]Remediation Steps:[/yellow]")
        for i, rec in enumerate(rejection_result.recommendations, 1):
            content_lines.append(f"  {i}. {rec}")

    # Add tip about --force
    content_lines.append("\n[dim]Tip: Use --force to proceed despite low quality scores[/dim]")

    panel = Panel(
        "\n".join(content_lines),
        title="Seed Rejected",
        border_style="red",
    )
    console.print()
    console.print(panel)


def _display_threshold_warning(rejection_result: RejectionResult) -> None:
    """Display warning when --force bypasses threshold rejection (Story 4.7).

    Args:
        rejection_result: The RejectionResult containing failure details.
    """
    # Build warning content
    content_lines = ["[yellow bold]Quality Threshold Bypassed with --force[/yellow bold]\n"]

    # Show which thresholds would have failed
    content_lines.append("[yellow]Thresholds that would have rejected:[/yellow]")
    for reason in rejection_result.reasons:
        content_lines.append(
            f"  [yellow]![/yellow] {reason.threshold_name.title()} Score: "
            f"{reason.actual_score:.2f} < {reason.required_score:.2f} required"
        )

    # Show remediation as recommendations for future
    if rejection_result.recommendations:
        content_lines.append("\n[dim]Consider addressing before production use:[/dim]")
        for rec in rejection_result.recommendations:
            content_lines.append(f"  - {rec}")

    panel = Panel(
        "\n".join(content_lines),
        title="Warning: Quality Bypass",
        border_style="yellow",
    )
    console.print(panel)


def _prompt_for_sop_override(
    conflict: SOPConflict,
    index: int,
) -> bool:
    """Prompt user to override a SOFT SOP conflict (Story 4.5).

    Args:
        conflict: The SOFT conflict to potentially override.
        index: The conflict index (1-based).

    Returns:
        True if user chose to override, False otherwise.
    """
    # Display conflict details
    panel_content = (
        f"[bold yellow]SOFT Conflict[/bold yellow] - Can be overridden\n\n"
        f"[bold]Rule:[/bold] {conflict.constraint.rule_text}\n"
        f"[bold]Category:[/bold] {conflict.constraint.category.value.title()}\n"
        f"[bold]Source:[/bold] {conflict.constraint.source}\n\n"
        f'[yellow]Conflicting text:[/yellow] "{conflict.seed_text}"\n'
        f"[yellow]Why it conflicts:[/yellow] {conflict.description}"
    )

    if conflict.resolution_options:
        panel_content += "\n\n[bold]Resolution options:[/bold]"
        for opt in conflict.resolution_options:
            panel_content += f"\n  - {opt}"

    console.print(Panel(panel_content, title=f"SOP Conflict #{index}", border_style="yellow"))

    # Prompt for override
    response = Prompt.ask(
        "\n[bold]Override this conflict?[/bold] (y/n)",
        default="n",
    )

    return response.lower() in ("y", "yes")


def _load_sop_store(store_path: Path | None) -> SOPStore:
    """Load SOP store from path or return empty in-memory store (Story 4.5).

    Args:
        store_path: Optional path to SOP store file (JSON format).

    Returns:
        SOPStore instance with loaded constraints.
    """
    store = InMemorySOPStore()

    if store_path is None:
        logger.debug("using_empty_sop_store")
        return store

    if not store_path.exists():
        logger.warning("sop_store_file_not_found", path=str(store_path))
        console.print(
            f"[yellow]Warning:[/yellow] SOP store file not found: {store_path}\n"
            f"[dim]Using empty constraint store.[/dim]"
        )
        return store

    try:
        import json as json_module

        content = store_path.read_text(encoding="utf-8")
        data = json_module.loads(content)
        constraints = data.get("constraints", [])

        # Load constraints into store - use internal dict directly for sync loading
        for constraint_data in constraints:
            constraint = SOPConstraint.from_dict(constraint_data)
            # Access internal dict directly to avoid async in sync context
            store._constraints[constraint.id] = constraint

        logger.info(
            "sop_store_loaded",
            path=str(store_path),
            constraint_count=len(constraints),
        )
        console.print(
            f"[blue]Loaded {len(constraints)} SOP constraint(s) from:[/blue] {store_path}"
        )

    except json.JSONDecodeError as e:
        logger.error("sop_store_parse_error", path=str(store_path), error=str(e))
        console.print(
            f"[red]Error:[/red] Failed to parse SOP store: {store_path}\n[dim]{e!s}[/dim]"
        )
    except Exception as e:
        logger.error("sop_store_load_error", path=str(store_path), error=str(e))
        console.print(f"[red]Error:[/red] Failed to load SOP store: {store_path}\n[dim]{e!s}[/dim]")

    return store


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
        f'[yellow]Source text:[/yellow] "{amb.source_text}"\n'
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
                f"[yellow]Note:[/yellow] Expected yes/no answer. Keeping original: '{response}'"
            )

    elif prompt.answer_format == AnswerFormat.NUMERIC:
        # Try to extract a number
        try:
            # Handle numbers with commas like "1,000"
            cleaned = response.replace(",", "")
            float(cleaned)  # Just validate it's a number
        except ValueError:
            console.print(
                f"[yellow]Note:[/yellow] Expected numeric answer. Keeping original: '{response}'"
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


def _persist_seed_state(result: SeedParseResult) -> None:
    """Persist seed parse result for downstream commands like `yolo run`."""
    seed_state_path = Path(".yolo") / "seed_state.json"
    seed_state_path.parent.mkdir(parents=True, exist_ok=True)
    seed_state_path.write_text(
        json.dumps(result.to_dict(), indent=2),
        encoding="utf-8",
    )


def _load_thresholds_from_config() -> QualityThreshold:
    """Load quality thresholds from config if available (Story 4.7).

    Attempts to load YoloConfig from the current directory's yolo.yaml.
    Falls back to default thresholds if config is not available.

    Returns:
        QualityThreshold configured from yolo.yaml or defaults.
    """
    try:
        from yolo_developer.config import ConfigurationError, load_config

        config = load_config()
        seed_config = config.quality.seed_thresholds
        return QualityThreshold(
            overall=seed_config.overall,
            ambiguity=seed_config.ambiguity,
            sop=seed_config.sop,
        )
    except (FileNotFoundError, ConfigurationError) as e:
        # Config file not found or invalid - use defaults
        logger.debug("config_load_failed_using_defaults", error=str(e))
        return QualityThreshold()
    except (ImportError, AttributeError) as e:
        # Missing config module or schema changes - use defaults
        logger.warning("config_module_error_using_defaults", error=str(e))
        return QualityThreshold()


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
    validate_sop_flag: bool = False,
    sop_store: SOPStore | None = None,
) -> SeedParseResult:
    """Async wrapper for parse_seed.

    Args:
        content: The seed document content.
        filename: The original filename for format detection.
        detect_ambiguities_flag: Whether to run ambiguity detection.
        validate_sop_flag: Whether to run SOP validation (Story 4.5).
        sop_store: SOP store for validation (Story 4.5).

    Returns:
        The parsed seed result.
    """
    try:
        from yolo_developer.config import ConfigurationError, load_config

        config = load_config()
    except (FileNotFoundError, ConfigurationError):
        from yolo_developer.config import YoloConfig

        config = YoloConfig(project_name="seed")

    primary_model = config.llm.cheap_model
    api_key = get_api_key_for_model(primary_model, config.llm)

    result = await parse_seed(
        content,
        filename=filename,
        model=primary_model,
        api_key=api_key,
        detect_ambiguities=detect_ambiguities_flag,
        validate_sop=validate_sop_flag,
        sop_store=sop_store,
    )
    metadata = dict(result.metadata)
    fallback_model = config.llm.premium_model
    if "error" in metadata and fallback_model != primary_model:
        fallback_key = get_api_key_for_model(fallback_model, config.llm)
        logger.warning(
            "seed_parse_retrying_with_fallback",
            primary_model=primary_model,
            fallback_model=fallback_model,
        )
        result = await parse_seed(
            content,
            filename=filename,
            model=fallback_model,
            api_key=fallback_key,
            detect_ambiguities=detect_ambiguities_flag,
            validate_sop=validate_sop_flag,
            sop_store=sop_store,
        )
    return result


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
            clarification_lines.append(f"- **{amb.source_text}**: {resolution.user_response}\n")

    return content + "".join(clarification_lines)


def seed_command(
    file_path: Path,
    verbose: bool = False,
    json_output: bool = False,
    interactive: bool = False,
    validate_sop: bool = False,
    sop_store_path: Path | None = None,
    override_soft: bool = False,
    report_format: str | None = None,
    report_output: Path | None = None,
    force: bool = False,
) -> None:
    """Parse a seed document and display structured results.

    Reads a natural language seed document, parses it into structured
    components (goals, features, constraints), and displays the results.
    In interactive mode, detects ambiguities and prompts for resolution.
    With --validate-sop, validates against SOP constraints.
    With --report-format, generates semantic validation reports (Story 4.6).
    With threshold enforcement, rejects low-quality seeds (Story 4.7).

    Args:
        file_path: Path to the seed document file.
        verbose: If True, show additional details in output.
        json_output: If True, output results as JSON instead of tables.
        interactive: If True, detect ambiguities and prompt for resolution.
        validate_sop: If True, validate against SOP constraints (Story 4.5).
        sop_store_path: Path to SOP store JSON file (Story 4.5).
        override_soft: If True, auto-override all SOFT conflicts (Story 4.5).
        report_format: Output format for validation report (json, markdown, rich).
        report_output: File path to write report to (optional).
        force: If True, bypass quality threshold rejection (Story 4.7).
    """
    logger.info(
        "seed_command_started",
        file_path=str(file_path),
        verbose=verbose,
        json_output=json_output,
        interactive=interactive,
        validate_sop=validate_sop,
        override_soft=override_soft,
        report_format=report_format,
        report_output=str(report_output) if report_output else None,
        force=force,
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
                    zip(
                        ambiguity_result.ambiguities,
                        ambiguity_result.resolution_prompts,
                        strict=False,
                    )
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
                    console.print(f"[green]Applied {len(resolutions)} clarification(s)[/green]")
                    content = _apply_resolutions_to_content(content, ambiguity_result, resolutions)
                    console.print()
                    console.print("[blue]Re-parsing with clarifications...[/blue]")
                    console.print()

                # Show unresolved ambiguities count
                unresolved = len(ambiguity_result.ambiguities) - len(resolutions)
                if unresolved > 0:
                    console.print(
                        f"[yellow]Note:[/yellow] {unresolved} ambiguity(ies) were not resolved."
                    )
                    console.print()

        except Exception as e:
            logger.error("ambiguity_detection_failed", error=str(e))
            console.print(
                f"[yellow]Warning:[/yellow] Ambiguity detection failed: {e!s}\n"
                f"[dim]Continuing with normal parsing...[/dim]"
            )
            console.print()

    # Load SOP store if validation requested (Story 4.5)
    sop_store: SOPStore | None = None
    if validate_sop:
        if not json_output:
            console.print("[blue]Loading SOP constraints...[/blue]")
        sop_store = _load_sop_store(sop_store_path)
        if not json_output:
            console.print()

    # Parse the seed content (with ambiguity detection if not interactive)
    try:
        # In interactive mode, we already did ambiguity detection separately
        # In normal mode with verbose, we can show ambiguities inline
        detect_flag = verbose and not interactive
        result = asyncio.run(
            _parse_seed_async(
                content,
                file_path.name,
                detect_flag,
                validate_sop_flag=validate_sop,
                sop_store=sop_store,
            )
        )
        logger.info(
            "seed_parsing_success",
            goals=result.goal_count,
            features=result.feature_count,
            constraints=result.constraint_count,
            sop_conflicts=(
                result.sop_validation.hard_conflict_count
                + result.sop_validation.soft_conflict_count
                if result.sop_validation
                else 0
            ),
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

    # Handle SOP conflicts (Story 4.5)
    sop_blocked = False
    if validate_sop and result.sop_validation:
        if not json_output:
            console.print()
            _display_sop_conflicts(result.sop_validation, verbose=verbose)

        # Check for HARD conflicts - these block processing
        if result.sop_validation.hard_conflict_count > 0:
            if not json_output:
                console.print()
                console.print(
                    "[red bold]BLOCKED:[/red bold] Seed cannot proceed due to "
                    f"{result.sop_validation.hard_conflict_count} HARD conflict(s).\n"
                    "[dim]Resolve these conflicts before continuing.[/dim]"
                )
            sop_blocked = True

        # Handle SOFT conflicts - offer override option
        elif result.sop_validation.soft_conflict_count > 0:
            if override_soft:
                # Auto-override all SOFT conflicts
                if not json_output:
                    console.print()
                    console.print(
                        f"[yellow]Auto-overriding {result.sop_validation.soft_conflict_count} "
                        f"SOFT conflict(s) (--override-soft)[/yellow]"
                    )
                # Log override decision
                for conflict in result.sop_validation.soft_conflicts:
                    logger.info(
                        "sop_soft_conflict_overridden",
                        constraint_id=conflict.constraint.id,
                        rule=conflict.constraint.rule_text[:50],
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        auto_override=True,
                    )
                result.sop_validation.override_applied = True
            elif not json_output:
                # Interactive override prompt for SOFT conflicts
                console.print()
                console.print("[bold]Would you like to override SOFT conflicts?[/bold]")
                console.print()

                overrides_applied = 0
                for i, conflict in enumerate(result.sop_validation.soft_conflicts, 1):
                    if _prompt_for_sop_override(conflict, i):
                        overrides_applied += 1
                        logger.info(
                            "sop_soft_conflict_overridden",
                            constraint_id=conflict.constraint.id,
                            rule=conflict.constraint.rule_text[:50],
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            auto_override=False,
                        )
                    console.print()

                if overrides_applied > 0:
                    console.print(f"[green]Overrode {overrides_applied} SOFT conflict(s)[/green]")
                    result.sop_validation.override_applied = True

    # Exit if blocked by HARD conflicts
    if sop_blocked:
        if json_output:
            _output_json(result)
        raise typer.Exit(code=1)

    _persist_seed_state(result)

    # Generate validation report if requested (Story 4.6)
    if report_format:
        report = generate_validation_report(result, source_file=str(file_path))
        logger.info(
            "validation_report_generated",
            report_id=report.report_id,
            overall_score=report.quality_metrics.overall_score,
            format=report_format,
        )

        # Check quality thresholds (Story 4.7)
        thresholds = _load_thresholds_from_config()
        rejection_result = create_rejection_with_remediation(
            report.quality_metrics, report, thresholds
        )

        if not rejection_result.passed:
            if force:
                # User bypassed rejection with --force
                logger.warning(
                    "quality_threshold_rejection_bypassed",
                    force=True,
                    failure_count=rejection_result.failure_count,
                    reasons=[r.to_dict() for r in rejection_result.reasons],
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                if not json_output:
                    console.print()
                    _display_threshold_warning(rejection_result)
            else:
                # Display rejection and exit
                logger.warning(
                    "quality_threshold_rejection",
                    passed=False,
                    failure_count=rejection_result.failure_count,
                    reasons=[r.to_dict() for r in rejection_result.reasons],
                )
                if not json_output:
                    _display_rejection(rejection_result)
                raise typer.Exit(code=1)

        # Generate formatted output
        if report_format == "json":
            formatted_output = format_report_json(report)
        elif report_format == "markdown":
            formatted_output = format_report_markdown(report)
        elif report_format == "rich":
            # Rich output goes directly to console
            formatted_output = None
        else:
            console.print(f"[red]Error:[/red] Unknown report format: {report_format}")
            raise typer.Exit(code=1)

        # Write to file or display
        if report_output:
            try:
                if report_format == "rich":
                    # For rich format, write markdown fallback to file
                    report_output.write_text(format_report_markdown(report), encoding="utf-8")
                else:
                    report_output.write_text(formatted_output or "", encoding="utf-8")
                console.print(f"[green]Report written to:[/green] {report_output}")
                logger.info("report_written_to_file", path=str(report_output))
            except OSError as e:
                console.print(f"[red]Error writing report:[/red] {e!s}")
                raise typer.Exit(code=1) from e
        else:
            if report_format == "rich":
                format_report_rich(report, console=console)
            elif report_format == "json":
                console.print_json(formatted_output or "")
            else:
                console.print(formatted_output or "")
    elif json_output:
        # Legacy --json output (just the parse result)
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

        # Show SOP summary
        if validate_sop and result.sop_validation:
            console.print()
            if result.sop_validation.passed:
                console.print("[green]SOP validation passed![/green]")
            elif result.sop_validation.override_applied:
                console.print("[yellow]SOP validation passed with overrides[/yellow]")

        console.print()
        console.print(
            "[green]Seed parsing complete![/green] "
            "[dim]Use --json for machine-readable output or --report-format for validation reports.[/dim]"
        )

    logger.info("seed_command_completed", file_path=str(file_path))
