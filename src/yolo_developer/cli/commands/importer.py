"""CLI commands for importing GitHub issues."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from yolo_developer.cli.display import error_panel
from yolo_developer.github.issue_import import IssueImporter, render_seed_markdown
from yolo_developer.github.models import GitHubError

app = typer.Typer(name="import", help="Import GitHub issues as user stories")
console = Console()


@app.command("issue")
def import_issue(
    issue_number: int = typer.Argument(..., help="GitHub issue number"),
    repo: str | None = typer.Option(None, "--repo", help="Repository owner/repo"),
    auto_seed: bool = typer.Option(False, "--auto-seed", help="Write seed file"),
    preview: bool = typer.Option(False, "--preview", help="Preview only"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output file"),
    fmt: str = typer.Option("markdown", "--format", help="Output format: markdown/json"),
) -> None:
    """Import a single GitHub issue."""
    try:
        importer = IssueImporter.from_config()
        result = asyncio.run(
            importer.import_issue(
                issue_number=issue_number,
                repo=repo,
                auto_seed=auto_seed,
                preview=preview,
            )
        )
    except (GitHubError, RuntimeError) as exc:
        error_panel(str(exc))
        raise typer.Exit(code=1) from exc

    if result.errors:
        for error in result.errors:
            console.print(f"[red]Error: {error}[/red]")
        raise typer.Exit(code=1)

    story = result.stories_generated[0]
    _display_story(story)

    if output:
        _export_story(story, output, fmt)
        console.print(f"[green]Story exported to {output}[/green]")

    if result.warnings:
        for warning in result.warnings:
            console.print(f"[yellow]{warning}[/yellow]")


@app.command("issues")
def import_issues(
    numbers: list[int] = typer.Argument(None, help="Issue numbers"),
    label: list[str] | None = typer.Option(None, "--label", "-l", help="Filter by label"),
    milestone: str | None = typer.Option(None, "--milestone", "-m", help="Filter by milestone"),
    query: str | None = typer.Option(None, "--query", "-q", help="GitHub search query"),
    auto_seed: bool = typer.Option(False, "--auto-seed", help="Write seed file"),
    preview: bool = typer.Option(False, "--preview", help="Preview only"),
) -> None:
    """Import multiple GitHub issues."""
    try:
        importer = IssueImporter.from_config()
        result = asyncio.run(
            importer.import_multiple(
                issue_numbers=numbers,
                labels=label,
                milestone=milestone,
                query=query,
                auto_seed=auto_seed,
                preview=preview,
            )
        )
    except (GitHubError, RuntimeError) as exc:
        error_panel(str(exc))
        raise typer.Exit(code=1) from exc

    _display_summary(result)
    for story in result.stories_generated:
        _display_story(story, compact=True)

    if result.errors:
        raise typer.Exit(code=1)


@app.command("preview")
def preview_issue(
    issue_number: int = typer.Argument(..., help="GitHub issue number"),
    repo: str | None = typer.Option(None, "--repo", help="Repository owner/repo"),
) -> None:
    """Preview seed output for an issue."""
    try:
        importer = IssueImporter.from_config()
        preview = importer.preview(issue_number=issue_number, repo=repo)
    except (GitHubError, RuntimeError) as exc:
        error_panel(str(exc))
        raise typer.Exit(code=1) from exc
    console.print(Panel(preview.seed_markdown, title=f"Issue #{issue_number} Preview"))


def _display_story(story, compact: bool = False) -> None:
    title = f"{story.id}: {story.title}"
    if compact:
        console.print(f"[green]{title}[/green] ({story.priority})")
        return

    table = Table(show_header=False)
    table.add_row("Type", story.type.value)
    table.add_row("Priority", story.priority.value)
    if story.estimation_points:
        table.add_row("Points", str(story.estimation_points))

    console.print(Panel(story.description, title=title))
    console.print(table)
    if story.acceptance_criteria:
        console.print("[bold]Acceptance Criteria[/bold]")
        for ac in story.acceptance_criteria:
            console.print(f"- {ac}")
    if story.technical_notes:
        console.print("\n[bold]Technical Notes[/bold]")
        console.print(story.technical_notes)


def _display_summary(result) -> None:
    table = Table(title="Import Summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Issues processed", str(result.issues_processed))
    table.add_row("Stories generated", str(len(result.stories_generated)))
    table.add_row("Requirements extracted", str(len(result.requirements_extracted)))
    table.add_row("Errors", str(len(result.errors)))
    console.print(table)


def _export_story(story, output: Path, fmt: str) -> None:
    if fmt.lower() == "json":
        payload = {
            "id": story.id,
            "title": story.title,
            "description": story.description,
            "priority": story.priority.value,
            "acceptance_criteria": story.acceptance_criteria,
            "technical_notes": story.technical_notes,
            "github_issue": story.github_issue,
            "tags": story.tags,
        }
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        output.write_text(render_seed_markdown([story]), encoding="utf-8")
