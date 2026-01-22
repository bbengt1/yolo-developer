"""Scan an existing project for brownfield context."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from yolo_developer.config.schema import BrownfieldConfig
from yolo_developer.scanner import ScannerManager

console = Console()


def scan_command(
    path: str | None = None,
    scan_depth: int | None = None,
    max_files: int | None = None,
    include_git_history: bool | None = None,
    interactive: bool = False,
    hint: str | None = None,
    refresh: bool = False,
    write_context: bool = True,
) -> None:
    """Scan a project and optionally write .yolo/project-context.yaml."""
    project_path = Path(path) if path else Path.cwd()
    project_path = project_path.resolve()

    config = BrownfieldConfig()
    depth = scan_depth or config.scan_depth
    max_files_to_analyze = max_files or config.max_files_to_analyze
    include_git = config.include_git_history if include_git_history is None else include_git_history

    manager = ScannerManager()
    report = manager.scan(
        project_path=project_path,
        scan_depth=depth,
        exclude_patterns=config.exclude_patterns,
        max_files=max_files_to_analyze,
        include_git_history=include_git,
        hint=hint,
    )

    _print_scan_report(report)

    if not write_context:
        console.print("[green]Scan complete (no context written).[/green]")
        return

    output_path = project_path / ".yolo" / "project-context.yaml"
    if output_path.exists() and not refresh:
        console.print(
            "[yellow]project-context.yaml already exists. Use --refresh to overwrite.[/yellow]"
        )
        return

    context = manager.build_project_context(
        report=report,
        project_name=project_path.name,
        interactive=interactive and config.interactive,
        console=console,
    )
    output_path = manager.write_project_context(project_path, context)
    console.print(f"[green]Generated project context:[/green] {output_path}")


def _print_scan_report(report) -> None:
    table = Table(title="Brownfield Scan Summary")
    table.add_column("Finding")
    table.add_column("Value")
    table.add_column("Confidence", justify="right")

    for finding in report.findings:
        if finding.key in {"docs", "git", "conventions"}:
            continue
        table.add_row(finding.key, str(finding.value), f"{finding.confidence:.0%}")

    console.print(table)

    if report.suggestions:
        console.print("[yellow]Suggestions:[/yellow]")
        for suggestion in report.suggestions:
            console.print(f"  - {suggestion}")
