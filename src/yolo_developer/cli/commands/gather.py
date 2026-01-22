"""CLI commands for interactive requirements gathering."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from yolo_developer.analyst import SessionManager
from yolo_developer.analyst.models import SessionPhase

app = typer.Typer(name="gather", help="Interactive requirements gathering")
console = Console()


@app.command("start")
def start(
    project_name: str = typer.Argument(..., help="Project name"),
    description: str | None = typer.Option(None, "--description", help="Initial description"),
    project_type: str | None = typer.Option(None, "--type", help="Project type hint"),
    resume: str | None = typer.Option(None, "--resume", help="Resume session ID"),
    non_interactive: bool = typer.Option(False, "--non-interactive", help="Print question only"),
) -> None:
    manager = SessionManager.from_config()
    if resume:
        session = manager.resume_session(resume)
    else:
        session = manager.start_session(project_name, description, project_type)

    console.print(
        Panel(
            f"Session ID: {session.id}\nProject: {session.project_name}\nPhase: {session.phase.value}",
            title="Requirements Gathering",
        )
    )

    question = manager.get_current_question(session.id)
    if question is None:
        console.print("[green]No questions remaining.[/green]")
        return

    if non_interactive:
        console.print(f"Question: {question.text}")
        return

    while question:
        console.print(f"\n[bold]Question[/bold] ({question.phase.value}): {question.text}")
        response = Prompt.ask("Your response")
        if response.strip().lower() in {"quit", "exit"}:
            console.print("[yellow]Session saved. Resume with --resume.[/yellow]")
            break
        result = manager.process_response(session.id, response)
        if result.new_requirements:
            console.print("[green]Extracted requirements:[/green]")
            for req in result.new_requirements:
                console.print(f"  - {req.description}")
        if result.is_complete:
            console.print("[green]Session complete. Export with `yolo gather export`.[/green]")
            break
        question = manager.get_current_question(session.id)


@app.command("list")
def list_sessions() -> None:
    manager = SessionManager.from_config()
    sessions = manager.list_sessions()
    if not sessions:
        console.print("[yellow]No sessions found.[/yellow]")
        return
    for session in sessions:
        console.print(
            f"{session.id} | {session.project_name} | {session.phase.value} | "
            f"{session.requirements_count} requirements"
        )


@app.command("export")
def export_session(
    session_id: str = typer.Argument(..., help="Session ID"),
    fmt: str = typer.Option("markdown", "--format", "-f", help="Export format"),
    output: str | None = typer.Option(None, "--output", "-o", help="Output file"),
) -> None:
    manager = SessionManager.from_config()
    document = manager.export_requirements(session_id, format=fmt)
    if output:
        path = output
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(document)
        console.print(f"[green]Exported to {path}[/green]")
        return
    console.print(Panel(document, title=f"Session {session_id}"))
