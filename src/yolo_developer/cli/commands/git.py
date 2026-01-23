"""Git operations CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from yolo_developer.github.git import GitManager

console = Console()
app = typer.Typer(help="Git operations")


@app.command("status")
def git_status(path: str | None = None) -> None:
    repo = GitManager(Path(path) if path else Path.cwd())
    status = repo.status()
    branch = status["branch"]
    console.print(f"Branch: {branch.name} (ahead {branch.ahead}, behind {branch.behind})")
    table = Table(title="Git Status")
    table.add_column("Category")
    table.add_column("Files")
    table.add_row("Staged", ", ".join(status["staged"]) or "-")
    table.add_row("Modified", ", ".join(status["modified"]) or "-")
    table.add_row("Untracked", ", ".join(status["untracked"]) or "-")
    console.print(table)


@app.command("commit")
def git_commit(
    message: str = typer.Option(..., "-m", "--message"),
    files: list[str] = typer.Option(None, "--file"),
    path: str | None = None,
) -> None:
    repo = GitManager(Path(path) if path else Path.cwd())
    if files:
        repo.stage_files(files)
    else:
        status = repo.status()
        modified = status["modified"]
        if not modified:
            console.print("[yellow]No tracked changes to commit.[/yellow]")
            raise typer.Exit(code=1)
        repo.stage_files(modified)
    result = repo.commit(message)
    console.print(f"Committed {result.sha}: {result.message}")


@app.command("push")
def git_push(
    branch: str | None = typer.Option(None, "--branch"),
    force: bool = typer.Option(False, "--force"),
    path: str | None = None,
) -> None:
    repo = GitManager(Path(path) if path else Path.cwd())
    repo.push(branch=branch, force=force, set_upstream=branch is None)
    console.print("Pushed to remote.")


@app.command("branch")
def git_branch(
    name: str = typer.Argument(...),
    base: str = typer.Option("main", "--base"),
    path: str | None = None,
) -> None:
    repo = GitManager(Path(path) if path else Path.cwd())
    branch = repo.create_branch(name, base=base)
    console.print(f"Created branch {branch.name}")


@app.command("checkout")
def git_checkout(branch: str, path: str | None = None) -> None:
    repo = GitManager(Path(path) if path else Path.cwd())
    info = repo.checkout(branch)
    console.print(f"Checked out {info.name}")


__all__ = ["app"]
