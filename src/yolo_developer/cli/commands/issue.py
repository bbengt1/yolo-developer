"""Issue CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from yolo_developer.config import load_config
from yolo_developer.github.client import GitHubClient
from yolo_developer.github.git import GitManager
from yolo_developer.github.issues import IssueManager

console = Console()
app = typer.Typer(help="Issue operations")


def _manager(path: Path) -> IssueManager:
    config = load_config(path / "yolo.yaml")
    repo = config.github.repository or GitManager(path).get_repo_slug()
    if not repo:
        raise RuntimeError("GitHub repository not configured")
    token = config.github.token.get_secret_value() if config.github.token else None
    client = GitHubClient(repo=repo, token=token, cwd=path)
    return IssueManager(client)


@app.command("create")
def issue_create(
    title: str = typer.Option(..., "--title"),
    body: str = typer.Option(..., "--body"),
    path: str | None = None,
) -> None:
    manager = _manager(Path(path) if path else Path.cwd())
    issue = manager.create(title=title, body=body)
    console.print(f"Created issue #{issue.number}: {issue.url}")


@app.command("close")
def issue_close(
    number: int = typer.Argument(...),
    comment: str | None = typer.Option(None, "--comment"),
    path: str | None = None,
) -> None:
    manager = _manager(Path(path) if path else Path.cwd())
    issue = manager.close(issue_number=number, comment=comment)
    console.print(f"Closed issue #{issue.number}")


__all__ = ["app"]
