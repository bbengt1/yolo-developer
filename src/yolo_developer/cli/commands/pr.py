"""Pull request CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from yolo_developer.config import load_config
from yolo_developer.github.client import GitHubClient
from yolo_developer.github.git import GitManager
from yolo_developer.github.pr import PRManager

console = Console()
app = typer.Typer(help="Pull request operations")


def _manager(path: Path) -> PRManager:
    config = load_config(path / "yolo.yaml")
    repo = config.github.repository or GitManager(path).get_repo_slug()
    if not repo:
        raise RuntimeError("GitHub repository not configured")
    token = config.github.token.get_secret_value() if config.github.token else None
    client = GitHubClient(repo=repo, token=token, cwd=path)
    return PRManager(client)


@app.command("create")
def pr_create(
    title: str = typer.Option(..., "--title"),
    body: str = typer.Option(..., "--body"),
    head: str | None = typer.Option(None, "--head"),
    base: str = typer.Option("main", "--base"),
    draft: bool = typer.Option(False, "--draft"),
    path: str | None = None,
) -> None:
    repo_path = Path(path) if path else Path.cwd()
    manager = _manager(repo_path)
    if head is None:
        head = GitManager(repo_path).get_current_branch().name
    pr = manager.create(title=title, body=body, head=head, base=base, draft=draft)
    console.print(f"Created PR #{pr.number}: {pr.url}")


@app.command("merge")
def pr_merge(
    number: int = typer.Argument(...),
    method: str = typer.Option("squash", "--method"),
    path: str | None = None,
) -> None:
    repo_path = Path(path) if path else Path.cwd()
    manager = _manager(repo_path)
    result = manager.merge(pr_number=number, method=method)
    console.print(f"Merge result: {result.get('message', '')}")


__all__ = ["app"]
