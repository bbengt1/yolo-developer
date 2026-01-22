"""Release CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from yolo_developer.config import load_config
from yolo_developer.github.client import GitHubClient
from yolo_developer.github.git import GitManager
from yolo_developer.github.releases import ReleaseManager

console = Console()
app = typer.Typer(help="Release operations")


def _manager(path: Path) -> ReleaseManager:
    config = load_config(path / "yolo.yaml")
    repo = config.github.repository or GitManager(path).get_repo_slug()
    if not repo:
        raise RuntimeError("GitHub repository not configured")
    token = config.github.token.get_secret_value() if config.github.token else None
    client = GitHubClient(repo=repo, token=token, cwd=path)
    return ReleaseManager(client)


@app.command("create")
def release_create(
    tag: str = typer.Option(..., "--tag"),
    name: str = typer.Option(..., "--name"),
    body: str = typer.Option("", "--body"),
    target: str = typer.Option("main", "--target"),
    draft: bool = typer.Option(False, "--draft"),
    prerelease: bool = typer.Option(False, "--prerelease"),
    path: str | None = None,
) -> None:
    manager = _manager(Path(path) if path else Path.cwd())
    release = manager.create(
        tag=tag,
        name=name,
        body=body,
        target=target,
        draft=draft,
        prerelease=prerelease,
        generate_notes=True,
    )
    console.print(f"Created release {release.tag}: {release.url}")


__all__ = ["app"]
