"""GitHub workflow automation commands."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from yolo_developer.config import load_config
from yolo_developer.github.client import GitHubClient
from yolo_developer.github.git import GitManager
from yolo_developer.github.issues import IssueManager
from yolo_developer.github.pr import PRManager
from yolo_developer.github.releases import ReleaseManager
from yolo_developer.github.workflows import GitHubWorkflow, StoryInfo, format_pr_body

console = Console()
app = typer.Typer(help="GitHub workflow automation")


def _workflow(path: Path) -> GitHubWorkflow:
    config = load_config(path / "yolo.yaml")
    repo = config.github.repository or GitManager(path).get_repo_slug()
    if not repo:
        raise RuntimeError("GitHub repository not configured")
    token = config.github.token.get_secret_value() if config.github.token else None
    client = GitHubClient(repo=repo, token=token, cwd=path)
    git = GitManager(path)
    prs = PRManager(client)
    issues = IssueManager(client)
    releases = ReleaseManager(client)
    return GitHubWorkflow(git=git, prs=prs, issues=issues, releases=releases)


@app.command("start")
def workflow_start(
    story_id: str = typer.Argument(...),
    title: str = typer.Option(..., "--title"),
    description: str = typer.Option("", "--description"),
    criteria: list[str] = typer.Option(None, "--criteria"),
    path: str | None = None,
) -> None:
    workflow = _workflow(Path(path) if path else Path.cwd())
    story = StoryInfo(
        story_id=story_id,
        title=title,
        description=description,
        acceptance_criteria=criteria or [],
    )
    result = workflow.start_story(story, branch_prefix="feature/")
    console.print(f"Created branch {result['branch'].name} and issue #{result['issue'].number}")


@app.command("complete")
def workflow_complete(
    story_id: str = typer.Argument(...),
    title: str = typer.Option(..., "--title"),
    description: str = typer.Option("", "--description"),
    criteria: list[str] = typer.Option(None, "--criteria"),
    commit_message: str = typer.Option("chore: update", "--commit"),
    files: list[str] = typer.Option(None, "--file"),
    path: str | None = None,
) -> None:
    repo_path = Path(path) if path else Path.cwd()
    workflow = _workflow(repo_path)
    story = StoryInfo(
        story_id=story_id,
        title=title,
        description=description,
        acceptance_criteria=criteria or [],
    )
    git = GitManager(repo_path)
    files_changed = files or git.status()["modified"]
    result = workflow.complete_story(
        story=story,
        files_changed=files_changed,
        commit_message=commit_message,
        pr_body="",
        base_branch="main",
    )
    pr_body = format_pr_body(story, result["commit"])
    if pr_body:
        workflow.prs.update(result["pr"].number, body=pr_body)
    console.print(f"Created PR #{result['pr'].number}")


__all__ = ["app"]
