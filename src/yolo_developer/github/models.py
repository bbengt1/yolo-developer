from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


class GitHubError(RuntimeError):
    """Errors raised during GitHub operations."""


@dataclass(frozen=True)
class CommitResult:
    sha: str
    message: str
    files_changed: list[str]
    insertions: int
    deletions: int


@dataclass(frozen=True)
class BranchInfo:
    name: str
    upstream: str | None
    ahead: int
    behind: int
    is_current: bool


@dataclass(frozen=True)
class PullRequest:
    number: int
    title: str
    body: str
    state: str
    head_branch: str
    head_sha: str
    base_branch: str
    url: str
    review_status: str
    mergeable: bool
    checks_passing: bool


@dataclass(frozen=True)
class Issue:
    number: int
    title: str
    body: str
    state: str
    labels: list[str]
    assignees: list[str]
    milestone: str | None
    url: str
    linked_prs: list[int]


@dataclass(frozen=True)
class Release:
    tag: str
    name: str
    body: str
    draft: bool
    prerelease: bool
    url: str
    assets: list[str]
    created_at: datetime | None


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def label_names(labels: list[dict[str, Any]] | None) -> list[str]:
    if not labels:
        return []
    return [label.get("name", "") for label in labels if label.get("name")]
