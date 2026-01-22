from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
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


class IssueType(str, Enum):
    FEATURE = "feature"
    BUG = "bug"
    ENHANCEMENT = "enhancement"
    TASK = "task"
    EPIC = "epic"
    UNKNOWN = "unknown"


class StoryPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class GitHubIssueInput:
    number: int
    title: str
    body: str
    labels: list[str]
    state: str
    author: str
    created_at: datetime
    comments: list[dict[str, Any]]
    linked_issues: list[int]
    milestone: str | None
    assignees: list[str]
    url: str


@dataclass(frozen=True)
class ParsedIssue:
    issue: GitHubIssueInput
    issue_type: IssueType
    objective: str
    requirements: list[str]
    acceptance_criteria: list[str]
    technical_notes: str
    dependencies: list[str]
    priority: StoryPriority


@dataclass(frozen=True)
class ExtractedRequirement:
    id: str
    description: str
    type: str
    priority: StoryPriority
    source: str
    acceptance_criteria: list[str]
    assumptions: list[str]
    dependencies: list[str]
    confidence: float


@dataclass(frozen=True)
class GeneratedStory:
    id: str
    title: str
    description: str
    type: IssueType
    priority: StoryPriority
    acceptance_criteria: list[str]
    technical_notes: str
    estimation_points: int | None
    github_issue: int
    dependencies: list[str]
    tags: list[str]


@dataclass(frozen=True)
class ImportResult:
    issues_processed: int
    stories_generated: list[GeneratedStory]
    requirements_extracted: list[ExtractedRequirement]
    warnings: list[str]
    errors: list[str]
    ready_for_sprint: bool


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
