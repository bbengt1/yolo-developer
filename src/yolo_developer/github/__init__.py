"""GitHub integration for YOLO Developer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from yolo_developer.github.client import GitHubClient  # noqa: F401
    from yolo_developer.github.config import GitHubConfig  # noqa: F401
    from yolo_developer.github.git import GitManager  # noqa: F401
    from yolo_developer.github.issue_import import IssueImporter  # noqa: F401
    from yolo_developer.github.issue_parser import IssueParser  # noqa: F401
    from yolo_developer.github.issues import IssueManager  # noqa: F401
    from yolo_developer.github.models import (  # noqa: F401
        BranchInfo,
        CommitResult,
        ExtractedRequirement,
        GeneratedStory,
        GitHubError,
        GitHubIssueInput,
        ImportResult,
        Issue,
        IssueType,
        ParsedIssue,
        PullRequest,
        Release,
        StoryPriority,
    )
    from yolo_developer.github.pr import PRManager  # noqa: F401
    from yolo_developer.github.requirement_extractor import RequirementExtractor  # noqa: F401
    from yolo_developer.github.releases import ReleaseManager  # noqa: F401
    from yolo_developer.github.story_generator import StoryGenerator  # noqa: F401
    from yolo_developer.github.workflows import GitHubWorkflow, StoryInfo  # noqa: F401

__all__ = [
    "BranchInfo",
    "CommitResult",
    "ExtractedRequirement",
    "GeneratedStory",
    "GitHubClient",
    "GitHubConfig",
    "GitHubError",
    "GitHubIssueInput",
    "GitManager",
    "GitHubWorkflow",
    "ImportResult",
    "Issue",
    "IssueImporter",
    "IssueParser",
    "IssueType",
    "IssueManager",
    "ParsedIssue",
    "PRManager",
    "PullRequest",
    "Release",
    "ReleaseManager",
    "RequirementExtractor",
    "StoryGenerator",
    "StoryPriority",
    "StoryInfo",
]


def __getattr__(name: str) -> Any:
    if name in {"GitHubClient", "GitHubConfig"}:
        from yolo_developer.github import config, client

        return getattr(client if name == "GitHubClient" else config, name)
    if name in {"GitManager"}:
        from yolo_developer.github import git

        return getattr(git, name)
    if name in {"IssueManager"}:
        from yolo_developer.github import issues

        return getattr(issues, name)
    if name in {"PRManager"}:
        from yolo_developer.github import pr

        return getattr(pr, name)
    if name in {"ReleaseManager"}:
        from yolo_developer.github import releases

        return getattr(releases, name)
    if name in {"GitHubWorkflow", "StoryInfo"}:
        from yolo_developer.github import workflows

        return getattr(workflows, name)
    if name in {
        "IssueImporter",
        "IssueParser",
        "RequirementExtractor",
        "StoryGenerator",
    }:
        from yolo_developer.github import issue_import, issue_parser, requirement_extractor, story_generator

        module_map = {
            "IssueImporter": issue_import,
            "IssueParser": issue_parser,
            "RequirementExtractor": requirement_extractor,
            "StoryGenerator": story_generator,
        }
        return getattr(module_map[name], name)
    if name in {
        "BranchInfo",
        "CommitResult",
        "ExtractedRequirement",
        "GeneratedStory",
        "GitHubError",
        "GitHubIssueInput",
        "ImportResult",
        "Issue",
        "IssueType",
        "ParsedIssue",
        "PullRequest",
        "Release",
        "StoryPriority",
    }:
        from yolo_developer.github import models

        return getattr(models, name)
    raise AttributeError(f"module 'yolo_developer.github' has no attribute {name}")
