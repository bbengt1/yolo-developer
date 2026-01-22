"""GitHub integration for YOLO Developer."""

from __future__ import annotations

from yolo_developer.github.client import GitHubClient
from yolo_developer.github.config import GitHubConfig
from yolo_developer.github.git import GitManager
from yolo_developer.github.issue_import import IssueImporter
from yolo_developer.github.issue_parser import IssueParser
from yolo_developer.github.issues import IssueManager
from yolo_developer.github.models import (
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
from yolo_developer.github.pr import PRManager
from yolo_developer.github.requirement_extractor import RequirementExtractor
from yolo_developer.github.releases import ReleaseManager
from yolo_developer.github.story_generator import StoryGenerator
from yolo_developer.github.workflows import GitHubWorkflow, StoryInfo

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
