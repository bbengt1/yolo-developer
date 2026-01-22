from __future__ import annotations

from pydantic import BaseModel, Field, SecretStr


class GitHubAutomationConfig(BaseModel):
    auto_commit: bool = Field(default=False, description="Auto-commit changes")
    auto_push: bool = Field(default=False, description="Auto-push after commit")
    auto_pr: bool = Field(default=False, description="Auto-create PRs")
    auto_merge: bool = Field(default=False, description="Auto-merge PRs")
    auto_release: bool = Field(default=False, description="Auto-create releases")
    respond_to_reviews: bool = Field(default=True, description="Respond to PR reviews")


class GitHubPullRequestConfig(BaseModel):
    draft_by_default: bool = Field(default=False, description="Open PRs as drafts")
    require_reviews: int = Field(default=1, description="Required review count")
    merge_method: str = Field(default="squash", description="merge, squash, rebase")
    delete_branch_on_merge: bool = Field(default=True, description="Delete branch")
    labels: list[str] = Field(default_factory=lambda: ["yolo-generated"], description="PR labels")


class GitHubIssueConfig(BaseModel):
    create_from_stories: bool = Field(default=True, description="Create issues from stories")
    close_on_pr_merge: bool = Field(default=True, description="Close issues on merge")
    labels: list[str] = Field(default_factory=lambda: ["yolo-story"], description="Issue labels")


class GitHubReleaseConfig(BaseModel):
    versioning: str = Field(default="semver", description="semver, calver, manual")
    generate_notes: bool = Field(default=True, description="Auto-generate release notes")
    create_on_sprint_complete: bool = Field(
        default=False, description="Create release on sprint complete"
    )


class GitHubCommitConfig(BaseModel):
    sign: bool = Field(default=False, description="Sign commits")
    co_author: str = Field(default="YOLO Developer <yolo@example.com>", description="Co-author")
    conventional: bool = Field(default=True, description="Use conventional commits")


class GitHubConfig(BaseModel):
    enabled: bool = Field(default=False, description="Enable GitHub automation")
    token: SecretStr | None = Field(default=None, description="GitHub token (env only)")
    repository: str | None = Field(default=None, description="owner/repo override")
    default_branch: str = Field(default="main", description="Default branch")
    branch_prefix: str = Field(default="feature/", description="Branch prefix")
    automation: GitHubAutomationConfig = Field(default_factory=GitHubAutomationConfig)
    pull_requests: GitHubPullRequestConfig = Field(default_factory=GitHubPullRequestConfig)
    issues: GitHubIssueConfig = Field(default_factory=GitHubIssueConfig)
    releases: GitHubReleaseConfig = Field(default_factory=GitHubReleaseConfig)
    commits: GitHubCommitConfig = Field(default_factory=GitHubCommitConfig)
