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


class GitHubImportStoryConfig(BaseModel):
    id_prefix: str = Field(default="US", description="Story ID prefix")
    include_technical_notes: bool = Field(
        default=True, description="Include technical notes in story output"
    )
    estimate_points: bool = Field(default=True, description="Estimate story points")
    default_priority: str = Field(default="medium", description="Default story priority")


class GitHubImportRequirementConfig(BaseModel):
    extract_nfr: bool = Field(default=True, description="Extract non-functional requirements")
    min_confidence: float = Field(default=0.7, description="Minimum confidence threshold")


class GitHubImportTemplateConfig(BaseModel):
    feature: list[str] = Field(default_factory=list, description="Feature template markers")
    bug: list[str] = Field(default_factory=list, description="Bug template markers")
    enhancement: list[str] = Field(default_factory=list, description="Enhancement template markers")
    custom: list[str] = Field(default_factory=list, description="Custom template markers")


class GitHubImportBatchConfig(BaseModel):
    max_issues: int = Field(default=20, description="Maximum issues to import per batch")
    detect_dependencies: bool = Field(default=True, description="Detect dependencies")
    group_by_epic: bool = Field(default=False, description="Group by epic labels")


class GitHubImportConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable issue import")
    default_repo: str | None = Field(default=None, description="Default repo for imports")
    update_issues: bool = Field(default=True, description="Update issues with story info")
    add_label: str | None = Field(default="yolo-imported", description="Label to add after import")
    add_comment: bool = Field(default=True, description="Add comment with story summary")
    story: GitHubImportStoryConfig = Field(default_factory=GitHubImportStoryConfig)
    requirements: GitHubImportRequirementConfig = Field(
        default_factory=GitHubImportRequirementConfig
    )
    templates: GitHubImportTemplateConfig = Field(default_factory=GitHubImportTemplateConfig)
    batch: GitHubImportBatchConfig = Field(default_factory=GitHubImportBatchConfig)


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
    import_config: GitHubImportConfig = Field(default_factory=GitHubImportConfig)
