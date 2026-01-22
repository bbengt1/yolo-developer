"""Import GitHub issues and convert them into user stories."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re

from yolo_developer.config import load_config
from yolo_developer.github.client import GitHubClient
from yolo_developer.github.config import GitHubImportConfig
from yolo_developer.github.git import GitManager
from yolo_developer.github.issue_parser import IssueParser
from yolo_developer.github.models import (
    GitHubIssueInput,
    ImportResult,
)
from yolo_developer.github.requirement_extractor import RequirementExtractor
from yolo_developer.github.story_generator import StoryGenerator


@dataclass(frozen=True)
class IssueImportPreview:
    issue: GitHubIssueInput
    seed_markdown: str


class IssueImporter:
    """Orchestrate GitHub issue to user story conversion."""

    def __init__(self, client: GitHubClient, config: GitHubImportConfig) -> None:
        self.client = client
        self.config = config
        self.parser = IssueParser(config.templates)
        self.extractor = RequirementExtractor()
        self.generator = StoryGenerator(config.story)

    @classmethod
    def from_config(cls) -> "IssueImporter":
        config = load_config()
        repo_path = Path.cwd()
        git = GitManager(repo_path)
        repo_slug = config.github.repository or git.get_repo_slug()
        if not repo_slug:
            raise ValueError("GitHub repository not configured")
        if not config.github.import_config.enabled:
            raise ValueError("GitHub issue import is disabled in configuration")
        token = config.github.token.get_secret_value() if config.github.token else None
        client = GitHubClient(repo=repo_slug, token=token, cwd=repo_path)
        return cls(client, config.github.import_config)

    async def import_issue(
        self,
        issue_number: int,
        repo: str | None = None,
        auto_seed: bool = False,
        preview: bool = False,
    ) -> ImportResult:
        repo_slug = repo or self.config.default_repo or self.client.repo
        issue_input = self._fetch_issue(repo_slug, issue_number)
        parsed = self.parser.parse(issue_input)
        requirements = self.extractor.extract(parsed)
        story = await self.generator.generate(
            issue=issue_input,
            requirements=requirements,
            issue_type=parsed.issue_type,
            priority=parsed.priority,
            technical_notes=parsed.technical_notes,
        )

        warnings: list[str] = []
        errors: list[str] = []

        if not preview and self.config.update_issues:
            self._annotate_issue(repo_slug, story, issue_input)

        if auto_seed and not preview:
            seed_markdown = render_seed_markdown([story])
            _write_seed_file(seed_markdown, issue_number)
            warnings.append("Seed content written to .yolo/imported-issues/.")

        return ImportResult(
            issues_processed=1,
            stories_generated=[story],
            requirements_extracted=requirements,
            warnings=warnings,
            errors=errors,
            ready_for_sprint=not preview,
        )

    async def import_multiple(
        self,
        issue_numbers: list[int] | None = None,
        labels: list[str] | None = None,
        milestone: str | None = None,
        query: str | None = None,
        auto_seed: bool = False,
        preview: bool = False,
    ) -> ImportResult:
        repo_slug = self.config.default_repo or self.client.repo
        issues = self._fetch_issues(repo_slug, issue_numbers, labels, milestone, query)

        stories = []
        requirements = []
        warnings: list[str] = []
        errors: list[str] = []

        for issue in issues:
            try:
                parsed = self.parser.parse(issue)
                reqs = self.extractor.extract(parsed)
                story = await self.generator.generate(
                    issue=issue,
                    requirements=reqs,
                    issue_type=parsed.issue_type,
                    priority=parsed.priority,
                    technical_notes=parsed.technical_notes,
                )
                stories.append(story)
                requirements.extend(reqs)
                if not preview and self.config.update_issues:
                    self._annotate_issue(repo_slug, story, issue)
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(f"Issue #{issue.number}: {exc}")

        if auto_seed and not preview and stories:
            seed_markdown = render_seed_markdown(stories)
            _write_seed_file(seed_markdown, "batch")
            warnings.append("Seed content written to .yolo/imported-issues/.")

        return ImportResult(
            issues_processed=len(issues),
            stories_generated=stories,
            requirements_extracted=requirements,
            warnings=warnings,
            errors=errors,
            ready_for_sprint=not preview,
        )

    def preview(self, issue_number: int, repo: str | None = None) -> IssueImportPreview:
        repo_slug = repo or self.config.default_repo or self.client.repo
        issue_input = self._fetch_issue(repo_slug, issue_number)
        parsed = self.parser.parse(issue_input)
        requirements = self.extractor.extract(parsed)
        seed_markdown = render_seed_markdown(
            [
                self._preview_story_stub(
                    issue_input,
                    requirements,
                )
            ]
        )
        return IssueImportPreview(issue=issue_input, seed_markdown=seed_markdown)

    def _fetch_issue(self, repo_slug: str, issue_number: int) -> GitHubIssueInput:
        data = self.client.api("GET", f"repos/{repo_slug}/issues/{issue_number}")
        comments = self.client.api(
            "GET", f"repos/{repo_slug}/issues/{issue_number}/comments"
        )
        return _to_issue_input(data, comments if isinstance(comments, list) else [])

    def _fetch_issues(
        self,
        repo_slug: str,
        issue_numbers: list[int] | None,
        labels: list[str] | None,
        milestone: str | None,
        query: str | None,
    ) -> list[GitHubIssueInput]:
        issues: list[GitHubIssueInput] = []
        if issue_numbers:
            for number in issue_numbers:
                issues.append(self._fetch_issue(repo_slug, number))
            return issues

        if query:
            q = query
            if f"repo:{repo_slug}" not in q:
                q = f"repo:{repo_slug} {q}"
            results = self.client.api("GET", "search/issues", data={"q": q})
            for item in results.get("items", []):
                issues.append(_to_issue_input(item, []))
            return issues

        params: dict[str, str] = {"state": "open"}
        if labels:
            params["labels"] = ",".join(labels)
        if milestone:
            params["milestone"] = milestone
        data = self.client.api("GET", f"repos/{repo_slug}/issues", data=params)
        for item in data if isinstance(data, list) else []:
            issues.append(_to_issue_input(item, []))
        return issues

    def _annotate_issue(
        self,
        repo_slug: str,
        story: object,
        issue: GitHubIssueInput,
    ) -> None:
        if not self.config.add_comment:
            return
        comment = _format_issue_comment(story)
        self.client.api(
            "POST",
            f"repos/{repo_slug}/issues/{issue.number}/comments",
            data={"body": comment},
        )
        if self.config.add_label:
            self.client.api(
                "POST",
                f"repos/{repo_slug}/issues/{issue.number}/labels",
                data={"labels": [self.config.add_label]},
            )

    def _preview_story_stub(
        self,
        issue: GitHubIssueInput,
        requirements: list[object],
    ):
        return {
            "id": f"{self.config.story.id_prefix}-{issue.number:04d}",
            "title": issue.title,
            "description": issue.body[:200] if issue.body else issue.title,
            "acceptance_criteria": [
                req.description
                for req in requirements
                if hasattr(req, "description")
            ],
            "technical_notes": issue.body[:200] if issue.body else "",
            "github_issue": issue.number,
        }


def render_seed_markdown(stories: list[object]) -> str:
    lines = ["# Imported GitHub Issues", ""]
    for story in stories:
        story_id = getattr(story, "id", "US-0000")
        title = getattr(story, "title", "Imported Story")
        description = getattr(story, "description", "")
        acs = getattr(story, "acceptance_criteria", [])
        notes = getattr(story, "technical_notes", "")
        lines.append(f"## {story_id}: {title}")
        if description:
            lines.append(description)
            lines.append("")
        if acs:
            lines.append("### Acceptance Criteria")
            for ac in acs:
                lines.append(f"- {ac}")
            lines.append("")
        if notes:
            lines.append("### Technical Notes")
            lines.append(notes)
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def _format_issue_comment(story: object) -> str:
    story_id = getattr(story, "id", "US-0000")
    priority = getattr(story, "priority", "")
    return (
        "## 🤖 YOLO Developer Story Generated\n\n"
        f"**Story ID:** `{story_id}`\n"
        f"**Priority:** {priority}\n\n"
        "*Generated by YOLO Developer issue import.*\n"
    )


def _to_issue_input(raw: dict, comments: list[dict]) -> GitHubIssueInput:
    created_at = raw.get("created_at")
    created = (
        datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        if created_at
        else datetime.now(timezone.utc)
    )
    return GitHubIssueInput(
        number=raw.get("number", 0),
        title=raw.get("title", ""),
        body=raw.get("body", "") or "",
        labels=[label.get("name", "") for label in raw.get("labels", []) if label.get("name")],
        state=raw.get("state", "open"),
        author=raw.get("user", {}).get("login", "unknown"),
        created_at=created,
        comments=comments,
        linked_issues=_extract_issue_refs(raw.get("body", "") or ""),
        milestone=(raw.get("milestone") or {}).get("title"),
        assignees=[a.get("login", "") for a in raw.get("assignees", []) if a.get("login")],
        url=raw.get("html_url", ""),
    )


def _extract_issue_refs(body: str) -> list[int]:
    return sorted({int(match) for match in re.findall(r"#(\\d+)", body)})


def _write_seed_file(content: str, suffix: str | int) -> None:
    directory = Path(".yolo/imported-issues")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"issue-import-{suffix}.md"
    path.write_text(content, encoding="utf-8")
