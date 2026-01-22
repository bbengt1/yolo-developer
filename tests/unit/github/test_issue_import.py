from __future__ import annotations

from datetime import datetime, timezone

import pytest

from yolo_developer.github.config import GitHubImportConfig
from yolo_developer.github.issue_import import IssueImporter
from yolo_developer.github.issue_parser import IssueParser
from yolo_developer.github.models import GitHubIssueInput, StoryPriority


class FakeGitHubClient:
    def __init__(self, issue: dict, comments: list[dict] | None = None) -> None:
        self.repo = "owner/repo"
        self._issue = issue
        self._comments = comments or []

    def api(self, method: str, endpoint: str, data: dict | None = None):
        if endpoint.endswith("/issues/42") and method == "GET":
            return self._issue
        if endpoint.endswith("/issues/42/comments") and method == "GET":
            return self._comments
        if endpoint.endswith("/issues") and method == "GET":
            return [self._issue]
        return {}


def test_issue_parser_extracts_sections() -> None:
    issue = GitHubIssueInput(
        number=42,
        title="Add avatars",
        body="## Requirements\n- Upload PNG\n- Resize to 200x200\n\n## Acceptance Criteria\n- [ ] User can upload\n- [ ] Images are resized",
        labels=["feature", "p1"],
        state="open",
        author="tester",
        created_at=datetime.now(timezone.utc),
        comments=[],
        linked_issues=[],
        milestone=None,
        assignees=[],
        url="https://example.com/issues/42",
    )
    parser = IssueParser()
    parsed = parser.parse(issue)
    assert parsed.requirements
    assert len(parsed.acceptance_criteria) == 2
    assert parsed.priority == StoryPriority.HIGH


@pytest.mark.asyncio
async def test_issue_importer_generates_story() -> None:
    issue = {
        "number": 42,
        "title": "Add avatars",
        "body": "## Requirements\n- Upload PNG\n- Resize to 200x200\n",
        "labels": [{"name": "feature"}],
        "state": "open",
        "user": {"login": "tester"},
        "created_at": "2025-01-01T00:00:00Z",
        "assignees": [],
        "html_url": "https://example.com/issues/42",
    }
    importer = IssueImporter(
        FakeGitHubClient(issue),
        GitHubImportConfig(update_issues=False, add_comment=False, add_label=None),
    )
    result = await importer.import_issue(issue_number=42, preview=True)
    assert result.stories_generated
    story = result.stories_generated[0]
    assert story.id.endswith("0042")
    assert story.title
