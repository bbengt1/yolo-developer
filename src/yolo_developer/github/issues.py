from __future__ import annotations

from typing import Any

from yolo_developer.github.client import GitHubClient
from yolo_developer.github.models import Issue, label_names


class IssueManager:
    """Manage GitHub issues."""

    def __init__(self, client: GitHubClient) -> None:
        self.client = client

    def create(
        self,
        title: str,
        body: str,
        labels: list[str] | None = None,
        assignees: list[str] | None = None,
        milestone: str | None = None,
    ) -> Issue:
        payload: dict[str, Any] = {"title": title, "body": body}
        if labels:
            payload["labels"] = labels
        if assignees:
            payload["assignees"] = assignees
        if milestone:
            payload["milestone"] = milestone
        data = self.client.api("POST", f"repos/{self.client.repo}/issues", payload)
        return self._to_issue(data)

    def update(
        self,
        issue_number: int,
        title: str | None = None,
        body: str | None = None,
        state: str | None = None,
        labels: list[str] | None = None,
    ) -> Issue:
        payload: dict[str, Any] = {}
        if title:
            payload["title"] = title
        if body:
            payload["body"] = body
        if state:
            payload["state"] = state
        if labels is not None:
            payload["labels"] = labels
        data = self.client.api("PATCH", f"repos/{self.client.repo}/issues/{issue_number}", payload)
        return self._to_issue(data)

    def close(self, issue_number: int, comment: str | None = None, reason: str = "completed") -> Issue:
        if comment:
            self.add_comment(issue_number, comment)
        data = self.client.api(
            "PATCH",
            f"repos/{self.client.repo}/issues/{issue_number}",
            {"state": "closed", "state_reason": reason},
        )
        return self._to_issue(data)

    def add_comment(self, issue_number: int, body: str) -> None:
        self.client.api(
            "POST",
            f"repos/{self.client.repo}/issues/{issue_number}/comments",
            {"body": body},
        )

    def link_to_pr(self, issue_number: int, pr_number: int) -> None:
        self.add_comment(issue_number, f"Linked to PR #{pr_number}")

    def search(self, query: str, state: str = "open", labels: list[str] | None = None) -> list[Issue]:
        q = f"repo:{self.client.repo} is:issue state:{state} {query}"
        if labels:
            q += " " + " ".join(f"label:{label}" for label in labels)
        data = self.client.api("GET", f"search/issues?q={q}")
        items = data.get("items", []) if isinstance(data, dict) else []
        return [self._to_issue(item) for item in items]

    def _to_issue(self, data: dict[str, Any]) -> Issue:
        return Issue(
            number=int(data.get("number", 0)),
            title=data.get("title", ""),
            body=data.get("body", ""),
            state=data.get("state", ""),
            labels=label_names(data.get("labels")),
            assignees=[assignee.get("login", "") for assignee in data.get("assignees", [])],
            milestone=data.get("milestone", {}).get("title") if data.get("milestone") else None,
            url=data.get("html_url", ""),
            linked_prs=[],
        )
