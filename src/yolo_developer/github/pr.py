from __future__ import annotations

from typing import Any

from yolo_developer.github.client import GitHubClient
from yolo_developer.github.models import PullRequest


class PRManager:
    """Manage GitHub pull requests."""

    def __init__(self, client: GitHubClient) -> None:
        self.client = client

    def create(
        self,
        title: str,
        body: str,
        head: str,
        base: str = "main",
        draft: bool = False,
        reviewers: list[str] | None = None,
        labels: list[str] | None = None,
        issue_refs: list[int] | None = None,
    ) -> PullRequest:
        if issue_refs:
            refs = "\n".join(f"Closes #{number}" for number in issue_refs)
            body = f"{body}\n\n---\n{refs}"

        data = self.client.api(
            "POST",
            f"repos/{self.client.repo}/pulls",
            {
                "title": title,
                "body": body,
                "head": head,
                "base": base,
                "draft": str(draft).lower(),
            },
        )
        pr = self._to_pull_request(data)

        if reviewers:
            self.client.api(
                "POST",
                f"repos/{self.client.repo}/pulls/{pr.number}/requested_reviewers",
                {"reviewers": reviewers},
            )
        if labels:
            self.client.api(
                "POST",
                f"repos/{self.client.repo}/issues/{pr.number}/labels",
                {"labels": labels},
            )

        return pr

    def update(
        self,
        pr_number: int,
        title: str | None = None,
        body: str | None = None,
        state: str | None = None,
    ) -> PullRequest:
        payload: dict[str, Any] = {}
        if title:
            payload["title"] = title
        if body:
            payload["body"] = body
        if state:
            payload["state"] = state
        data = self.client.api(
            "PATCH",
            f"repos/{self.client.repo}/pulls/{pr_number}",
            payload,
        )
        return self._to_pull_request(data)

    def merge(
        self,
        pr_number: int,
        method: str = "squash",
        commit_title: str | None = None,
        commit_message: str | None = None,
        delete_branch: bool = True,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"merge_method": method}
        if commit_title:
            payload["commit_title"] = commit_title
        if commit_message:
            payload["commit_message"] = commit_message
        result = self.client.api(
            "PUT",
            f"repos/{self.client.repo}/pulls/{pr_number}/merge",
            payload,
        )

        if delete_branch and result.get("merged"):
            pr = self.get(pr_number)
            try:
                self.client.api(
                    "DELETE",
                    f"repos/{self.client.repo}/git/refs/heads/{pr.head_branch}",
                )
            except Exception:
                pass

        return result

    def add_comment(self, pr_number: int, body: str) -> None:
        self.client.api(
            "POST",
            f"repos/{self.client.repo}/issues/{pr_number}/comments",
            {"body": body},
        )

    def request_changes(
        self,
        pr_number: int,
        body: str,
        comments: list[dict[str, Any]] | None = None,
    ) -> None:
        payload = {"body": body, "event": "REQUEST_CHANGES", "comments": comments or []}
        self.client.api(
            "POST",
            f"repos/{self.client.repo}/pulls/{pr_number}/reviews",
            payload,
        )

    def approve(self, pr_number: int, body: str = "LGTM!") -> None:
        payload = {"body": body, "event": "APPROVE"}
        self.client.api(
            "POST",
            f"repos/{self.client.repo}/pulls/{pr_number}/reviews",
            payload,
        )

    def get_review_comments(self, pr_number: int) -> list[dict[str, Any]]:
        comments = self.client.api(
            "GET",
            f"repos/{self.client.repo}/pulls/{pr_number}/comments",
        )
        if isinstance(comments, dict):
            return []
        return [
            {
                "id": c.get("id"),
                "body": c.get("body", ""),
                "path": c.get("path"),
                "line": c.get("line"),
                "author": c.get("user", {}).get("login"),
                "created_at": c.get("created_at"),
            }
            for c in comments
        ]

    def respond_to_review(self, comment_id: int, response: str) -> None:
        self.client.api(
            "POST",
            f"repos/{self.client.repo}/pulls/comments/{comment_id}/replies",
            {"body": response},
        )

    def check_status(self, pr_number: int) -> dict[str, Any]:
        pr = self.get(pr_number)
        checks = self.client.api(
            "GET",
            f"repos/{self.client.repo}/commits/{pr.head_sha}/check-runs",
        )
        runs = checks.get("check_runs", []) if isinstance(checks, dict) else []
        return {
            "checks": [
                {
                    "name": run.get("name"),
                    "status": run.get("status"),
                    "conclusion": run.get("conclusion"),
                    "url": run.get("html_url"),
                }
                for run in runs
            ],
            "all_passing": all(run.get("conclusion") == "success" for run in runs),
        }

    def get(self, pr_number: int) -> PullRequest:
        data = self.client.api("GET", f"repos/{self.client.repo}/pulls/{pr_number}")
        return self._to_pull_request(data)

    def _to_pull_request(self, data: dict[str, Any]) -> PullRequest:
        return PullRequest(
            number=int(data.get("number", 0)),
            title=data.get("title", ""),
            body=data.get("body", ""),
            state=data.get("state", ""),
            head_branch=data.get("head", {}).get("ref", ""),
            head_sha=data.get("head", {}).get("sha", ""),
            base_branch=data.get("base", {}).get("ref", ""),
            url=data.get("html_url", ""),
            review_status=data.get("mergeable_state", ""),
            mergeable=bool(data.get("mergeable")),
            checks_passing=False,
        )
