from __future__ import annotations

from typing import Any

from yolo_developer.github.client import GitHubClient
from yolo_developer.github.models import Release, parse_datetime


class ReleaseManager:
    """Manage GitHub releases."""

    def __init__(self, client: GitHubClient) -> None:
        self.client = client

    def create(
        self,
        tag: str,
        name: str,
        body: str,
        target: str = "main",
        draft: bool = False,
        prerelease: bool = False,
        generate_notes: bool = True,
    ) -> Release:
        if generate_notes:
            body = self._generate_release_notes(tag, body, target)

        data = self.client.api(
            "POST",
            f"repos/{self.client.repo}/releases",
            {
                "tag_name": tag,
                "name": name,
                "body": body,
                "draft": draft,
                "prerelease": prerelease,
                "target_commitish": target,
            },
        )
        return self._to_release(data)

    def _generate_release_notes(self, tag: str, base_body: str, target: str) -> str:
        data = self.client.api(
            "POST",
            f"repos/{self.client.repo}/releases/generate-notes",
            {"tag_name": tag, "target_commitish": target},
        )
        notes = data.get("body", "") if isinstance(data, dict) else ""
        if base_body:
            return f"{base_body}\n\n{notes}" if notes else base_body
        return notes

    def list(self) -> list[Release]:
        data = self.client.api("GET", f"repos/{self.client.repo}/releases")
        if isinstance(data, dict):
            return []
        return [self._to_release(item) for item in data]

    def _to_release(self, data: dict[str, Any]) -> Release:
        return Release(
            tag=data.get("tag_name", ""),
            name=data.get("name", ""),
            body=data.get("body", ""),
            draft=bool(data.get("draft")),
            prerelease=bool(data.get("prerelease")),
            url=data.get("html_url", ""),
            assets=[asset.get("name", "") for asset in data.get("assets", [])],
            created_at=parse_datetime(data.get("created_at")),
        )
