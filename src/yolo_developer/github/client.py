from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from yolo_developer.github.models import GitHubError


class GitHubClient:
    """Wrapper around the GitHub CLI for API operations."""

    def __init__(self, repo: str, token: str | None = None, cwd: Path | None = None) -> None:
        self.repo = repo
        self.token = token
        self.cwd = cwd or Path.cwd()
        if not shutil.which("gh"):
            raise GitHubError("GitHub CLI not found. Install 'gh' to use GitHub features.")

    def api(self, method: str, endpoint: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = data or {}
        args = ["gh", "api", f"-X", method, endpoint]
        for key, value in payload.items():
            if value is None:
                continue
            if isinstance(value, list):
                for item in value:
                    _append_field(args, f"{key}[]", item)
            else:
                _append_field(args, key, value)
        output = self._run(args)
        if not output:
            return {}
        try:
            return json.loads(output)
        except json.JSONDecodeError as exc:
            raise GitHubError(f"Invalid JSON response from GitHub API: {exc}") from exc

    def _run(self, args: list[str]) -> str:
        env = os.environ.copy()
        if self.token:
            env.setdefault("GH_TOKEN", self.token)
            env.setdefault("GITHUB_TOKEN", self.token)
        result = subprocess.run(
            args,
            cwd=self.cwd,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        if result.returncode != 0:
            raise GitHubError(result.stderr.strip() or result.stdout.strip())
        return result.stdout.strip()


def _append_field(args: list[str], key: str, value: Any) -> None:
    if isinstance(value, bool):
        args.extend(["-F", f"{key}={str(value).lower()}"])
    else:
        args.extend(["-f", f"{key}={value}"])
