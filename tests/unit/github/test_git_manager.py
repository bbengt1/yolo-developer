from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from yolo_developer.github.git import GitManager, _parse_repo_slug
from yolo_developer.github.models import GitHubError


def _run(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def test_git_manager_status_and_commit(tmp_path: Path) -> None:
    _run(tmp_path, "init")
    _run(tmp_path, "config", "user.email", "dev@example.com")
    _run(tmp_path, "config", "user.name", "Dev")

    file_path = tmp_path / "README.md"
    file_path.write_text("hello\n")

    manager = GitManager(tmp_path)
    status = manager.status()
    assert status["untracked"] == ["README.md"]

    manager.stage_files("README.md")
    commit = manager.commit("feat: init")
    assert commit.message == "feat: init"
    assert commit.files_changed == ["README.md"]

    status = manager.status()
    assert status["clean"] is True


def test_git_manager_requires_repo(tmp_path: Path) -> None:
    with pytest.raises(GitHubError):
        GitManager(tmp_path)


def test_parse_repo_slug() -> None:
    assert _parse_repo_slug("git@github.com:org/repo.git") == "org/repo"
    assert _parse_repo_slug("https://github.com/org/repo.git") == "org/repo"
    assert _parse_repo_slug("not-a-url") is None
