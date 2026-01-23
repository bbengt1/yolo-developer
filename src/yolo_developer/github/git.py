from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from yolo_developer.github.models import BranchInfo, CommitResult, GitHubError


class GitManager:
    """Manage local Git operations."""

    def __init__(self, repo_path: Path) -> None:
        self.repo_path = repo_path
        self._validate_repo()

    def _validate_repo(self) -> None:
        if not (self.repo_path / ".git").exists():
            raise GitHubError(f"Git repository not found: {self.repo_path}")

    def _run_git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise GitHubError(result.stderr.strip() or result.stdout.strip())
        return result.stdout.rstrip()

    def _get_upstream(self, branch: str) -> str | None:
        try:
            return self._run_git("rev-parse", "--abbrev-ref", f"{branch}@{{upstream}}")
        except GitHubError:
            return None

    def _get_ahead_behind(self, branch: str) -> tuple[int, int]:
        upstream = self._get_upstream(branch)
        if not upstream:
            return (0, 0)
        output = self._run_git("rev-list", "--left-right", "--count", f"{branch}...{upstream}")
        ahead, behind = output.split()
        return int(ahead), int(behind)

    def get_current_branch(self) -> BranchInfo:
        name = self._run_git("rev-parse", "--abbrev-ref", "HEAD")
        ahead, behind = self._get_ahead_behind(name)
        return BranchInfo(
            name=name,
            upstream=self._get_upstream(name),
            ahead=ahead,
            behind=behind,
            is_current=True,
        )

    def list_branches(self, remote: bool = False) -> list[BranchInfo]:
        flag = "-r" if remote else ""
        output = self._run_git("branch", flag, "--format=%(refname:short)")
        branches = []
        for branch in output.splitlines():
            branch = branch.strip()
            if not branch:
                continue
            branches.append(self._get_branch_info(branch))
        return branches

    def _get_branch_info(self, branch: str) -> BranchInfo:
        ahead, behind = self._get_ahead_behind(branch)
        return BranchInfo(
            name=branch,
            upstream=self._get_upstream(branch),
            ahead=ahead,
            behind=behind,
            is_current=branch == self.get_current_branch().name,
        )

    def create_branch(self, name: str, base: str = "main") -> BranchInfo:
        self._run_git("checkout", "-b", name, base)
        return self.get_current_branch()

    def checkout(self, branch: str) -> BranchInfo:
        self._run_git("checkout", branch)
        return self.get_current_branch()

    def delete_branch(self, name: str, force: bool = False) -> None:
        flag = "-D" if force else "-d"
        self._run_git("branch", flag, name)

    def stage_files(self, files: list[str] | str = ".") -> list[str]:
        if isinstance(files, str):
            files = [files]
        for file in files:
            self._run_git("add", file)
        return self.get_staged_files()

    def get_staged_files(self) -> list[str]:
        output = self._run_git("diff", "--cached", "--name-only")
        return [line for line in output.splitlines() if line]

    def commit(
        self,
        message: str,
        body: str | None = None,
        co_authors: list[str] | None = None,
    ) -> CommitResult:
        staged_files = self.get_staged_files()
        full_message = message
        if body:
            full_message += f"\n\n{body}"
        if co_authors:
            full_message += "\n\n" + "\n".join(
                f"Co-Authored-By: {author}" for author in co_authors
            )

        self._run_git("commit", "-m", full_message)
        sha = self._run_git("rev-parse", "HEAD")
        stats = self._run_git("diff", "--stat", "HEAD~1", "HEAD")
        insertions, deletions = _parse_stats(stats)

        return CommitResult(
            sha=sha[:8],
            message=message,
            files_changed=staged_files,
            insertions=insertions,
            deletions=deletions,
        )

    def amend(self, message: str | None = None) -> CommitResult:
        if message:
            self._run_git("commit", "--amend", "-m", message)
        else:
            self._run_git("commit", "--amend", "--no-edit")
        return self._get_last_commit()

    def _get_last_commit(self) -> CommitResult:
        sha = self._run_git("rev-parse", "HEAD")
        message = self._run_git("log", "-1", "--pretty=%s")
        stats = self._run_git("diff", "--stat", "HEAD~1", "HEAD")
        insertions, deletions = _parse_stats(stats)
        return CommitResult(
            sha=sha[:8],
            message=message,
            files_changed=[],
            insertions=insertions,
            deletions=deletions,
        )

    def push(self, branch: str | None = None, force: bool = False, set_upstream: bool = False) -> None:
        args = ["push"]
        if force:
            args.append("--force-with-lease")
        if set_upstream:
            target = branch or self.get_current_branch().name
            args.extend(["-u", "origin", target])
        elif branch:
            args.extend(["origin", branch])
        self._run_git(*args)

    def pull(self, rebase: bool = True) -> None:
        if rebase:
            self._run_git("pull", "--rebase")
        else:
            self._run_git("pull")

    def fetch(self, prune: bool = True) -> None:
        args = ["fetch"]
        if prune:
            args.append("--prune")
        self._run_git(*args)

    def status(self) -> dict[str, object]:
        return {
            "branch": self.get_current_branch(),
            "staged": self.get_staged_files(),
            "modified": self._get_modified_files(),
            "untracked": self._get_untracked_files(),
            "clean": self._is_clean(),
        }

    def diff(self, staged: bool = False, file: str | None = None) -> str:
        args = ["diff"]
        if staged:
            args.append("--cached")
        if file:
            args.append(file)
        return self._run_git(*args)

    def _get_modified_files(self) -> list[str]:
        output = self._run_git("status", "--porcelain")
        modified = []
        for line in output.splitlines():
            if line[:2].strip() and not line.startswith("??"):
                modified.append(line[3:])
        return modified

    def _get_untracked_files(self) -> list[str]:
        output = self._run_git("status", "--porcelain")
        return [line[3:] for line in output.splitlines() if line.startswith("??")]

    def _is_clean(self) -> bool:
        return self._run_git("status", "--porcelain") == ""

    def get_repo_slug(self) -> str | None:
        try:
            url = self._run_git("config", "--get", "remote.origin.url")
        except GitHubError:
            return None
        return _parse_repo_slug(url)


def _parse_repo_slug(url: str) -> str | None:
    match = re.search(r"github.com[:/](?P<slug>[^/]+/[^/.]+)", url)
    if match:
        return match.group("slug")
    return None


def _parse_stats(stats: str) -> tuple[int, int]:
    insertions = 0
    deletions = 0
    summary_match = re.search(r"(\d+) insertion", stats)
    if summary_match:
        insertions = int(summary_match.group(1))
    deletion_match = re.search(r"(\d+) deletion", stats)
    if deletion_match:
        deletions = int(deletion_match.group(1))
    return insertions, deletions
