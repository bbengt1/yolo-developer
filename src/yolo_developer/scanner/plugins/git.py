from __future__ import annotations

import subprocess

from yolo_developer.scanner.base import ScanContext, ScanFinding, ScanResult


class GitScanner:
    name = "git"
    priority = 40

    def applies_to(self, context: ScanContext) -> bool:
        return (context.project_path / ".git").exists()

    def scan(self, context: ScanContext) -> ScanResult:
        branch = _git_command(context.project_path, ["rev-parse", "--abbrev-ref", "HEAD"])
        findings = [
            ScanFinding(
                key="git",
                value={"enabled": True, "branch": branch or ""},
                confidence=0.7,
                source=self.name,
            )
        ]
        return ScanResult(
            name=self.name,
            confidence=0.7,
            findings=findings,
            suggestions=[],
        )


def _git_command(project_path, args):
    try:
        result = subprocess.run(
            ["git", "-C", str(project_path), *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()
