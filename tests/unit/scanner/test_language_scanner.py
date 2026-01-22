from __future__ import annotations

from pathlib import Path

from yolo_developer.scanner import ScannerManager


def test_language_scanner_detects_python(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[project]
name = "example"
requires-python = ">=3.11"
""".strip()
    )
    (tmp_path / "main.py").write_text("print('hello')\n")

    manager = ScannerManager()
    report = manager.scan(
        project_path=tmp_path,
        scan_depth=2,
        exclude_patterns=[],
        max_files=100,
        include_git_history=False,
    )

    language = report.best_finding("language")
    version = report.best_finding("language_version")

    assert language is not None
    assert language.value == "python"
    assert version is not None
    assert "3.11" in version.value


def test_framework_scanner_detects_fastapi(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("fastapi==0.109.0\n")
    (tmp_path / "main.py").write_text("from fastapi import FastAPI\n")

    manager = ScannerManager()
    report = manager.scan(
        project_path=tmp_path,
        scan_depth=2,
        exclude_patterns=[],
        max_files=100,
        include_git_history=False,
    )

    frameworks = report.best_finding("frameworks")
    assert frameworks is not None
    assert any(item["name"] == "fastapi" for item in frameworks.value)
