from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib

from yolo_developer.scanner.base import ScanContext, ScanFinding, ScanResult


@dataclass
class LanguageIndicators:
    files: tuple[str, ...]
    markers: tuple[str, ...] = ()
    extensions: tuple[str, ...] = ()


LANGUAGE_INDICATORS: dict[str, LanguageIndicators] = {
    "python": LanguageIndicators(
        files=("pyproject.toml", "setup.py", "requirements.txt"),
        markers=("__init__.py", ".python-version"),
        extensions=(".py",),
    ),
    "javascript": LanguageIndicators(
        files=("package.json",),
        markers=(".nvmrc", ".node-version"),
        extensions=(".js", ".jsx"),
    ),
    "typescript": LanguageIndicators(
        files=("tsconfig.json",),
        markers=(),
        extensions=(".ts", ".tsx"),
    ),
    "go": LanguageIndicators(
        files=("go.mod", "go.sum"),
        markers=(),
        extensions=(".go",),
    ),
    "rust": LanguageIndicators(
        files=("Cargo.toml",),
        markers=(),
        extensions=(".rs",),
    ),
}


class LanguageScanner:
    name = "language"
    priority = 100

    def applies_to(self, context: ScanContext) -> bool:
        return True

    def scan(self, context: ScanContext) -> ScanResult:
        scores: dict[str, int] = {lang: 0 for lang in LANGUAGE_INDICATORS}
        files_set = {path.name for path in context.files}
        extensions = [path.suffix for path in context.files]

        for language, indicators in LANGUAGE_INDICATORS.items():
            for filename in indicators.files:
                if filename in files_set:
                    scores[language] += 3
            for marker in indicators.markers:
                if marker in files_set:
                    scores[language] += 2
            for ext in indicators.extensions:
                scores[language] += extensions.count(ext)

        best_language = max(scores.items(), key=lambda item: item[1])
        language, score = best_language
        confidence = 0.0 if score == 0 else min(1.0, score / 10)

        findings: list[ScanFinding] = []
        if score > 0:
            findings.append(
                ScanFinding(
                    key="language",
                    value=language,
                    confidence=confidence,
                    source=self.name,
                )
            )
            version = _detect_language_version(context.project_path, language)
            if version:
                findings.append(
                    ScanFinding(
                        key="language_version",
                        value=version,
                        confidence=confidence,
                        source=self.name,
                    )
                )

        suggestions = []
        if score == 0:
            suggestions.append("Could not detect primary language; consider providing a hint.")

        return ScanResult(
            name=self.name,
            confidence=confidence,
            findings=findings,
            suggestions=suggestions,
        )


def _detect_language_version(project_path: Path, language: str) -> str | None:
    if language == "python":
        pyversion = _read_text(project_path / ".python-version")
        if pyversion:
            return pyversion.strip()
        pyproject = project_path / "pyproject.toml"
        if pyproject.exists():
            try:
                data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            except (OSError, tomllib.TOMLDecodeError):
                return None
            requires_python = data.get("project", {}).get("requires-python")
            if isinstance(requires_python, str):
                return requires_python
    return None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None
