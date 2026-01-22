from __future__ import annotations

import re
from collections import Counter

from yolo_developer.scanner.base import ScanContext, ScanFinding, ScanResult


class PatternsScanner:
    name = "patterns"
    priority = 55

    def applies_to(self, context: ScanContext) -> bool:
        return True

    def scan(self, context: ScanContext) -> ScanResult:
        naming_style = _detect_naming_style(context)
        docstring_style = _detect_docstring_style(context)
        type_hints = _detect_type_hints(context)

        patterns = {
            "naming": naming_style,
            "docstrings": docstring_style,
            "type_hints": type_hints,
        }
        findings = [
            ScanFinding(
                key="patterns",
                value=patterns,
                confidence=0.6,
                source=self.name,
            )
        ]
        return ScanResult(
            name=self.name,
            confidence=0.6,
            findings=findings,
            suggestions=[],
        )


def _detect_naming_style(context: ScanContext) -> str:
    filenames = [path.stem for path in context.files if path.suffix in {".py", ".js", ".ts"}]
    counts = Counter()
    for name in filenames:
        if "_" in name:
            counts["snake_case"] += 1
        elif name and name[0].isupper():
            counts["PascalCase"] += 1
        elif re.search(r"[a-z][A-Z]", name):
            counts["camelCase"] += 1
    if not counts:
        return "snake_case"
    return counts.most_common(1)[0][0]


def _detect_docstring_style(context: ScanContext) -> str:
    python_files = [path for path in context.files if path.suffix == ".py"]
    for path in python_files[:20]:
        try:
            content = (context.project_path / path).read_text(encoding="utf-8")
        except OSError:
            continue
        if "Args:" in content or "Returns:" in content:
            return "google"
        if ":param" in content:
            return "sphinx"
        if "Parameters" in content and "-------" in content:
            return "numpy"
    return "none"


def _detect_type_hints(context: ScanContext) -> bool:
    python_files = [path for path in context.files if path.suffix == ".py"]
    for path in python_files[:20]:
        try:
            content = (context.project_path / path).read_text(encoding="utf-8")
        except OSError:
            continue
        if "->" in content or ":" in content:
            return True
    return False
