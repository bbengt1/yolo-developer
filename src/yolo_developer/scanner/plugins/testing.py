from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import tomllib

from yolo_developer.scanner.base import ScanContext, ScanFinding, ScanResult


class TestingScanner:
    name = "testing"
    priority = 70

    def applies_to(self, context: ScanContext) -> bool:
        return True

    def scan(self, context: ScanContext) -> ScanResult:
        files_set = {path.name for path in context.files}
        testing_framework: dict[str, Any] | None = None
        confidence = 0.0

        if "pytest.ini" in files_set or "tox.ini" in files_set:
            testing_framework = {"framework": "pytest"}
            confidence = 0.9
        pyproject = context.project_path / "pyproject.toml"
        if pyproject.exists():
            try:
                data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            except (OSError, tomllib.TOMLDecodeError):
                data = {}
            if "pytest" in str(data).lower():
                testing_framework = {"framework": "pytest"}
                confidence = max(confidence, 0.8)
        package_json = context.project_path / "package.json"
        if package_json.exists():
            try:
                data = json.loads(package_json.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
            deps = {
                **data.get("dependencies", {}),
                **data.get("devDependencies", {}),
            }
            if "jest" in deps:
                testing_framework = {"framework": "jest"}
                confidence = max(confidence, 0.8)
            if "vitest" in deps:
                testing_framework = {"framework": "vitest"}
                confidence = max(confidence, 0.8)
        if any(path.suffix == ".go" for path in context.files):
            testing_framework = testing_framework or {"framework": "go test"}
            confidence = max(confidence, 0.6)

        findings = []
        if testing_framework:
            findings.append(
                ScanFinding(
                    key="testing",
                    value=testing_framework,
                    confidence=confidence,
                    source=self.name,
                )
            )

        suggestions = []
        if not testing_framework:
            suggestions.append("No test framework detected; configure testing manually.")

        return ScanResult(
            name=self.name,
            confidence=confidence,
            findings=findings,
            suggestions=suggestions,
        )
