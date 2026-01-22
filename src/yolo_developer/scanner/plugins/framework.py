from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import tomllib

from yolo_developer.scanner.base import ScanContext, ScanFinding, ScanResult


@dataclass
class FrameworkIndicator:
    files: tuple[str, ...] = ()
    imports: tuple[str, ...] = ()
    deps: tuple[str, ...] = ()


FRAMEWORK_INDICATORS: dict[str, FrameworkIndicator] = {
    "fastapi": FrameworkIndicator(files=("main.py",), imports=("fastapi", "FastAPI")),
    "django": FrameworkIndicator(files=("manage.py", "settings.py"), imports=("django",)),
    "flask": FrameworkIndicator(imports=("flask", "Flask")),
    "react": FrameworkIndicator(deps=("react", "react-dom")),
    "nextjs": FrameworkIndicator(files=("next.config.js", "next.config.mjs"), deps=("next",)),
    "express": FrameworkIndicator(deps=("express",)),
    "nest": FrameworkIndicator(deps=("@nestjs/core",)),
    "gin": FrameworkIndicator(imports=("github.com/gin-gonic/gin",)),
    "echo": FrameworkIndicator(imports=("github.com/labstack/echo",)),
    "axum": FrameworkIndicator(imports=("axum",)),
}


class FrameworkScanner:
    name = "framework"
    priority = 90

    def applies_to(self, context: ScanContext) -> bool:
        return True

    def scan(self, context: ScanContext) -> ScanResult:
        deps = _load_dependencies(context.project_path)
        imports = _scan_imports(context)
        files_set = {path.name for path in context.files}

        frameworks: list[dict[str, Any]] = []
        for framework, indicator in FRAMEWORK_INDICATORS.items():
            match_score = 0
            for filename in indicator.files:
                if filename in files_set:
                    match_score += 2
            for dep in indicator.deps:
                if dep in deps:
                    match_score += 3
            for imp in indicator.imports:
                if imp in imports:
                    match_score += 1
            if match_score > 0:
                frameworks.append(
                    {
                        "name": framework,
                        "version": deps.get(framework) or deps.get(dep_key(framework)),
                        "confidence": min(1.0, match_score / 4),
                    }
                )

        confidence = 0.0 if not frameworks else max(item["confidence"] for item in frameworks)
        findings: list[ScanFinding] = []
        if frameworks:
            findings.append(
                ScanFinding(
                    key="frameworks",
                    value=frameworks,
                    confidence=confidence,
                    source=self.name,
                )
            )
        suggestions = []
        if not frameworks:
            suggestions.append("No framework detected; consider adding a project hint.")

        return ScanResult(
            name=self.name,
            confidence=confidence,
            findings=findings,
            suggestions=suggestions,
        )


def _load_dependencies(project_path: Path) -> dict[str, str]:
    deps: dict[str, str] = {}
    pyproject = project_path / "pyproject.toml"
    if pyproject.exists():
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            data = {}
        for item in data.get("project", {}).get("dependencies", []) or []:
            name = item.split(" ")[0].split("=")[0]
            deps[name.lower()] = item
    requirements = project_path / "requirements.txt"
    if requirements.exists():
        for line in requirements.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name = line.split("=")[0].split("<")[0].split(">=")[0]
            deps[name.lower()] = line
    package_json = project_path / "package.json"
    if package_json.exists():
        try:
            data = json.loads(package_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        for section in ("dependencies", "devDependencies", "peerDependencies"):
            for name, version in data.get(section, {}).items():
                deps[name.lower()] = version
    go_mod = project_path / "go.mod"
    if go_mod.exists():
        for line in go_mod.read_text(encoding="utf-8").splitlines():
            if line.startswith("require "):
                parts = line.split()
                if len(parts) >= 3:
                    deps[parts[1]] = parts[2]
    cargo = project_path / "Cargo.toml"
    if cargo.exists():
        try:
            data = tomllib.loads(cargo.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            data = {}
        for name, version in data.get("dependencies", {}).items():
            deps[name.lower()] = str(version)

    return deps


def _scan_imports(context: ScanContext) -> set[str]:
    imports: set[str] = set()
    for path in context.files:
        if path.suffix not in (".py", ".go", ".rs"):
            continue
        try:
            content = (context.project_path / path).read_text(encoding="utf-8")
        except OSError:
            continue
        for line in content.splitlines()[:200]:
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                imports.add(stripped.replace("import", "").replace("from", "").strip())
            if stripped.startswith("package "):
                imports.add(stripped)
            if stripped.startswith("use "):
                imports.add(stripped)
    return imports


def dep_key(framework: str) -> str:
    mapping = {
        "nextjs": "next",
        "nest": "@nestjs/core",
    }
    return mapping.get(framework, framework)
