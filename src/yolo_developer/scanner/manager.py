from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml
from rich.console import Console

from yolo_developer.scanner.base import ScanContext, ScanFinding, ScanResult, ScannerPlugin
from yolo_developer.scanner.plugins import (
    BuildScanner,
    DocsScanner,
    FrameworkScanner,
    GitScanner,
    LanguageScanner,
    PatternsScanner,
    StructureScanner,
    TestingScanner,
)
from yolo_developer.scanner.report import ScanReport


@dataclass
class ProjectContext:
    """Project context derived from scan results."""

    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)


class ScannerManager:
    """Orchestrates scanner plugins and assembles scan reports."""

    def __init__(self, plugins: Iterable[ScannerPlugin] | None = None) -> None:
        if plugins is None:
            plugins = [
                LanguageScanner(),
                FrameworkScanner(),
                BuildScanner(),
                TestingScanner(),
                StructureScanner(),
                PatternsScanner(),
                DocsScanner(),
                GitScanner(),
            ]
        self._plugins = sorted(plugins, key=lambda plugin: plugin.priority, reverse=True)

    def scan(
        self,
        project_path: Path,
        scan_depth: int,
        exclude_patterns: list[str],
        max_files: int,
        include_git_history: bool,
        hint: str | None = None,
    ) -> ScanReport:
        files, directories = _walk_project(project_path, scan_depth, exclude_patterns, max_files)
        context = ScanContext(
            project_path=project_path,
            files=files,
            directories=directories,
            scan_depth=scan_depth,
            max_files=max_files,
            hint=hint,
        )
        report = ScanReport(metadata={"scanned_files": len(files)})

        for plugin in self._plugins:
            if not include_git_history and isinstance(plugin, GitScanner):
                continue
            if not plugin.applies_to(context):
                continue
            result = plugin.scan(context)
            report.add_result(result)

        return report

    def build_project_context(
        self,
        report: ScanReport,
        project_name: str,
        interactive: bool,
        console: Console | None = None,
    ) -> ProjectContext:
        console = console or Console()
        context: dict[str, Any] = {
            "project": {
                "name": project_name,
            }
        }

        _apply_finding(report, context, "language", "project", "language", interactive, console)
        _apply_finding(report, context, "language_version", "project", "version", interactive, console)
        _apply_frameworks(report, context)
        _apply_structure(report, context)
        _apply_patterns(report, context)
        _apply_testing(report, context)
        _apply_dependencies(report, context)
        _apply_conventions(report, context)
        _apply_entry_points(report, context)

        return ProjectContext(data=context)

    def write_project_context(self, project_path: Path, context: ProjectContext) -> Path:
        yolo_dir = project_path / ".yolo"
        yolo_dir.mkdir(parents=True, exist_ok=True)
        output_path = yolo_dir / "project-context.yaml"
        with output_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(context.to_dict(), handle, default_flow_style=False, sort_keys=False)
        return output_path


def _walk_project(
    project_path: Path,
    scan_depth: int,
    exclude_patterns: list[str],
    max_files: int,
) -> tuple[list[Path], list[Path]]:
    files: list[Path] = []
    directories: list[Path] = []
    exclude_set = set(exclude_patterns)

    for root, dirnames, filenames in _walk_depth(project_path, scan_depth):
        rel_root = root.relative_to(project_path)
        dirnames[:] = [
            d for d in dirnames if not _is_excluded(rel_root / d, exclude_set)
        ]
        for dirname in dirnames:
            directories.append(rel_root / dirname)
        for filename in filenames:
            rel_path = rel_root / filename
            if _is_excluded(rel_path, exclude_set):
                continue
            files.append(rel_path)
            if len(files) >= max_files:
                return files, directories

    return files, directories


def _walk_depth(project_path: Path, max_depth: int):
    stack = [(project_path, 0)]
    while stack:
        current, depth = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        dirnames = [entry.name for entry in entries if entry.is_dir()]
        filenames = [entry.name for entry in entries if entry.is_file()]
        yield current, dirnames, filenames
        if depth < max_depth:
            for dirname in dirnames:
                stack.append((current / dirname, depth + 1))


def _is_excluded(path: Path, exclude_set: set[str]) -> bool:
    for part in path.parts:
        if part in exclude_set:
            return True
    return False


def _apply_finding(
    report: ScanReport,
    context: dict[str, Any],
    key: str,
    section: str,
    field: str,
    interactive: bool,
    console: Console,
) -> None:
    finding = report.best_finding(key)
    if finding is None:
        return
    if interactive and 0.5 <= finding.confidence < 0.9:
        from typer import confirm

        confirmed = confirm(
            f"Detected {key}: {finding.value}. Is this correct?",
            default=True,
        )
        if not confirmed:
            return
    context.setdefault(section, {})
    context[section][field] = finding.value


def _apply_frameworks(report: ScanReport, context: dict[str, Any]) -> None:
    finding = report.best_finding("frameworks")
    if finding is None:
        return
    context["frameworks"] = finding.value


def _apply_structure(report: ScanReport, context: dict[str, Any]) -> None:
    finding = report.best_finding("structure")
    if finding is None:
        return
    context["structure"] = finding.value


def _apply_patterns(report: ScanReport, context: dict[str, Any]) -> None:
    finding = report.best_finding("patterns")
    if finding is None:
        return
    context["patterns"] = finding.value


def _apply_testing(report: ScanReport, context: dict[str, Any]) -> None:
    finding = report.best_finding("testing")
    if finding is None:
        return
    context["testing"] = finding.value


def _apply_dependencies(report: ScanReport, context: dict[str, Any]) -> None:
    finding = report.best_finding("dependencies")
    if finding is None:
        return
    context["dependencies"] = finding.value


def _apply_conventions(report: ScanReport, context: dict[str, Any]) -> None:
    finding = report.best_finding("conventions")
    if finding is None:
        return
    context["conventions"] = finding.value


def _apply_entry_points(report: ScanReport, context: dict[str, Any]) -> None:
    finding = report.best_finding("entry_points")
    if finding is None:
        return
    context["entry_points"] = finding.value
