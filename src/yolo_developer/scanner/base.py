from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class ScanContext:
    """Context passed to scanner plugins."""

    project_path: Path
    files: list[Path]
    directories: list[Path]
    scan_depth: int
    max_files: int
    hint: str | None = None


@dataclass(frozen=True)
class ScanFinding:
    """Single scanner finding with confidence metadata."""

    key: str
    value: Any
    confidence: float
    source: str


@dataclass
class ScanResult:
    """Results returned by a scanner plugin."""

    name: str
    confidence: float
    findings: list[ScanFinding]
    suggestions: list[str]


class ScannerPlugin(Protocol):
    """Protocol for scanner plugins."""

    name: str
    priority: int

    def applies_to(self, context: ScanContext) -> bool:
        """Return True if the plugin should run for this project."""

    def scan(self, context: ScanContext) -> ScanResult:
        """Scan project and return findings."""
