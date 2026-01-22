from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from yolo_developer.scanner.base import ScanFinding, ScanResult


@dataclass
class ScanReport:
    """Aggregated scan results across plugins."""

    results: list[ScanResult] = field(default_factory=list)
    findings: list[ScanFinding] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_result(self, result: ScanResult) -> None:
        self.results.append(result)
        self.findings.extend(result.findings)
        self.suggestions.extend(result.suggestions)

    def get_findings(self, key: str) -> list[ScanFinding]:
        return [finding for finding in self.findings if finding.key == key]

    def best_finding(self, key: str) -> ScanFinding | None:
        candidates = self.get_findings(key)
        if not candidates:
            return None
        return max(candidates, key=lambda finding: finding.confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "findings": [
                {
                    "key": finding.key,
                    "value": finding.value,
                    "confidence": finding.confidence,
                    "source": finding.source,
                }
                for finding in self.findings
            ],
            "suggestions": list(self.suggestions),
            "metadata": dict(self.metadata),
        }
