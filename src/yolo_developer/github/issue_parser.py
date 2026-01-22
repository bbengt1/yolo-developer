"""Parse GitHub issues into structured components."""

from __future__ import annotations

import re
from typing import Iterable

from yolo_developer.github.config import GitHubImportTemplateConfig
from yolo_developer.github.models import GitHubIssueInput, IssueType, ParsedIssue, StoryPriority
from yolo_developer.github.templates import DEFAULT_TEMPLATE_MARKERS


class IssueParser:
    """Parse GitHub issue content into structured fields."""

    def __init__(self, templates: GitHubImportTemplateConfig | None = None) -> None:
        self.templates = templates or GitHubImportTemplateConfig()

    def parse(self, issue: GitHubIssueInput) -> ParsedIssue:
        issue_type = _detect_issue_type(issue)
        priority = _detect_priority(issue)
        template_name = self._detect_template(issue.body)
        sections = _split_sections(issue.body)

        objective = _first_nonempty(
            sections.get("overview"),
            sections.get("summary"),
            sections.get("feature description"),
            sections.get("bug description"),
            sections.get("current behavior"),
            issue.title,
        )

        requirements = _extract_list_items(
            sections.get("requirements", ""),
            sections.get("feature requirements", ""),
            sections.get("acceptance criteria", ""),
            sections.get("proposed enhancement", ""),
        )
        if not requirements:
            requirements = _extract_paragraphs(issue.body, limit=3)

        acceptance_criteria = _extract_list_items(sections.get("acceptance criteria", ""))
        if not acceptance_criteria and template_name == "bug":
            acceptance_criteria = _extract_list_items(sections.get("expected behavior", ""))

        technical_notes = _first_nonempty(
            sections.get("technical notes"),
            sections.get("implementation notes"),
            sections.get("additional context"),
            "",
        )

        dependencies = _extract_dependencies(issue.body)

        return ParsedIssue(
            issue=issue,
            issue_type=issue_type,
            objective=objective,
            requirements=requirements,
            acceptance_criteria=acceptance_criteria,
            technical_notes=technical_notes,
            dependencies=dependencies,
            priority=priority,
        )

    def _detect_template(self, body: str) -> str | None:
        markers = dict(DEFAULT_TEMPLATE_MARKERS)
        if self.templates.feature:
            markers["feature"] = self.templates.feature
        if self.templates.bug:
            markers["bug"] = self.templates.bug
        if self.templates.enhancement:
            markers["enhancement"] = self.templates.enhancement
        if self.templates.custom:
            markers["custom"] = self.templates.custom

        for name, required in markers.items():
            if required and all(marker in body for marker in required[:2]):
                return name
        return None


def _detect_issue_type(issue: GitHubIssueInput) -> IssueType:
    label_map = {
        "bug": IssueType.BUG,
        "feature": IssueType.FEATURE,
        "enhancement": IssueType.ENHANCEMENT,
        "task": IssueType.TASK,
        "epic": IssueType.EPIC,
    }
    for label in issue.labels:
        label_lower = label.lower()
        for key, issue_type in label_map.items():
            if key in label_lower:
                return issue_type
    return IssueType.UNKNOWN


def _detect_priority(issue: GitHubIssueInput) -> StoryPriority:
    priority_map = {
        "p0": StoryPriority.CRITICAL,
        "p1": StoryPriority.HIGH,
        "p2": StoryPriority.MEDIUM,
        "p3": StoryPriority.LOW,
        "critical": StoryPriority.CRITICAL,
        "high": StoryPriority.HIGH,
        "medium": StoryPriority.MEDIUM,
        "low": StoryPriority.LOW,
    }
    for label in issue.labels:
        label_lower = label.lower()
        for key, priority in priority_map.items():
            if key in label_lower:
                return priority
    return StoryPriority.MEDIUM


def _split_sections(body: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current = "overview"
    lines: list[str] = []
    for line in body.splitlines():
        heading = re.match(r"^#{2,6}\s+(.*)", line.strip())
        if heading:
            if lines:
                sections[current] = "\n".join(lines).strip()
            current = heading.group(1).strip().lower()
            lines = []
        else:
            lines.append(line)
    if lines:
        sections[current] = "\n".join(lines).strip()
    return sections


def _extract_list_items(*chunks: str) -> list[str]:
    items: list[str] = []
    for chunk in chunks:
        if not chunk:
            continue
        for line in chunk.splitlines():
            match = re.match(r"^\s*[-*]\s*(?:\[[xX ]\]\s*)?(.*)", line)
            if match and match.group(1).strip():
                items.append(match.group(1).strip())
    return [item for item in items if item]


def _extract_paragraphs(text: str, limit: int = 3) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs[:limit] if paragraphs else []


def _extract_dependencies(text: str) -> list[str]:
    dependencies: list[str] = []
    for match in re.finditer(r"#(\d+)", text):
        issue_ref = match.group(1)
        dependencies.append(f"#{issue_ref}")
    return sorted(set(dependencies))


def _first_nonempty(*values: str) -> str:
    for value in values:
        if value and value.strip():
            return value.strip()
    return ""
