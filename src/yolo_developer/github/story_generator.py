"""Generate user stories from extracted requirements."""

from __future__ import annotations

from yolo_developer.agents.pm.llm import _extract_story_components, _generate_acceptance_criteria_llm
from yolo_developer.agents.pm.node import _estimate_complexity
from yolo_developer.github.config import GitHubImportStoryConfig
from yolo_developer.github.models import (
    ExtractedRequirement,
    GeneratedStory,
    GitHubIssueInput,
    IssueType,
    StoryPriority,
)

_POINTS_MAP = {"S": 3, "M": 5, "L": 8, "XL": 13}


class StoryGenerator:
    """Generate user stories from extracted requirements."""

    def __init__(self, config: GitHubImportStoryConfig | None = None) -> None:
        self.config = config or GitHubImportStoryConfig()

    async def generate(
        self,
        issue: GitHubIssueInput,
        requirements: list[ExtractedRequirement],
        issue_type: IssueType,
        priority: StoryPriority,
        technical_notes: str = "",
    ) -> GeneratedStory:
        requirement_text = _join_requirements(requirements, issue)
        story_components = await _extract_story_components(
            requirement_id=f"issue-{issue.number}",
            requirement_text=requirement_text,
            category="functional",
        )
        ac_data = await _generate_acceptance_criteria_llm(
            requirement_id=f"issue-{issue.number}",
            requirement_text=requirement_text,
            story_components=story_components,
        )
        acceptance_criteria = _render_acceptance_criteria(ac_data, requirements)

        description = (
            f"As a {story_components.get('role', 'user')}, "
            f"I want {story_components.get('action', requirement_text)} "
            f"so that {story_components.get('benefit', 'the system meets the requirement')}."
        )

        estimation_points = None
        if self.config.estimate_points:
            complexity = _estimate_complexity(requirement_text, len(acceptance_criteria))
            estimation_points = _POINTS_MAP.get(complexity)

        notes = technical_notes if self.config.include_technical_notes else ""
        if not notes:
            notes = _derive_notes(issue, requirements)

        return GeneratedStory(
            id=f"{self.config.id_prefix}-{issue.number:04d}",
            title=story_components.get("title", issue.title),
            description=description,
            type=issue_type,
            priority=priority,
            acceptance_criteria=acceptance_criteria,
            technical_notes=notes,
            estimation_points=estimation_points,
            github_issue=issue.number,
            dependencies=_collect_dependencies(requirements),
            tags=issue.labels,
        )


def _join_requirements(
    requirements: list[ExtractedRequirement], issue: GitHubIssueInput
) -> str:
    if requirements:
        return " ".join(req.description for req in requirements)
    return issue.body or issue.title


def _render_acceptance_criteria(
    ac_data: list[dict[str, object]],
    requirements: list[ExtractedRequirement],
) -> list[str]:
    if ac_data:
        return [
            _format_ac(item)
            for item in ac_data
            if isinstance(item, dict)
        ]
    fallback = []
    for requirement in requirements:
        fallback.extend(requirement.acceptance_criteria)
    return fallback or ["Requirement is satisfied and verified."]


def _format_ac(item: dict[str, object]) -> str:
    given = str(item.get("given", "")).strip()
    when = str(item.get("when", "")).strip()
    then = str(item.get("then", "")).strip()
    and_clauses = item.get("and_clauses", [])
    extra = ""
    if isinstance(and_clauses, list) and and_clauses:
        extra = " AND " + " AND ".join(str(clause) for clause in and_clauses)
    return f"Given {given}, when {when}, then {then}{extra}."


def _derive_notes(issue: GitHubIssueInput, requirements: list[ExtractedRequirement]) -> str:
    notes = []
    if issue.body:
        notes.append(issue.body.strip())
    if requirements:
        notes.append("Requirements captured from issue import.")
    return "\n\n".join(note for note in notes if note)


def _collect_dependencies(requirements: list[ExtractedRequirement]) -> list[str]:
    deps: list[str] = []
    for requirement in requirements:
        deps.extend(requirement.dependencies)
    return sorted(set(deps))
