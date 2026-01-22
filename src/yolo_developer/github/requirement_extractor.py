"""Extract requirements from parsed GitHub issues."""

from __future__ import annotations

from yolo_developer.github.models import ExtractedRequirement, ParsedIssue, StoryPriority

_NFR_KEYWORDS = {
    "performance": ["performance", "latency", "throughput", "response time"],
    "security": ["security", "secure", "encryption", "authentication", "authorization"],
    "scalability": ["scale", "scalable", "concurrent", "load"],
    "reliability": ["uptime", "reliability", "fault tolerant", "resilience"],
    "usability": ["usability", "user-friendly", "accessibility"],
}

_CONSTRAINT_KEYWORDS = ["must", "only", "cannot", "requires", "requirement", "constraint"]


class RequirementExtractor:
    """Extract structured requirements from parsed issues."""

    def extract(self, parsed: ParsedIssue) -> list[ExtractedRequirement]:
        requirements = []
        acceptance = parsed.acceptance_criteria
        for idx, description in enumerate(parsed.requirements, start=1):
            req_type = _categorize(description)
            requirements.append(
                ExtractedRequirement(
                    id=f"req-{parsed.issue.number}-{idx:02d}",
                    description=description,
                    type=req_type,
                    priority=parsed.priority,
                    source=f"issue-{parsed.issue.number}",
                    acceptance_criteria=acceptance,
                    assumptions=[],
                    dependencies=parsed.dependencies,
                    confidence=0.85 if parsed.requirements else 0.6,
                )
            )
        if not requirements:
            requirements.append(
                ExtractedRequirement(
                    id=f"req-{parsed.issue.number}-01",
                    description=parsed.objective or parsed.issue.title,
                    type="functional",
                    priority=parsed.priority,
                    source=f"issue-{parsed.issue.number}",
                    acceptance_criteria=acceptance,
                    assumptions=[],
                    dependencies=parsed.dependencies,
                    confidence=0.6,
                )
            )
        return requirements


def _categorize(text: str) -> str:
    text_lower = text.lower()
    for keywords in _NFR_KEYWORDS.values():
        if any(keyword in text_lower for keyword in keywords):
            return "non_functional"
    if any(keyword in text_lower for keyword in _CONSTRAINT_KEYWORDS):
        return "constraint"
    return "functional"
