from __future__ import annotations

import json

from yolo_developer.analyst.models import ExtractedRequirement


class RequirementsExporter:
    """Export requirements to markdown, json, or yaml."""

    def export(
        self,
        requirements: list[ExtractedRequirement],
        format: str = "markdown",
        metadata: dict[str, object] | None = None,
    ) -> str:
        fmt = format.lower()
        metadata = metadata or {}
        if fmt == "json":
            return json.dumps(_build_payload(requirements, metadata), indent=2)
        if fmt == "yaml":
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("PyYAML is required for YAML export") from exc
            return yaml.safe_dump(_build_payload(requirements, metadata), sort_keys=False)
        return _export_markdown(requirements, metadata)


def _build_payload(
    requirements: list[ExtractedRequirement],
    metadata: dict[str, object],
) -> dict[str, object]:
    return {
        "metadata": metadata,
        "requirements": [
            {
                "id": req.id,
                "description": req.description,
                "type": req.type,
                "priority": req.priority,
                "confidence": req.confidence,
            }
            for req in requirements
        ],
    }


def _export_markdown(
    requirements: list[ExtractedRequirement],
    metadata: dict[str, object],
) -> str:
    lines = ["# Requirements", ""]
    if metadata:
        lines.append("## Metadata")
        for key, value in metadata.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("")
    if not requirements:
        lines.append("_No requirements captured yet._")
        return "\n".join(lines)
    lines.append("## Requirements")
    for req in requirements:
        lines.append(
            f"- **{req.id}** ({req.type}, {req.priority}): {req.description}"
        )
    return "\n".join(lines)
