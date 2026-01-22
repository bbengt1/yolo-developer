"""Issue template markers for GitHub issue import."""

from __future__ import annotations

from yolo_developer.github.templates.bug import BUG_MARKERS
from yolo_developer.github.templates.custom import CUSTOM_MARKERS
from yolo_developer.github.templates.enhancement import ENHANCEMENT_MARKERS
from yolo_developer.github.templates.feature import FEATURE_MARKERS

DEFAULT_TEMPLATE_MARKERS: dict[str, list[str]] = {
    "feature": FEATURE_MARKERS,
    "bug": BUG_MARKERS,
    "enhancement": ENHANCEMENT_MARKERS,
    "custom": CUSTOM_MARKERS,
}

__all__ = [
    "BUG_MARKERS",
    "CUSTOM_MARKERS",
    "DEFAULT_TEMPLATE_MARKERS",
    "ENHANCEMENT_MARKERS",
    "FEATURE_MARKERS",
]
