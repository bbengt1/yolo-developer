"""Unit test fixtures for environment isolation."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def clear_unit_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure unit tests do not inherit global config overrides."""
    monkeypatch.delenv("YOLO_PROJECT_NAME", raising=False)
