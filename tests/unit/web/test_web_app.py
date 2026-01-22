from __future__ import annotations

import pytest


def test_create_app() -> None:
    pytest.importorskip("fastapi")
    from yolo_developer.web.app import create_app

    app = create_app(api_only=True)
    assert app.title == "YOLO Developer Web"
