from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from yolo_developer.web.api import api_router
from yolo_developer.web.websocket import attach_websocket_routes


def create_app(api_only: bool = False) -> FastAPI:
    app = FastAPI(title="YOLO Developer Web")
    app.include_router(api_router, prefix="/api/v1")
    attach_websocket_routes(app)

    if not api_only:
        static_dir = Path(__file__).parent / "static"
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(static_dir / "index.html")

    return app
