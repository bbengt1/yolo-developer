"""CLI command to launch the web UI."""

from __future__ import annotations

import typer
import uvicorn

from yolo_developer.config import load_config
from yolo_developer.web.app import create_app

app = typer.Typer(name="web", help="Start the YOLO Developer web UI")


@app.command()
def start(
    host: str | None = typer.Option(None, "--host", help="Host to bind"),
    port: int | None = typer.Option(None, "--port", help="Port to bind"),
    api_only: bool | None = typer.Option(None, "--api-only", help="Run API only"),
) -> None:
    config = load_config()
    web_config = config.web
    uvicorn.run(
        create_app(api_only=api_only if api_only is not None else web_config.api_only),
        host=host or web_config.host,
        port=port or web_config.port,
    )
