from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from yolo_developer.sdk import YoloClient


class ConnectionManager:
    def __init__(self) -> None:
        self.active: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        self.active.discard(websocket)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        for connection in list(self.active):
            try:
                await connection.send_json(payload)
            except Exception:
                await self.disconnect(connection)


def attach_websocket_routes(app: FastAPI) -> None:
    manager = ConnectionManager()

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        from yolo_developer.orchestrator.runtime_state import get_runtime_state_manager

        await manager.connect(websocket)
        client = YoloClient()
        runtime_manager = get_runtime_state_manager()

        try:
            while True:
                # Get both SDK status and runtime state
                status = await client.status_async()
                runtime_state = runtime_manager.get_state()

                # Build agent states for display
                agents = []
                for agent in runtime_state.agents:
                    display_name = agent.name.capitalize()
                    if display_name == "Tea":
                        display_name = "TEA"
                    elif display_name == "Pm":
                        display_name = "PM"
                    elif display_name == "Sm":
                        display_name = "SM"
                    agents.append({"name": display_name, "state": agent.state})

                # Build gates for display
                gates = [
                    {"name": g.name, "score": g.score}
                    for g in runtime_state.gates
                ] if runtime_state.gates else []

                payload = {
                    "event": "status.update",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": {
                        "project_name": status.project_name,
                        "workflow_status": runtime_state.workflow_status,
                        "active_agent": runtime_state.active_agent,
                        "stories_completed": runtime_state.stories_completed,
                        "stories_total": runtime_state.stories_total,
                        "eta_minutes": runtime_state.eta_minutes,
                        "agents": agents,
                        "gates": gates,
                    },
                }
                await websocket.send_json(payload)
                await asyncio.sleep(1)  # Poll more frequently for responsiveness
        except WebSocketDisconnect:
            await manager.disconnect(websocket)
