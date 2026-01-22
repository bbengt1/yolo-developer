from __future__ import annotations

import asyncio
from datetime import datetime, timezone

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

    async def broadcast(self, payload: dict) -> None:
        for connection in list(self.active):
            try:
                await connection.send_json(payload)
            except Exception:
                await self.disconnect(connection)


def attach_websocket_routes(app: FastAPI) -> None:
    manager = ConnectionManager()

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await manager.connect(websocket)
        client = YoloClient()
        try:
            while True:
                status = client.status()
                payload = {
                    "event": "status.update",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": {
                        "project_name": status.project_name,
                        "workflow_status": status.workflow_status,
                        "active_agent": status.active_agent,
                    },
                }
                await websocket.send_json(payload)
                await asyncio.sleep(3)
        except WebSocketDisconnect:
            await manager.disconnect(websocket)
