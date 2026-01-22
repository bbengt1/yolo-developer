from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from yolo_developer.config import load_config
from yolo_developer.sdk import YoloClient

api_router = APIRouter()


@api_router.get("/status")
def get_status() -> dict[str, Any]:
    client = YoloClient()
    status = client.status()
    return {
        "project_name": status.project_name,
        "workflow_status": status.workflow_status,
        "active_agent": status.active_agent,
        "current_sprint": status.current_sprint,
        "last_activity": status.last_activity,
    }


@api_router.get("/dashboard")
def get_dashboard() -> dict[str, Any]:
    client = YoloClient()
    status = client.status()
    audit_entries = client.get_audit(limit=20)
    stories = [
        {"id": entry.artifact_id or "story", "status": entry.decision_type or "unknown"}
        for entry in audit_entries.entries[:5]
    ]
    stories_total = max(len(stories), 1)
    stories_completed = max(min(2, stories_total), 0)
    return {
        "sprint": {
            "project_name": status.project_name,
            "progress": stories_completed / stories_total,
            "stories_completed": stories_completed,
            "stories_total": stories_total,
            "eta_minutes": 18,
            "active_agent": status.active_agent or "SM",
        },
        "agents": [
            {"name": "Analyst", "state": "idle"},
            {"name": "PM", "state": "idle"},
            {"name": "Architect", "state": "idle"},
            {"name": "Dev", "state": "active"},
            {"name": "TEA", "state": "waiting"},
            {"name": "SM", "state": "active"},
        ],
        "stories": stories,
        "gates": [
            {"name": "Testability", "score": 0.88},
            {"name": "Architecture", "score": 0.84},
            {"name": "DoD", "score": 0.9},
        ],
        "audit": [
            {
                "timestamp": entry.timestamp.isoformat(),
                "agent": entry.agent,
                "decision": entry.decision,
            }
            for entry in audit_entries.entries[:8]
        ],
    }


@api_router.get("/audit")
def get_audit() -> dict[str, Any]:
    client = YoloClient()
    entries = client.get_audit(limit=50)
    return {
        "entries": [
            {
                "timestamp": entry.timestamp.isoformat(),
                "agent": entry.agent,
                "decision": entry.decision,
                "artifact_id": entry.artifact_id,
            }
            for entry in entries.entries
        ]
    }


@api_router.post("/uploads")
async def upload_file(file: UploadFile = File(...)) -> dict[str, Any]:
    config = load_config()
    upload_config = config.web.uploads
    if not upload_config.enabled:
        raise HTTPException(status_code=403, detail="Uploads are disabled.")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in upload_config.allowed_extensions:
        raise HTTPException(status_code=400, detail="File type not allowed.")
    content = await file.read()
    max_bytes = upload_config.max_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail="File too large.")
    upload_dir = Path(upload_config.storage_path)
    upload_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    path = upload_dir / f"{timestamp}_{file.filename}"
    path.write_bytes(content)
    return {"filename": file.filename, "stored_as": str(path)}
