from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from yolo_developer.config import ConfigurationError, YoloConfig, load_config
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
    """Get full dashboard data including runtime state.

    Returns real-time workflow status, agent states, quality gates,
    and audit entries from the runtime state tracker.
    """
    from yolo_developer.orchestrator.runtime_state import get_runtime_state_manager

    client = YoloClient()
    status = client.status()
    audit_entries = client.get_audit(limit=20)

    # Get runtime state for live data
    runtime_manager = get_runtime_state_manager()
    runtime_state = runtime_manager.get_state()

    # Build stories from runtime state or audit entries
    stories = []
    if runtime_state.stories:
        stories = [
            {"id": s.story_id, "status": s.status}
            for s in runtime_state.stories
        ]
    else:
        stories = [
            {
                "id": entry.entry_id or "story",
                "status": entry.decision_type or "unknown",
            }
            for entry in audit_entries[:5]
        ]

    stories_total = runtime_state.stories_total or max(len(stories), 1)
    stories_completed = runtime_state.stories_completed
    progress = stories_completed / stories_total if stories_total > 0 else 0.0

    # Build agent states from runtime state
    agents = []
    if runtime_state.agents:
        for agent in runtime_state.agents:
            # Capitalize agent name for display
            display_name = agent.name.capitalize()
            if display_name == "Tea":
                display_name = "TEA"
            elif display_name == "Pm":
                display_name = "PM"
            elif display_name == "Sm":
                display_name = "SM"
            agents.append({"name": display_name, "state": agent.state})
    else:
        # Default fallback
        agents = [
            {"name": "Analyst", "state": "idle"},
            {"name": "PM", "state": "idle"},
            {"name": "Architect", "state": "idle"},
            {"name": "Dev", "state": "idle"},
            {"name": "TEA", "state": "idle"},
            {"name": "SM", "state": "idle"},
        ]

    # Build gates from runtime state
    gates = []
    if runtime_state.gates:
        gates = [
            {"name": g.name, "score": g.score}
            for g in runtime_state.gates
        ]
    else:
        # Default placeholder gates (no data yet)
        gates = [
            {"name": "Testability", "score": 0.0},
            {"name": "Architecture", "score": 0.0},
            {"name": "DoD", "score": 0.0},
        ]

    return {
        "sprint": {
            "project_name": status.project_name,
            "progress": progress,
            "stories_completed": stories_completed,
            "stories_total": stories_total,
            "eta_minutes": runtime_state.eta_minutes,
            "active_agent": runtime_state.active_agent or status.active_agent,
            "workflow_status": runtime_state.workflow_status,
        },
        "agents": agents,
        "stories": stories,
        "gates": gates,
        "audit": [
            {
                "timestamp": entry.timestamp.isoformat(),
                "agent": entry.agent,
                "decision": entry.content or entry.decision_type,
            }
            for entry in audit_entries[:8]
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
                "decision": entry.content or entry.decision_type,
                "entry_id": entry.entry_id,
            }
            for entry in entries
        ]
    }


@api_router.post("/uploads")
async def upload_file(file: UploadFile = File(...)) -> dict[str, Any]:
    config = _load_config_or_default()
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


def _load_config_or_default() -> YoloConfig:
    try:
        return load_config()
    except ConfigurationError:
        return YoloConfig(project_name="web")
