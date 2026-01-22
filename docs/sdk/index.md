---
layout: default
title: Python SDK
nav_order: 6
has_children: true
---

# Python SDK
{: .no_toc }

Programmatic API for integrating YOLO Developer into your applications.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

The YOLO Developer Python SDK provides a programmatic interface for all CLI functionality. Use it to integrate autonomous development into CI/CD pipelines, custom tooling, or larger automation workflows.

### Installation

The SDK is included with YOLO Developer:

```python
from yolo_developer import YoloClient
```

---

## Quick Start

```python
import asyncio
from yolo_developer import YoloClient

async def main():
    # Initialize client
    client = YoloClient()

    # Seed requirements
    seed_result = await client.seed("""
        Build a REST API with:
        - User registration and login
        - JWT authentication
        - Profile management
    """)
    print(f"Seed ID: {seed_result.seed_id}")

    # Run autonomous sprint
    sprint = await client.run()
    print(f"Sprint completed: {sprint.stories_completed} stories")

    # Check status
    status = await client.status()
    print(f"Coverage: {status.test_coverage}%")

asyncio.run(main())
```

---

## YoloClient

The main entry point for SDK functionality.

### Constructor

```python
YoloClient(
    config_path: str | Path | None = None,
    project_dir: str | Path | None = None,
)
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `config_path` | str \| Path | None | Path to yolo.yaml (auto-detected if None) |
| `project_dir` | str \| Path | None | Project directory (current dir if None) |

### Example

```python
from pathlib import Path
from yolo_developer import YoloClient

# Default (uses current directory)
client = YoloClient()

# Specific project
client = YoloClient(project_dir="/path/to/project")

# Specific config
client = YoloClient(config_path="/custom/yolo.yaml")
```

---

## Core Methods

### seed()

Seed requirements for autonomous development.

```python
async def seed(
    content: str | None = None,
    file_path: str | Path | None = None,
    validate: bool = True,
) -> SeedResult
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `content` | str | None | Requirements as text |
| `file_path` | str \| Path | None | Path to requirements file |
| `validate` | bool | True | Run validation checks |

**Returns:** `SeedResult`

```python
@dataclass
class SeedResult:
    seed_id: str
    content_length: int
    requirements_count: int
    categories: dict[str, int]
    quality_score: float
    ambiguities: list[Ambiguity]
    contradictions: list[Contradiction]
```

**Example:**

```python
# From text
result = await client.seed(content="Build a user management API")

# From file
result = await client.seed(file_path="requirements.md")

# Without validation
result = await client.seed(content="...", validate=False)

# Access results
print(f"Seed ID: {result.seed_id}")
print(f"Requirements: {result.requirements_count}")
print(f"Quality: {result.quality_score}")

if result.ambiguities:
    for amb in result.ambiguities:
        print(f"Ambiguity: {amb.text}")
        print(f"Question: {amb.clarification_question}")
```

---

### run()

Execute an autonomous development sprint.

```python
async def run(
    seed_id: str | None = None,
    agents: list[str] | None = None,
    max_iterations: int = 10,
    timeout: int = 300,
    dry_run: bool = False,
    on_progress: Callable[[SprintProgress], None] | None = None,
) -> SprintResult
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `seed_id` | str | None | Seed to use (latest if None) |
| `agents` | list[str] | None | Agents to run (all if None) |
| `max_iterations` | int | 10 | Max iterations per agent |
| `timeout` | int | 300 | Timeout seconds per agent |
| `dry_run` | bool | False | Simulate without changes |
| `on_progress` | Callable | None | Progress callback |

**Returns:** `SprintResult`

```python
@dataclass
class SprintResult:
    sprint_id: str
    status: SprintStatus
    duration_seconds: float
    stories_completed: int
    stories_total: int
    test_coverage: float
    quality_gates: dict[str, GateResult]
    tokens_used: int
    cost_usd: float
    artifacts: SprintArtifacts
```

**Example:**

```python
# Basic run
result = await client.run()

# Specific seed
result = await client.run(seed_id="seed_abc123")

# Only analysis agents
result = await client.run(agents=["analyst", "pm", "architect"])

# With progress callback
def on_progress(progress: SprintProgress):
    print(f"[{progress.agent}] {progress.message}")
    print(f"Progress: {progress.percent}%")

result = await client.run(on_progress=on_progress)

# Dry run
result = await client.run(dry_run=True)

# Access results
print(f"Duration: {result.duration_seconds}s")
print(f"Stories: {result.stories_completed}/{result.stories_total}")
print(f"Coverage: {result.test_coverage}%")
print(f"Cost: ${result.cost_usd:.2f}")
```

---

### status()

Get current sprint status.

```python
async def status(
    sprint_id: str | None = None,
    include_stories: bool = False,
    include_gates: bool = False,
) -> SprintStatus
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `sprint_id` | str | None | Sprint ID (current if None) |
| `include_stories` | bool | False | Include story details |
| `include_gates` | bool | False | Include gate details |

**Returns:** `SprintStatus`

```python
@dataclass
class SprintStatus:
    status: str  # "IN_PROGRESS", "COMPLETED", "FAILED"
    progress: float  # 0.0 to 1.0
    current_agent: str | None
    current_task: str | None
    stories: StoryProgress | None
    quality_gates: dict[str, GateResult] | None
    tokens_used: int
    elapsed_seconds: float
```

**Example:**

```python
# Basic status
status = await client.status()
print(f"Status: {status.status}")
print(f"Progress: {status.progress * 100}%")

# With details
status = await client.status(include_stories=True, include_gates=True)

for story in status.stories.items:
    print(f"  {story.id}: {story.title} [{story.status}]")

for gate_name, gate in status.quality_gates.items():
    print(f"  {gate_name}: {gate.score} ({'PASS' if gate.passing else 'FAIL'})")
```

---

### get_audit_trail()

Access the audit trail and decision history.

```python
async def get_audit_trail(
    agent: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 100,
) -> AuditTrail
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `agent` | str | None | Filter by agent |
| `since` | datetime | None | Start time filter |
| `until` | datetime | None | End time filter |
| `limit` | int | 100 | Maximum entries |

**Returns:** `AuditTrail`

```python
@dataclass
class AuditTrail:
    entries: list[AuditEntry]
    total_count: int

@dataclass
class AuditEntry:
    timestamp: datetime
    agent: str
    decision: str
    context: dict
    confidence: float
    tokens_used: int
```

**Example:**

```python
# All entries
audit = await client.get_audit_trail()

# Filter by agent
audit = await client.get_audit_trail(agent="architect")

# Filter by time
from datetime import datetime, timedelta
since = datetime.now() - timedelta(hours=1)
audit = await client.get_audit_trail(since=since)

# Process entries
for entry in audit.entries:
    print(f"[{entry.timestamp}] {entry.agent}: {entry.decision}")
    print(f"  Confidence: {entry.confidence}")
```

---

## Configuration API

### get_config()

Get current configuration.

```python
async def get_config() -> YoloConfig
```

**Example:**

```python
config = await client.get_config()
print(f"Project: {config.project_name}")
print(f"Coverage threshold: {config.quality.test_coverage_threshold}")
```

---

### update_config()

Update configuration values.

```python
async def update_config(
    **kwargs
) -> YoloConfig
```

**Example:**

```python
# Update single value
config = await client.update_config(
    quality__test_coverage_threshold=0.9
)

# Update multiple values
config = await client.update_config(
    project_name="my-api",
    quality__gate_pass_threshold=0.8,
    agents__max_iterations=15,
)
```

---

## Event Hooks

Register callbacks for agent events.

### on()

Register an event handler.

```python
def on(
    event: str,
    handler: Callable[[Event], Awaitable[None]] | Callable[[Event], None],
) -> None
```

**Available Events:**

| Event | Description | Payload |
|:------|:------------|:--------|
| `agent.started` | Agent started processing | `AgentStartedEvent` |
| `agent.completed` | Agent completed | `AgentCompletedEvent` |
| `agent.failed` | Agent failed | `AgentFailedEvent` |
| `story.started` | Story implementation started | `StoryStartedEvent` |
| `story.completed` | Story completed | `StoryCompletedEvent` |
| `gate.evaluated` | Quality gate evaluated | `GateEvaluatedEvent` |
| `sprint.started` | Sprint started | `SprintStartedEvent` |
| `sprint.completed` | Sprint completed | `SprintCompletedEvent` |

**Example:**

```python
from yolo_developer import YoloClient
from yolo_developer.events import AgentCompletedEvent, GateEvaluatedEvent

client = YoloClient()

@client.on("agent.completed")
async def on_agent_done(event: AgentCompletedEvent):
    print(f"Agent {event.agent} completed in {event.duration}s")
    print(f"Tokens used: {event.tokens_used}")

@client.on("gate.evaluated")
def on_gate(event: GateEvaluatedEvent):
    status = "PASS" if event.passing else "FAIL"
    print(f"Gate {event.gate_name}: {event.score:.2f} [{status}]")

@client.on("story.completed")
async def on_story(event):
    print(f"Completed: {event.story_id} - {event.title}")

# Run with event handlers
result = await client.run()
```

---

### remove_handler()

Remove a registered event handler.

```python
def remove_handler(event: str, handler: Callable) -> None
```

**Example:**

```python
def my_handler(event):
    print(event)

client.on("agent.completed", my_handler)
# ... later ...
client.remove_handler("agent.completed", my_handler)
```

---

## Error Handling

### Exception Types

```python
from yolo_developer.exceptions import (
    YoloError,           # Base exception
    ConfigurationError,  # Config issues
    ValidationError,     # Seed validation failed
    ApiError,            # LLM API errors
    GateFailureError,    # Quality gate failed
    TimeoutError,        # Operation timeout
)
```

### Example

```python
from yolo_developer import YoloClient
from yolo_developer.exceptions import (
    ValidationError,
    GateFailureError,
    ApiError,
)

client = YoloClient()

try:
    result = await client.seed(content="")
except ValidationError as e:
    print(f"Validation failed: {e.message}")
    for issue in e.issues:
        print(f"  - {issue}")

try:
    result = await client.run()
except GateFailureError as e:
    print(f"Gate failed: {e.gate_name}")
    print(f"Score: {e.score} (required: {e.threshold})")
    print(f"Remediation: {e.remediation}")
except ApiError as e:
    print(f"API error: {e.message}")
    print(f"Status code: {e.status_code}")
```

---

## Advanced Usage

### Context Manager

```python
async with YoloClient() as client:
    result = await client.seed(content="...")
    sprint = await client.run()
# Resources cleaned up automatically
```

### Parallel Operations

```python
import asyncio
from yolo_developer import YoloClient

async def process_projects(projects: list[str]):
    async def process_one(project_dir: str):
        client = YoloClient(project_dir=project_dir)
        await client.seed(file_path=f"{project_dir}/requirements.md")
        return await client.run()

    results = await asyncio.gather(*[
        process_one(p) for p in projects
    ])
    return results

# Process multiple projects in parallel
results = asyncio.run(process_projects([
    "/project/a",
    "/project/b",
    "/project/c",
]))
```

### Custom Configuration

```python
from yolo_developer import YoloClient
from yolo_developer.config import YoloConfig, LLMConfig, QualityConfig

# Create custom config
config = YoloConfig(
    project_name="my-api",
    llm=LLMConfig(
        cheap_model="gpt-4o-mini",
        premium_model="claude-sonnet-4-20250514",
        best_model="claude-opus-4-5-20251101",
    ),
    quality=QualityConfig(
        test_coverage_threshold=0.9,
        gate_pass_threshold=0.8,
    ),
)

client = YoloClient()
await client.update_config(**config.model_dump())
```

### Streaming Results

```python
async def stream_sprint():
    client = YoloClient()

    # Track progress in real-time
    async for event in client.run_streaming():
        if event.type == "progress":
            print(f"\r[{event.agent}] {event.percent}%", end="")
        elif event.type == "story_completed":
            print(f"\n✓ {event.story_id}: {event.title}")
        elif event.type == "completed":
            print(f"\nSprint completed!")
            return event.result
```

---

## Integration Examples

### CI/CD Pipeline

```python
# ci_script.py
import asyncio
import sys
from yolo_developer import YoloClient
from yolo_developer.exceptions import GateFailureError

async def run_quality_check():
    client = YoloClient()

    # Seed from PR description or requirements file
    await client.seed(file_path="requirements.md")

    try:
        # Run only analysis agents
        result = await client.run(
            agents=["analyst", "pm", "architect"],
            dry_run=True,  # Don't generate code
        )

        # Check quality gates
        status = await client.status(include_gates=True)

        all_passing = all(
            gate.passing for gate in status.quality_gates.values()
        )

        if not all_passing:
            print("Quality gates failed!")
            for name, gate in status.quality_gates.items():
                if not gate.passing:
                    print(f"  {name}: {gate.score:.2f} < {gate.threshold}")
            sys.exit(1)

        print("All quality gates passed!")

    except GateFailureError as e:
        print(f"Gate failure: {e.gate_name}")
        sys.exit(1)

asyncio.run(run_quality_check())
```

### Jupyter Notebook

```python
# In Jupyter
from yolo_developer import YoloClient

client = YoloClient()

# Seed interactively
result = await client.seed(content="""
Build a data pipeline that:
- Ingests CSV files from S3
- Transforms data with pandas
- Loads to PostgreSQL
- Runs on a schedule
""")

print(f"Parsed {result.requirements_count} requirements")
print(f"Quality score: {result.quality_score}")

# Run and display progress
from IPython.display import display, clear_output

def show_progress(progress):
    clear_output(wait=True)
    display(f"[{progress.agent}] {progress.message}")
    display(f"Progress: {'█' * int(progress.percent / 5)}{'░' * (20 - int(progress.percent / 5))} {progress.percent}%")

sprint = await client.run(on_progress=show_progress)
```

---

## Type Reference

All SDK types are fully typed and available for import:

```python
from yolo_developer.types import (
    SeedResult,
    SprintResult,
    SprintStatus,
    SprintProgress,
    AuditTrail,
    AuditEntry,
    StoryProgress,
    GateResult,
    Ambiguity,
    Contradiction,
)

from yolo_developer.events import (
    AgentStartedEvent,
    AgentCompletedEvent,
    AgentFailedEvent,
    StoryStartedEvent,
    StoryCompletedEvent,
    GateEvaluatedEvent,
    SprintStartedEvent,
    SprintCompletedEvent,
)
```

---

## Next Steps

- [Configuration](/yolo-developer/configuration/) - All config options
- [Architecture](/yolo-developer/architecture/) - How the system works
- [CLI Reference](/yolo-developer/cli/) - Command-line usage
