---
layout: default
title: MCP Integration
nav_order: 5
has_children: true
---

# MCP Integration
{: .no_toc }

Use YOLO Developer with Claude Code and other MCP-compatible clients.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

YOLO Developer exposes an MCP (Model Context Protocol) server that allows AI assistants like Claude to directly interact with the autonomous development system. This enables conversational development workflows where you can seed requirements, run sprints, and monitor progress through natural language.

### What is MCP?

MCP (Model Context Protocol) is a standard protocol for AI assistants to interact with external tools and services. When integrated with Claude Code or Claude Desktop, YOLO Developer becomes an extension of the AI assistant's capabilities.

### Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Claude Code    │────▶│   MCP Server    │────▶│ YOLO Developer  │
│  (MCP Client)   │◀────│  (FastMCP 2.x)  │◀────│    (Backend)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        │    Tool Calls         │    Function Calls     │
        │◀─────────────────────▶│◀─────────────────────▶│
        │                       │                       │
```

---

## Quick Start

### 1. Start the MCP Server

```bash
# STDIO transport (for Claude Desktop)
yolo mcp

# HTTP transport (for remote access)
yolo mcp --transport http --port 8080
```

### 1a. Quick MCP Tool Verification (CLI)

This uses an in-process FastMCP client to list tool names. Use this for local sanity checks; for real clients (Claude Desktop or HTTP) you still need a running MCP server.

```bash
uv run python - <<'PY'
import asyncio
from fastmcp import Client
from yolo_developer.mcp import mcp

async def main() -> None:
    async with Client(mcp) as client:
        tools = await client.list_tools()
        print([t.name for t in tools])

asyncio.run(main())
PY
```

### 2. Configure Claude Desktop

Add to your Claude Desktop configuration file:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

**Linux:** `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "yolo-developer": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/yolo-developer", "yolo", "mcp"]
    }
  }
}
```

### 3. Restart Claude Desktop

Close and reopen Claude Desktop. You should see YOLO Developer tools available.

### 4. Start Using

In Claude Desktop, you can now say:

> "Use yolo_seed to start a new project with requirements for a REST API with user authentication"

---

## Available MCP Tools

### yolo_seed

Provide seed requirements for autonomous development.

#### Parameters

| Parameter | Type | Required | Description |
|:----------|:-----|:---------|:------------|
| `content` | string | No* | Requirements as plain text |
| `file_path` | string | No* | Path to requirements file |

*One of `content` or `file_path` must be provided.

#### Returns

```json
{
  "status": "accepted",
  "seed_id": "seed_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "content_length": 1234,
  "source": "text"
}
```

#### Example Usage in Claude

**User prompt:**
> Seed these requirements for my new project:
> - User registration with email verification
> - OAuth2 login with Google and GitHub
> - Profile management with avatar upload
> - Admin dashboard for user management

**Claude response:**
> I'll seed these requirements using YOLO Developer.
>
> [Calls yolo_seed with content]
>
> Your requirements have been seeded successfully:
> - **Seed ID:** seed_a1b2c3d4...
> - **Requirements parsed:** 4
> - **Ready for:** `yolo run`
>
> Would you like me to start the autonomous development sprint?

#### Error Responses

**Empty content:**
```json
{
  "status": "error",
  "error": "Content cannot be empty or whitespace-only"
}
```

**File not found:**
```json
{
  "status": "error",
  "error": "File not found: /path/to/missing.md"
}
```

**No input provided:**
```json
{
  "status": "error",
  "error": "Either content or file_path must be provided"
}
```

---

### yolo_run

Execute an autonomous development sprint.

#### Parameters

| Parameter | Type | Required | Description |
|:----------|:-----|:---------|:------------|
| `seed_id` | string | No* | Seed ID returned by `yolo_seed` (preferred) |
| `seed_content` | string | No* | Raw seed content (used if `seed_id` not provided) |

*One of `seed_id` or `seed_content` must be provided.

#### Returns

```json
{
  "status": "started",
  "sprint_id": "sprint-abcdef12",
  "seed_id": "550e8400-e29b-41d4-a716-446655440000",
  "thread_id": "thread-1234abcd",
  "started_at": "2026-01-21T09:12:34.567890+00:00"
}
```

#### Error Responses

```json
{
  "status": "error",
  "error": "Seed not found for seed_id: ..."
}
```

---

### yolo_status

Query current sprint status.

#### Parameters

| Parameter | Type | Required | Description |
|:----------|:-----|:---------|:------------|
| `sprint_id` | string | Yes | Sprint identifier returned by `yolo_run` |

#### Returns

```json
{
  "status": "running",
  "sprint_id": "sprint-abcdef12",
  "seed_id": "550e8400-e29b-41d4-a716-446655440000",
  "thread_id": "thread-1234abcd",
  "started_at": "2026-01-21T09:12:34.567890+00:00",
  "completed_at": null,
  "error": null
}
```

#### Error Responses

```json
{
  "status": "error",
  "error": "Sprint not found"
}
```

---

## yolo_status Walkthrough

1) Seed requirements
```bash
cat <<'EOF' > requirements.md
Build a REST API for user management with JWT authentication.
EOF
yolo seed requirements.md
```

2) Start a sprint and capture `sprint_id`
```json
{
  "status": "started",
  "sprint_id": "sprint-abcdef12",
  "seed_id": "550e8400-e29b-41d4-a716-446655440000",
  "thread_id": "thread-1234abcd",
  "started_at": "2026-01-21T09:12:34.567890+00:00"
}
```

3) Query sprint status
```json
{
  "status": "running",
  "sprint_id": "sprint-abcdef12",
  "seed_id": "550e8400-e29b-41d4-a716-446655440000",
  "thread_id": "thread-1234abcd",
  "started_at": "2026-01-21T09:12:34.567890+00:00",
  "completed_at": null,
  "error": null
}
```

---

### yolo_audit (Coming Soon)

Access audit trail and decision history.

#### Parameters

| Parameter | Type | Required | Description |
|:----------|:-----|:---------|:------------|
| `agent` | string | No | Filter by agent |
| `since` | string | No | ISO timestamp filter |
| `limit` | integer | No | Maximum entries |

#### Returns

```json
{
  "entries": [
    {
      "timestamp": "2024-01-15T10:23:45Z",
      "agent": "ANALYST",
      "decision": "Crystallized requirement",
      "context": {
        "original": "User registration",
        "output": "3 detailed specifications"
      }
    }
  ],
  "total_count": 45
}
```

---

## Transport Options

### STDIO Transport (Default)

Used for local integration with Claude Desktop.

```bash
yolo mcp
# or explicitly
yolo mcp --transport stdio
```

**Advantages:**
- No network configuration
- Secure (no open ports)
- Automatic lifecycle management

**Use case:** Claude Desktop on the same machine

---

### HTTP Transport

Used for remote or networked access.

```bash
yolo mcp --transport http --port 8080 --host 0.0.0.0
```

**Advantages:**
- Remote access
- Multiple clients
- Load balancing possible

**Use case:** Shared development server, CI/CD integration

{: .warning }
> HTTP transport should only be used on trusted networks. Consider using a reverse proxy with authentication for production deployments.

---

## Configuration

### Server Configuration

The MCP server respects `yolo.yaml` configuration:

```yaml
mcp:
  # Server name shown to clients
  name: "YOLO Developer"

  # Default transport
  transport: stdio

  # HTTP settings (when using http transport)
  http:
    port: 8080
    host: "127.0.0.1"

  # Error handling
  mask_errors: true  # Hide internal errors from clients
```

### Environment Variables

| Variable | Description | Default |
|:---------|:------------|:--------|
| `YOLO_MCP__TRANSPORT` | Transport type | stdio |
| `YOLO_MCP__HTTP__PORT` | HTTP port | 8080 |
| `YOLO_MCP__HTTP__HOST` | HTTP host | 127.0.0.1 |

---

## Claude Desktop Configuration

### Basic Configuration

```json
{
  "mcpServers": {
    "yolo-developer": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/yolo-developer", "yolo", "mcp"]
    }
  }
}
```

### With Environment Variables

```json
{
  "mcpServers": {
    "yolo-developer": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/yolo-developer", "yolo", "mcp"],
      "env": {
        "YOLO_LLM__OPENAI_API_KEY": "sk-proj-..."
      }
    }
  }
}
```

### Multiple Projects

```json
{
  "mcpServers": {
    "yolo-project-a": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/project-a", "yolo", "mcp"]
    },
    "yolo-project-b": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/project-b", "yolo", "mcp"]
    }
  }
}
```

---

## Usage Examples

### Example 1: Complete Development Workflow

**User:**
> I want to build a todo list API. Here are my requirements:
> - CRUD operations for todo items
> - User authentication with JWT
> - Mark todos as complete
> - Due date reminders
> - Categories/tags for organization
>
> Please seed these and start development.

**Claude with YOLO Developer:**
> I'll use YOLO Developer to handle this project. Let me start by seeding your requirements.
>
> [Calls yolo_seed]
>
> Requirements seeded successfully (Seed ID: seed_abc123). I found:
> - 5 functional requirements
> - Implicit non-functional requirements added (security, performance)
>
> Starting the autonomous development sprint...
>
> [Calls yolo_run]
>
> Sprint started. Here's the identifier you can use to track progress:
>
> ```json
> {
>   "status": "started",
>   "sprint_id": "sprint-abcdef12",
>   "seed_id": "seed_abc123",
>   "thread_id": "thread-1234abcd",
>   "started_at": "2026-01-21T09:12:34.567890+00:00"
> }
> ```
>
> Want me to check the sprint status now?

---

### Example 2: Checking Status Mid-Sprint

**User:**
> What's the current status of my development sprint?

**Claude:**
> Let me check the sprint status.
>
> [Calls yolo_status]
>
> ```json
> {
>   "status": "running",
>   "sprint_id": "sprint-abcdef12",
>   "seed_id": "seed_abc123",
>   "thread_id": "thread-1234abcd",
>   "started_at": "2026-01-21T09:12:34.567890+00:00",
>   "completed_at": null,
>   "error": null
> }
> ```
>
> The sprint is still running. I can check again or wait for completion.

---

### Example 3: Reviewing Decisions

**User:**
> Why did the architect choose JWT over session-based auth?

**Claude:**
> Let me check the audit trail for that decision.
>
> [Calls yolo_audit with agent="ARCHITECT"]
>
> **ADR-001: Authentication Strategy**
>
> **Decision:** Use JWT (JSON Web Tokens) for authentication
>
> **Context:**
> - Requirements specify stateless API
> - Multiple client types expected (web, mobile)
> - Scalability is a concern
>
> **Rationale:**
> 1. JWTs are stateless - no server-side session storage needed
> 2. Works well with horizontal scaling
> 3. Standard format supported by all client platforms
> 4. Can embed user claims for authorization
>
> **Trade-offs:**
> - Cannot invalidate tokens before expiry (mitigated with short expiry + refresh tokens)
> - Larger payload than session cookies
>
> **Confidence Score:** 0.92
>
> Would you like me to explain any other architectural decisions?

---

## Troubleshooting

### Server Won't Start

**Problem:** `Error: Address already in use`

**Solution:**
```bash
# Find process using the port
lsof -i :8080

# Kill the process or use different port
yolo mcp --transport http --port 8081
```

### Claude Desktop Doesn't Show Tools

**Problem:** Tools not appearing in Claude Desktop

**Solutions:**
1. Verify config file path is correct
2. Ensure uv is in PATH
3. Check Claude Desktop logs for errors
4. Restart Claude Desktop completely

**Debug:**
```bash
# Test server manually
yolo mcp

# In another terminal, verify it starts without errors
```

### Connection Refused

**Problem:** `Connection refused` when using HTTP transport

**Solution:**
```bash
# Ensure server is running
yolo mcp --transport http --port 8080

# Check if port is accessible
curl http://localhost:8080/health
```

### Authentication Errors

**Problem:** `API key not configured`

**Solution:**
Ensure API keys are set before starting MCP server:
```bash
export YOLO_LLM__OPENAI_API_KEY=sk-...
yolo mcp
```

Or in Claude Desktop config:
```json
{
  "mcpServers": {
    "yolo-developer": {
      "command": "uv",
      "args": ["..."],
      "env": {
        "YOLO_LLM__OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

---

## Best Practices

### 1. Use STDIO for Local Development

STDIO transport is more secure and simpler for local use:
```bash
yolo mcp  # defaults to stdio
```

### 2. Set API Keys in Environment

Don't hardcode API keys in config files:
```bash
export YOLO_LLM__OPENAI_API_KEY=sk-...
```

### 3. Monitor Token Usage

YOLO Developer tracks token usage. Check periodically:
```bash
yolo status --format json | jq '.tokens'
```

### 4. Use Specific Seed IDs

When resuming work, reference specific seeds:
> "Run the sprint for seed_abc123"

### 5. Review Before Deploying

Always review generated code before deploying:
> "Show me the authentication implementation from the last sprint"

---

## Next Steps

- [Python SDK](/yolo-developer/sdk/) - Programmatic integration
- [Configuration](/yolo-developer/configuration/) - Customize MCP behavior
- [Architecture](/yolo-developer/architecture/) - How agents work
