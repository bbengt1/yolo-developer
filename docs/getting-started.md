---
layout: default
title: Getting Started
nav_order: 2
---

# Getting Started
{: .no_toc }

Get up and running with YOLO Developer in under 5 minutes.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Prerequisites

Before installing YOLO Developer, ensure you have:

- **Python 3.10 or higher** - Check with `python --version`
- **uv package manager** - Install from [astral.sh/uv](https://astral.sh/uv)
- **API Key** - OpenAI or Anthropic API key for LLM access

---

## Step 1: Install YOLO Developer

```bash
# Clone the repository
git clone https://github.com/bbengt1/yolo-developer.git
cd yolo-developer

# Install dependencies
uv sync

# Verify installation
uv run yolo --help
```

**Expected output:**
```
Usage: yolo [OPTIONS] COMMAND [ARGS]...

  YOLO Developer - Autonomous AI Development System

Options:
  --version  Show version and exit.
  --help     Show this message and exit.

Commands:
  config  Manage project configuration.
  init    Initialize a new YOLO project.
  logs    View agent activity logs.
  mcp     Start MCP server for Claude Code.
  run     Execute autonomous development sprint.
  seed    Seed requirements for development.
  status  Display current sprint status.
  tune    Adjust quality thresholds.
```

---

## Step 2: Configure API Keys

YOLO Developer requires an LLM API key. Set it via environment variable:

{: .warning }
> Never put API keys in configuration files. Always use environment variables.

### OpenAI

```bash
export YOLO_LLM__OPENAI_API_KEY=sk-proj-...
```

### Anthropic

```bash
export YOLO_LLM__ANTHROPIC_API_KEY=sk-ant-api03-...
```

### Verify Configuration

```bash
uv run yolo config show
```

**Expected output:**
```yaml
project_name: my-project
llm:
  smart_model: gpt-4o
  routine_model: gpt-4o-mini
  openai_api_key: sk-proj-****  # Masked
quality:
  test_coverage_threshold: 0.8
  gate_pass_threshold: 0.7
```

---

## Step 3: Initialize a Project

Navigate to your project directory and initialize:

```bash
cd /path/to/your/project
uv run yolo init
```

**Interactive output:**
```
Initializing YOLO Developer project...

? Project name: my-awesome-api
? Primary language: Python
? Test framework: pytest
? Include CI/CD templates? Yes

Creating configuration...
  ✓ Created yolo.yaml
  ✓ Created .yolo/ directory
  ✓ Initialized memory store

Project initialized successfully!

Next steps:
  1. Review yolo.yaml configuration
  2. Create a requirements document
  3. Run: yolo seed <requirements-file>
```

This creates:
- `yolo.yaml` - Project configuration
- `.yolo/` - Memory and cache directory
- `.yolo/memory/` - ChromaDB vector storage

---

## Step 4: Create Requirements

Create a requirements document. YOLO Developer accepts Markdown or plain text:

**requirements.md:**
```markdown
# User Management API

## Overview
Build a REST API for user management with authentication.

## Requirements

### Functional
- Users can register with email and password
- Users can login and receive JWT token
- Users can view and update their profile
- Admins can list and manage all users

### Non-Functional
- Response time < 200ms for all endpoints
- Support 1000 concurrent users
- 99.9% uptime SLA

### Constraints
- Must use PostgreSQL database
- Must follow OpenAPI 3.0 specification
- Must include rate limiting
```

---

## Step 5: Seed Requirements

Feed your requirements to YOLO Developer:

```bash
uv run yolo seed requirements.md
```

**Output:**
```
Parsing requirements document...
  ✓ Parsed 4 functional requirements
  ✓ Parsed 3 non-functional requirements
  ✓ Parsed 3 constraints

Running validation...
  ✓ No ambiguities detected
  ✓ No contradictions found
  ✓ SOP constraints validated

Seed Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Requirements: 10 total
  Categories:
    - Functional: 4
    - Non-Functional: 3
    - Constraints: 3
  Quality Score: 0.92

Seed ID: seed_a1b2c3d4-e5f6-7890-abcd-ef1234567890

Ready to run: yolo run
```

---

## Step 6: Run Autonomous Development

Start the autonomous development sprint:

```bash
uv run yolo run
```

**Real-time output:**
```
Starting autonomous development sprint...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[ANALYST] Analyzing requirements...
  → Crystallizing requirement: "Users can register with email and password"
  → Crystallizing requirement: "Users can login and receive JWT token"
  ✓ Crystallized 4 requirements into 6 detailed specifications

[PM] Generating user stories...
  → Creating story: "User Registration"
  → Creating story: "User Authentication"
  → Creating story: "Profile Management"
  → Creating story: "Admin User Management"
  ✓ Generated 8 user stories with acceptance criteria

[ARCHITECT] Designing system architecture...
  → Analyzing 12-Factor compliance
  → Generating ADR: "JWT Authentication Strategy"
  → Generating ADR: "Database Schema Design"
  → Evaluating quality attributes
  ✓ Created 3 ADRs, identified 2 technical risks

[DEV] Implementing stories...
  → Implementing: US-001 User Registration
    - Creating models/user.py
    - Creating routes/auth.py
    - Creating tests/test_auth.py
  ✓ Completed US-001 (3 files, 89% coverage)

[TEA] Validating test coverage...
  → Running test suite
  → Analyzing coverage gaps
  ✓ Coverage: 87% (threshold: 80%)

[SM] Sprint complete!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Sprint Summary:
  Duration: 12m 34s
  Stories Completed: 8/8
  Test Coverage: 87%
  Quality Gates: 4/4 passing
  Token Usage: 125,430 tokens ($1.23)

Artifacts generated:
  - 3 Architecture Decision Records
  - 8 User Stories with Acceptance Criteria
  - 24 Source Files
  - 45 Test Files
  - API Documentation (OpenAPI 3.0)
```

---

## Step 7: Review Results

### Check Status

```bash
uv run yolo status
```

### View Logs

```bash
# All logs
uv run yolo logs

# Filter by agent
uv run yolo logs --agent dev

# Filter by time
uv run yolo logs --since 1h
```

### Export Audit Trail

```bash
uv run yolo logs --export audit.json
```

---

## Next Steps

Now that you have YOLO Developer running:

1. **[CLI Reference](/yolo-developer/cli/)** - Learn all available commands
2. **[Configuration Guide](/yolo-developer/configuration/)** - Customize behavior
3. **[MCP Integration](/yolo-developer/mcp/)** - Use with Claude Code
4. **[Architecture Deep Dive](/yolo-developer/architecture/)** - Understand how it works

---

## Troubleshooting

### "API key not found"

Ensure your API key is set correctly:
```bash
echo $YOLO_LLM__OPENAI_API_KEY
```

### "ChromaDB initialization failed"

Clear the memory directory and reinitialize:
```bash
rm -rf .yolo/memory
uv run yolo init --reinit
```

### "Quality gate failed"

View the gate failure report:
```bash
uv run yolo status --gates
```

Adjust thresholds if needed:
```bash
uv run yolo tune --coverage 0.7
```
