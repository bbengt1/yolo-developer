---
layout: default
title: Home
nav_order: 1
description: "YOLO Developer - Autonomous multi-agent AI development system"
permalink: /
---

# YOLO Developer Documentation
{: .fs-9 }

Autonomous multi-agent AI development system that orchestrates specialized AI agents to handle software development from requirements to implementation.
{: .fs-6 .fw-300 }

[Get Started](/yolo-developer/getting-started){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/bbengt1/yolo-developer){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## What is YOLO Developer?

YOLO Developer is an autonomous software development system that uses multiple specialized AI agents working together through a LangGraph-based orchestration engine. Each agent has a specific role:

| Agent | Role | Responsibilities |
|:------|:-----|:-----------------|
| **Analyst** | Requirements Engineering | Crystallize vague inputs, detect ambiguities, flag contradictions |
| **PM** | Product Management | Generate user stories, prioritize backlog, identify dependencies |
| **Architect** | System Design | Create ADRs, validate architecture, assess technical risks |
| **Dev** | Development | Generate code, write tests, follow patterns |
| **TEA** | Test Engineering | Validate coverage, categorize risks, audit testability |
| **SM** | Scrum Master | Orchestrate sprints, mediate conflicts, manage handoffs |

## Key Features

### Multi-Agent Orchestration
The agents communicate through a shared state machine, passing context and decisions between each other. The SM agent monitors health, detects circular logic, and escalates to humans when needed.

### Quality Gates
Every agent output passes through configurable quality gates that enforce standards for testability, architecture compliance, and definition of done.

### Memory & Learning
ChromaDB-backed vector storage enables semantic search across decisions, while pattern learning automatically detects your codebase conventions.

### Brownfield Support
Scan existing projects to detect language, frameworks, structure, and testing setup, then generate `.yolo/project-context.yaml` for agent guidance.

### GitHub Automation
Manage branches, commits, PRs, issues, and releases directly from YOLO Developer.

### Issue Import
Convert GitHub issues into structured user stories for sprint planning.

### Full Observability
Complete audit trail with decision logging, token cost tracking, and requirement traceability from seed to implementation.

### Interactive Gathering
Run guided Q&A sessions to crystallize requirements before seeding.

### Web Dashboard
Use the local web UI to monitor sprint status and agent activity.

![Web Dashboard Preview](/yolo-developer/assets/images/dashboard.svg)

---

## Quick Example

```bash
# Initialize a project
yolo init --name my-api

# Seed requirements
yolo seed requirements.md

# Run autonomous development
yolo run

# Check progress
yolo status
```

**Output:**
```
Sprint Status: IN_PROGRESS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Stories: 3/8 completed
Current: [DEV] Implementing user authentication endpoint

Agent Activity:
  ✓ Analyst: Crystallized 12 requirements (2m 34s)
  ✓ PM: Generated 8 user stories (1m 12s)
  ✓ Architect: Created 3 ADRs, validated 12-Factor (3m 45s)
  → Dev: Implementing story US-003 (in progress)
  ○ TEA: Pending
  ○ SM: Monitoring

Quality Gates: 4/4 passing
Token Usage: 45,230 tokens ($0.42)
```

---

## Documentation Sections

<div class="code-example" markdown="1">

### [Installation Guide](/yolo-developer/installation)
Complete installation instructions for all platforms and environments.

### [CLI Reference](/yolo-developer/cli/)
Detailed documentation for all CLI commands with examples and options.

### [MCP Integration](/yolo-developer/mcp/)
Guide to using YOLO Developer with Claude Code and other MCP clients.

### [Python SDK](/yolo-developer/sdk/)
Programmatic API reference with code examples.

### [Configuration](/yolo-developer/configuration/)
All configuration options, environment variables, and best practices.

### [Architecture](/yolo-developer/architecture/)
Deep dive into agents, orchestration, memory, and quality gates.

### [Brownfield Guide](/yolo-developer/guides/brownfield)
Scan and integrate existing projects with generated context.

### [GitHub Automation](/yolo-developer/guides/github)
Manage GitHub workflow automation from YOLO Developer.

### [Issue Import](/yolo-developer/guides/issue-import)
Import GitHub issues and generate user stories.

### [Interactive Gathering](/yolo-developer/guides/gathering)
Guided requirements elicitation sessions.

### [Web Dashboard](/yolo-developer/guides/web)
Local web UI for sprint visualization.

</div>

---

## System Requirements

| Requirement | Minimum | Recommended |
|:------------|:--------|:------------|
| Python | 3.10 - 3.13 | 3.12 - 3.13 |
| Memory | 4 GB | 8 GB |
| Disk | 500 MB | 2 GB (with memory persistence) |
| OS | macOS, Linux, Windows (WSL2) | macOS, Linux |

---

## Roadmap

### Current Status

| Epic | Status | Description |
|:-----|:-------|:------------|
| 1-13 | ✅ Complete | Core infrastructure, all agents, CLI, SDK |
| 14 | ✅ Complete | MCP integration + Codex compatibility |
| 2 | ✅ Complete | Brownfield project support |
| 12 | ✅ Complete | GitHub repository management |

### Recently Completed

- [#8](https://github.com/bbengt1/yolo-developer/issues/8) ChatGPT Codex Support (OpenAI/Codex provider + hybrid routing)
- MCP integration tools, walkthroughs, and audit access
- [#14](https://github.com/bbengt1/yolo-developer/issues/14) Interactive requirements gathering sessions
- [#3](https://github.com/bbengt1/yolo-developer/issues/3) Web interface with dashboard UI
- [#7](https://github.com/bbengt1/yolo-developer/issues/7) Sprint visualization dashboard

### Planned Features

#### LLM Providers

| Issue | Feature | Description |
|:------|:--------|:------------|
| [#1](https://github.com/bbengt1/yolo-developer/issues/1) | **Local LLM Support** | Ollama, LM Studio, vLLM integration with hybrid routing |

#### IDE Integrations

| Issue | Feature | Description |
|:------|:--------|:------------|
| [#9](https://github.com/bbengt1/yolo-developer/issues/9) | **Cursor IDE Support** | VS Code extension with MCP integration for Cursor |
| [#10](https://github.com/bbengt1/yolo-developer/issues/10) | **GitHub Copilot Support** | `@yolo` chat participant and Copilot Workspace integration |

#### User Interfaces

| Issue | Feature | Description |
|:------|:--------|:------------|
| (complete) | **Web Dashboard** | Local UI with REST API, WebSocket updates, and sprint visualization |

#### Core Enhancements

| Issue | Feature | Description |
|:------|:--------|:------------|
| [#6](https://github.com/bbengt1/yolo-developer/issues/6) | **Plugin System** | Create and integrate custom agents into the workflow |
| [#11](https://github.com/bbengt1/yolo-developer/issues/11) | **Course Correction** | Mid-sprint requirement changes with impact analysis |
| [#13](https://github.com/bbengt1/yolo-developer/issues/13) | **Issue Import** | Convert GitHub issues to user stories for development |

#### Performance

| Issue | Feature | Description |
|:------|:--------|:------------|
| [#4](https://github.com/bbengt1/yolo-developer/issues/4) | **Token Efficiencies** | Context optimization and deduplication for reduced costs |
| [#5](https://github.com/bbengt1/yolo-developer/issues/5) | **Large Codebase Support** | Performance optimization for 10,000+ file repositories |
| [#15](https://github.com/bbengt1/yolo-developer/issues/15) | **Token Limit Scheduler** | Automatic rate limit handling with pause/resume for long sprints |

[View all issues on GitHub](https://github.com/bbengt1/yolo-developer/issues){: .btn .btn-outline .fs-5 }

---

## Getting Help

- **GitHub Issues**: [Report bugs or request features](https://github.com/bbengt1/yolo-developer/issues)
- **Discussions**: [Ask questions and share ideas](https://github.com/bbengt1/yolo-developer/discussions)

---

Built with the [BMad Method](https://github.com/bmadcode/BMAD-METHOD) for AI-assisted development.
