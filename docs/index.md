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

### Full Observability
Complete audit trail with decision logging, token cost tracking, and requirement traceability from seed to implementation.

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

</div>

---

## System Requirements

| Requirement | Minimum | Recommended |
|:------------|:--------|:------------|
| Python | 3.10+ | 3.12+ |
| Memory | 4 GB | 8 GB |
| Disk | 500 MB | 2 GB (with memory persistence) |
| OS | macOS, Linux, Windows (WSL2) | macOS, Linux |

---

## Roadmap

### Current Status

| Epic | Status | Description |
|:-----|:-------|:------------|
| 1-13 | ✅ Complete | Core infrastructure, all agents, CLI, SDK |
| 14 | 🚧 In Progress | MCP Integration |

### Planned Features

#### LLM Providers

| Issue | Feature | Description |
|:------|:--------|:------------|
| [#1](https://github.com/bbengt1/yolo-developer/issues/1) | **Local LLM Support** | Ollama, LM Studio, vLLM integration with hybrid routing |
| [#8](https://github.com/bbengt1/yolo-developer/issues/8) | **ChatGPT Codex Support** | OpenAI models as LLM provider with Azure support |

#### IDE Integrations

| Issue | Feature | Description |
|:------|:--------|:------------|
| [#9](https://github.com/bbengt1/yolo-developer/issues/9) | **Cursor IDE Support** | VS Code extension with MCP integration for Cursor |
| [#10](https://github.com/bbengt1/yolo-developer/issues/10) | **GitHub Copilot Support** | `@yolo` chat participant and Copilot Workspace integration |

#### User Interfaces

| Issue | Feature | Description |
|:------|:--------|:------------|
| [#3](https://github.com/bbengt1/yolo-developer/issues/3) | **Web Interface** | Full web UI with REST API, WebSocket updates, document upload |
| [#7](https://github.com/bbengt1/yolo-developer/issues/7) | **Sprint Dashboard** | Real-time visualization of sprint progress and agent activity |

#### Core Enhancements

| Issue | Feature | Description |
|:------|:--------|:------------|
| [#2](https://github.com/bbengt1/yolo-developer/issues/2) | **Brownfield Support** | Add YOLO to existing projects with deep codebase scanning |
| [#6](https://github.com/bbengt1/yolo-developer/issues/6) | **Plugin System** | Create and integrate custom agents into the workflow |
| [#11](https://github.com/bbengt1/yolo-developer/issues/11) | **Course Correction** | Mid-sprint requirement changes with impact analysis |
| [#12](https://github.com/bbengt1/yolo-developer/issues/12) | **GitHub Management** | Full Git/GitHub workflow: commits, PRs, issues, releases |
| [#13](https://github.com/bbengt1/yolo-developer/issues/13) | **Issue Import** | Convert GitHub issues to user stories for development |
| [#14](https://github.com/bbengt1/yolo-developer/issues/14) | **Requirements Gathering** | Interactive session to elicit and crystallize requirements |

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
