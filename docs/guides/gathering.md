---
layout: default
title: Interactive Gathering
nav_order: 11
parent: null
---

# Interactive Requirements Gathering
{: .no_toc }

Run guided Q&A sessions with the Analyst agent to create sprint-ready requirements.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Start a Session

```bash
yolo gather start task-manager --description "A simple task tracking app"
```

Answer each question. Type `quit` to save and exit, then resume later with:

```bash
yolo gather start task-manager --resume SESSION_ID
```

---

## List Sessions

```bash
yolo gather list
```

---

## Export Requirements

```bash
yolo gather export SESSION_ID --format markdown --output requirements.md
```

Then seed the requirements:

```bash
yolo seed requirements.md
```

---

## Example Session Transcript

```
$ yolo gather start task-manager --description "A simple task tracking app"

Session ID: 20250122093000
Project: task-manager
Phase: discovery

Question (discovery): What would you like to build? Please describe your idea briefly.
Your response: A task tracker where users can add tasks, organize them, and mark them done.

Extracted requirements:
  - Users can add tasks
  - Users can organize tasks
  - Users can mark tasks as done

Question (use_cases): Describe the primary workflow a user should complete.
Your response: Create a project, add tasks, then complete them as work is finished.

Extracted requirements:
  - Users can create projects
  - Users can complete tasks

Question (requirements): List the main features or actions users need.
Your response: Task priorities, due dates, and reminders.

Extracted requirements:
  - Tasks can have priorities
  - Tasks can have due dates
  - Users receive reminders

Question (constraints): Are there any required technologies, integrations, or constraints?
Your response: Must be web-based and use PostgreSQL.

Extracted requirements:
  - Must be web-based
  - Must use PostgreSQL

Question (edge_cases): What edge cases or failure scenarios should be handled?
Your response: Handle duplicate tasks and missing due dates.

Extracted requirements:
  - Prevent duplicate tasks
  - Support tasks without due dates

Question (validation): Is anything missing or unclear in the requirements so far?
Your response: Include basic authentication.

Extracted requirements:
  - Users can authenticate

Question (refinement): Confirm: Are these requirements complete and accurate?
Your response: yes

Session complete. Export with `yolo gather export`.
```
