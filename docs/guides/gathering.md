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
