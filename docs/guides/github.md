---
layout: default
title: GitHub Automation
nav_order: 9
parent: null
---

# GitHub Automation
{: .no_toc }

Use YOLO Developer to manage branches, commits, pull requests, issues, and releases.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Prerequisites

- Install the GitHub CLI (`gh`) and authenticate (`gh auth login`).
- Ensure `yolo.yaml` includes the `github` configuration or set `YOLO_GITHUB__TOKEN`.

## Git Operations

```bash
yolo git status
yolo git commit -m "feat: update"
yolo git push
```

## Pull Requests

```bash
yolo pr create --title "Add feature" --body "Implements XYZ"
yolo pr merge 123 --method squash
```

## Issues

```bash
yolo issue create --title "Bug" --body "Steps to reproduce..."
yolo issue close 456 --comment "Fixed in PR #123"
```

## Releases

```bash
yolo release create --tag v1.2.0 --name "Release 1.2.0" --body "Notes..."
```

## Workflow Shortcuts

```bash
yolo workflow start US-001 --title "Add endpoint" --description "..."
yolo workflow complete US-001 --title "Add endpoint" --description "..." --commit "feat: add endpoint"
```
