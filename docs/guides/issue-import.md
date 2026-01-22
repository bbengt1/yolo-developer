---
layout: default
title: Issue Import
nav_order: 10
parent: null
---

# Issue Import
{: .no_toc }

Import GitHub issues and convert them into user stories ready for sprint planning.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Prerequisites

- Install the GitHub CLI (`gh`) and authenticate (`gh auth login`).
- Configure `YOLO_GITHUB__TOKEN` or log in with `gh`.

---

## Import a Single Issue

```bash
yolo import issue 42
```

This generates a story with acceptance criteria, then posts a summary comment
back to the issue (unless `--preview` is used).

---

## Preview Without Updating the Issue

```bash
yolo import preview 42
```

---

## Import Multiple Issues

```bash
yolo import issues 12 34 56
```

Filter by label:

```bash
yolo import issues --label "ready" --auto-seed
```

Search query:

```bash
yolo import issues --query "is:open label:feature"
```

---

## Seed File Output

Use `--auto-seed` to write a seed file for later use:

```bash
yolo import issue 42 --auto-seed
```

Seed files are stored in `.yolo/imported-issues/`.

---

## Configuration

Customize import behavior in `yolo.yaml`:

```yaml
github:
  import_config:
    enabled: true
    update_issues: true
    add_label: yolo-imported
    story:
      id_prefix: US
      estimate_points: true
```
