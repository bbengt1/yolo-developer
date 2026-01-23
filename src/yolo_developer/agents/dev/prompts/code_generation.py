"""Code generation prompt templates for Dev agent (Story 8.2).

This module provides the core prompt template and builder function for
LLM-powered code generation. The template emphasizes maintainability,
following the maintainability-first hierarchy from the architecture.

Key Concepts:
    - **Maintainability-First:** Readability > Simplicity > Maintainability > Performance
    - **YAGNI:** Don't add features not in requirements
    - **Clean Code:** Small functions, descriptive names, low complexity

Example:
    >>> from yolo_developer.agents.dev.prompts.code_generation import (
    ...     build_code_generation_prompt,
    ... )
    >>>
    >>> prompt = build_code_generation_prompt(
    ...     story_title="User Authentication",
    ...     requirements="Implement login functionality",
    ...     acceptance_criteria=["Users can login with email/password"],
    ... )
    >>> "User Authentication" in prompt
    True
"""

from __future__ import annotations

from typing import Any

# =============================================================================
# Maintainability Guidelines (AC1, AC2, AC3, AC4)
# =============================================================================

MAINTAINABILITY_GUIDELINES = """
## Maintainability Guidelines

Follow the maintainability-first hierarchy:
1. **Readability** - Code should be immediately understandable
2. **Simplicity** - Avoid unnecessary complexity
3. **Maintainability** - Easy to modify and extend
4. **Performance** - Only optimize when measurably needed

### Function Size and Complexity
- Keep functions under 50 lines
- Maximum nesting depth: 3 levels
- Cyclomatic complexity < 10 per function
- Single responsibility per function
- Break complex logic into helper functions

### Naming Conventions
- Clear, descriptive variable names that explain purpose
- No single-letter variables except standard iterators (i, j, k in loops)
- Names should be self-documenting
- Avoid abbreviations unless universally understood

### Code Structure
- Prefer explicit over implicit
- Prefer simple over clever
- Avoid premature optimization
- Follow YAGNI - don't add features not in requirements
"""

# =============================================================================
# Project Conventions (per architecture.md)
# =============================================================================

PROJECT_CONVENTIONS = """
## Project Conventions

### Python Style
- Use snake_case for functions and variables
- Use PascalCase for classes
- Use UPPER_SNAKE_CASE for constants
- Use leading underscore for private functions (_helper)

### Type Annotations
- Full type annotations required on all functions
- Use `from __future__ import annotations` at file top
- Type hint return values, even None -> None
- Use generic types from typing module

### Async Patterns
- Use async/await for all I/O operations
- Use async def for functions that call async code
- Never block the event loop with sync I/O

### Documentation
- Include module-level docstring
- Include docstring for all public functions
- Use Google-style docstrings with Args, Returns, Example

### Error Handling
- Raise specific exceptions, not generic Exception
- Include context in error messages
- Log errors with structlog
"""

# =============================================================================
# Code Generation Template
# =============================================================================

CODE_GENERATION_TEMPLATE = """You are a senior Python developer implementing code \
for the YOLO Developer project.

# Project Structure

{project_structure}

# Story Information

**Story Title:** {story_title}

**Requirements:**
{requirements}

**Acceptance Criteria:**
{acceptance_criteria}

# Design Context

{design_decisions}

{maintainability_guidelines}

{project_conventions}

# Additional Context

{additional_context}

# Instructions

Generate clean, maintainable Python code that implements the requirements above.

**IMPORTANT:** You are implementing code for the actual yolo-developer project, \
NOT a standalone module.
- Place the code in the appropriate location within `src/yolo_developer/` based on the feature
- For memory-related features, use `src/yolo_developer/memory/`
- For agent-related features, use `src/yolo_developer/agents/`
- For orchestration features, use `src/yolo_developer/orchestrator/`
- For configuration features, use `src/yolo_developer/config/`
- Create new modules/files as appropriate for the feature being implemented

Requirements for the generated code:
1. Include `from __future__ import annotations` at the top
2. Include complete type annotations on all functions
3. Include docstrings for all public functions (Google style)
4. Follow the naming conventions (snake_case functions, PascalCase classes)
5. Keep functions small and focused (< 50 lines each)
6. Avoid deep nesting (max 3 levels)
7. Use descriptive variable names
8. Implement ONLY what is required - no extra features

**Output Format:**
First, specify the file path on its own line starting with `FILE_PATH:`, then the code.
Example:
FILE_PATH: src/yolo_developer/memory/context_manager.py
```python
# code here
```

You may specify multiple files if needed:
FILE_PATH: src/yolo_developer/memory/context_manager.py
```python
# first file code
```

FILE_PATH: src/yolo_developer/memory/token_tracker.py
```python
# second file code
```
"""


def build_code_generation_prompt(
    story_title: str,
    requirements: str,
    acceptance_criteria: list[str] | None = None,
    design_decisions: dict[str, Any] | None = None,
    additional_context: str = "",
    include_maintainability: bool = True,
    include_conventions: bool = True,
    project_structure: str = "",
) -> str:
    """Build a complete code generation prompt for the LLM.

    Constructs a prompt that includes story requirements, acceptance criteria,
    design decisions, maintainability guidelines, and project conventions.

    Args:
        story_title: Title of the story being implemented.
        requirements: Detailed requirements for the implementation.
        acceptance_criteria: List of acceptance criteria strings. Defaults to empty.
        design_decisions: Dictionary of design decisions and patterns. Defaults to empty.
        additional_context: Any additional context for code generation. Defaults to empty.
        include_maintainability: Whether to include maintainability guidelines.
            Defaults to True.
        include_conventions: Whether to include project conventions. Defaults to True.
        project_structure: Description of project structure for file placement.
            Defaults to empty.

    Returns:
        Formatted prompt string ready for LLM.

    Example:
        >>> prompt = build_code_generation_prompt(
        ...     story_title="User Login",
        ...     requirements="Implement user login with email and password",
        ...     acceptance_criteria=["Users can login", "Invalid credentials rejected"],
        ...     design_decisions={"pattern": "Repository", "auth_method": "JWT"},
        ... )
        >>> "User Login" in prompt
        True
        >>> "Implement user login" in prompt
        True
    """
    # Format acceptance criteria
    ac_text = ""
    if acceptance_criteria:
        ac_lines = [f"- {ac}" for ac in acceptance_criteria]
        ac_text = "\n".join(ac_lines)
    else:
        ac_text = "(No specific acceptance criteria provided)"

    # Format design decisions
    design_text = ""
    if design_decisions:
        design_lines = ["## Design Decisions", ""]
        for key, value in design_decisions.items():
            design_lines.append(f"- **{key}:** {value}")
        design_text = "\n".join(design_lines)
    else:
        design_text = "## Design Decisions\n\n(No specific design decisions provided)"

    # Include or exclude guidelines
    maintainability_text = MAINTAINABILITY_GUIDELINES if include_maintainability else ""
    conventions_text = PROJECT_CONVENTIONS if include_conventions else ""

    # Format additional context
    context_text = ""
    if additional_context:
        context_text = f"## Additional Context\n\n{additional_context}"
    else:
        context_text = "(No additional context provided)"

    # Format project structure
    structure_text = ""
    if project_structure:
        structure_text = project_structure
    else:
        structure_text = """This is the yolo-developer project with structure:
```
src/yolo_developer/
├── agents/         # Individual agent modules (analyst, pm, architect, dev, sm, tea)
├── orchestrator/   # LangGraph graph definition, state schema, node functions
├── memory/         # Vector store (ChromaDB) and graph store integration
├── gates/          # Quality gate framework with blocking mechanism
├── audit/          # Decision logging and traceability
├── config/         # Pydantic configuration with YAML + env var support
├── cli/            # Typer CLI interface
├── sdk/            # Python SDK for programmatic access
└── mcp/            # FastMCP server for external integration
```"""

    # Build the complete prompt
    prompt = CODE_GENERATION_TEMPLATE.format(
        story_title=story_title,
        requirements=requirements,
        acceptance_criteria=ac_text,
        design_decisions=design_text,
        maintainability_guidelines=maintainability_text,
        project_conventions=conventions_text,
        additional_context=context_text,
        project_structure=structure_text,
    )

    return prompt


def build_retry_prompt(
    original_prompt: str,
    error_message: str,
    previous_code: str,
) -> str:
    """Build a retry prompt when code generation fails syntax validation.

    Args:
        original_prompt: The original code generation prompt.
        error_message: The syntax error message from validation.
        previous_code: The code that failed validation.

    Returns:
        Modified prompt for retry with error context.

    Example:
        >>> retry_prompt = build_retry_prompt(
        ...     original_prompt="Generate user login...",
        ...     error_message="Line 5: unexpected indent",
        ...     previous_code="def login():\\n  pass",
        ... )
        >>> "syntax error" in retry_prompt.lower()
        True
    """
    retry_template = """{original_prompt}

## Previous Attempt Failed

The previous code generation attempt failed with a syntax error:

**Error:** {error_message}

**Previous Code (excerpt):**
```python
{previous_code_excerpt}
```

Please fix the syntax error and generate valid Python code.
Ensure all brackets, parentheses, and indentation are correct.
"""

    # Limit previous code excerpt to first 500 chars
    code_excerpt = previous_code[:500]
    if len(previous_code) > 500:
        code_excerpt += "\n... (truncated)"

    return retry_template.format(
        original_prompt=original_prompt,
        error_message=error_message,
        previous_code_excerpt=code_excerpt,
    )
