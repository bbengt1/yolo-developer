"""Documentation generation prompt templates for Dev agent (Story 8.5).

This module provides the core prompt templates and builder functions for
LLM-powered documentation generation. Templates emphasize Google-style
docstrings, comprehensive module documentation, and explanatory comments.

Key Concepts:
    - **Google-style Docstrings:** Standard format with Args, Returns, Raises, Example
    - **Module Documentation:** One-line summary, purpose, key concepts, examples
    - **Complex Logic Comments:** Explain "why" not just "what"
    - **Documentation Enhancement:** LLM-powered improvement of existing docs

Example:
    >>> from yolo_developer.agents.dev.prompts.documentation_generation import (
    ...     build_documentation_prompt,
    ... )
    >>>
    >>> prompt = build_documentation_prompt(
    ...     code_content="def process(x): return x * 2",
    ...     documentation_analysis="Missing module docstring",
    ...     complex_sections="None detected",
    ... )
    >>> "documentation" in prompt.lower()
    True
"""

from __future__ import annotations

# =============================================================================
# Documentation Guidelines (AC1, AC3, AC4)
# =============================================================================

DOCUMENTATION_GUIDELINES = """
## Documentation Guidelines

### Google-style Docstring Format (AC1, AC4)
Follow Google Python Style Guide for all docstrings:

**Function Docstrings:**
- One-line summary ending with a period
- Extended description if behavior is non-obvious
- Args: List each parameter with type and description
- Returns: Describe return value and type
- Raises: List exceptions that may be raised
- Example: Include runnable doctest code

**Example Function Docstring:**
```python
def calculate_total(items: list[Item], tax_rate: float = 0.1) -> float:
    \"\"\"Calculate total price including tax.

    Computes the sum of all item prices and applies the specified
    tax rate to determine the final total.

    Args:
        items: List of items to calculate total for.
        tax_rate: Tax rate to apply. Defaults to 0.1 (10%).

    Returns:
        Total price including tax as a float.

    Raises:
        ValueError: If tax_rate is negative.

    Example:
        >>> items = [Item(price=10.0), Item(price=20.0)]
        >>> calculate_total(items, tax_rate=0.1)
        33.0
    \"\"\"
```

### Module Docstring Format (AC3)
- First line: One-sentence summary of module purpose
- Extended paragraph explaining functionality and use cases
- Key Concepts/Functions section listing main exports with brief descriptions
- Example section with runnable code

**Example Module Docstring:**
```python
\"\"\"User authentication and session management.

This module provides functions for authenticating users, managing
sessions, and validating access tokens. It integrates with the
identity provider and maintains session state.

Key Functions:
    - authenticate_user: Validate credentials and create session
    - validate_token: Check if access token is valid
    - refresh_session: Extend session lifetime

Example:
    >>> from auth import authenticate_user
    >>> session = authenticate_user("user@example.com", "password")
    >>> session.is_valid
    True
\"\"\"
```

### Comment Guidelines (AC2)
- Add comments above complex code sections
- Explain the "why" not just the "what"
- Reference acceptance criteria or requirements where applicable
- Keep comments concise but informative
- Do not add comments for self-explanatory code
"""

# =============================================================================
# Module Docstring Template (AC3)
# =============================================================================

MODULE_DOCSTRING_TEMPLATE = """
## Module Docstring Requirements

A comprehensive module docstring should include:

1. **One-line Summary** (required)
   - First line describes module purpose in one sentence
   - Ends with a period

2. **Extended Description** (required for non-trivial modules)
   - Paragraph explaining what the module does
   - Use cases and when to use this module
   - How it fits into the larger system

3. **Key Concepts/Functions Section** (required)
   - List main exports with brief descriptions
   - Use bullet points with function/class names
   - Include type information where helpful

4. **Example Section** (required)
   - Runnable doctest code showing typical usage
   - Import statements included
   - Expected output shown

**Pattern:**
```python
\"\"\"One-line summary describing module purpose.

Extended description explaining what the module does, its purpose,
and how it should be used. Include context about where this fits
in the larger system.

Key Functions:
    - function_one: Brief description of what it does
    - function_two: Brief description of what it does
    - ClassName: Brief description of the class

Example:
    >>> from module import function_one
    >>> result = function_one(arg)
    >>> result
    expected_output
\"\"\"
```
"""

# =============================================================================
# Function Docstring Template (AC1)
# =============================================================================

FUNCTION_DOCSTRING_TEMPLATE = """
## Function Docstring Requirements (Google Style)

Every public function should have a docstring with:

1. **One-line Summary** (required)
   - First line describes what the function does
   - Written in imperative mood ("Calculate", "Return", "Check")
   - Ends with a period

2. **Extended Description** (optional, for complex functions)
   - Additional context about behavior
   - Algorithm explanation if non-obvious
   - Side effects or important notes

3. **Args Section** (required if function has parameters)
   - List each parameter on its own line
   - Format: `param_name: Description of the parameter.`
   - Include type info if not obvious from annotation
   - Note default values: "Defaults to X."

4. **Returns Section** (required if function returns a value)
   - Describe what is returned
   - Include type information
   - For tuples, describe each element

5. **Raises Section** (required if function raises exceptions)
   - List each exception type and when it's raised
   - Format: `ExceptionType: When this exception is raised.`

6. **Example Section** (recommended)
   - Doctest-style code with expected output
   - Show typical usage patterns
   - Include edge cases if space permits

**Pattern:**
```python
def function_name(param1: str, param2: int = 0) -> bool:
    \"\"\"One-line summary of what the function does.

    Extended description providing more context about the function's
    behavior, algorithm, or any important notes.

    Args:
        param1: Description of first parameter.
        param2: Description of second parameter. Defaults to 0.

    Returns:
        Description of return value. True if condition, False otherwise.

    Raises:
        ValueError: If param1 is empty.
        TypeError: If param2 is not an integer.

    Example:
        >>> function_name("test", 42)
        True
    \"\"\"
```
"""

# =============================================================================
# Documentation Generation Template (AC5)
# =============================================================================

DOCUMENTATION_GENERATION_TEMPLATE = """You are a senior Python developer adding comprehensive documentation.

# Source Code to Document

```python
{code_content}
```

# Documentation Analysis

{documentation_analysis}

# Complex Sections Needing Comments

{complex_sections}

{documentation_guidelines}

{module_template}

{function_template}

# Additional Context

{additional_context}

# Instructions

Enhance the code with comprehensive documentation following these steps:

1. **Module Docstring**: Add or improve the module docstring at the top
   - Include one-line summary, extended description, key concepts, example

2. **Function Docstrings**: Add or improve docstrings for all public functions
   - Include Args, Returns, Raises (if applicable), and Example sections
   - Ensure type information matches function annotations

3. **Complex Logic Comments**: Add explanatory comments for identified complex sections
   - Place comments above the complex code
   - Explain the "why" behind the logic
   - Keep comments concise

4. **Type Consistency**: Ensure docstring type descriptions match annotations

5. **Terminology**: Use terminology consistent with the codebase

Output Requirements:
- Output the fully documented Python code
- Do not include explanations outside the code
- Wrap the code in ```python and ``` markers
- Preserve all existing functionality - only add/improve documentation
"""


def build_documentation_prompt(
    code_content: str,
    documentation_analysis: str,
    complex_sections: str,
    additional_context: str = "",
    include_guidelines: bool = True,
    include_module_template: bool = True,
    include_function_template: bool = True,
) -> str:
    """Build a complete documentation generation prompt for the LLM.

    Constructs a prompt that includes the source code, documentation
    analysis results, complex sections needing comments, and templates
    for proper documentation format.

    Args:
        code_content: The Python code to add documentation to.
        documentation_analysis: Analysis of current documentation status.
        complex_sections: Description of complex code sections needing comments.
        additional_context: Any additional context for documentation.
            Defaults to empty string.
        include_guidelines: Whether to include documentation guidelines.
            Defaults to True.
        include_module_template: Whether to include module docstring template.
            Defaults to True.
        include_function_template: Whether to include function docstring template.
            Defaults to True.

    Returns:
        Formatted prompt string ready for LLM.

    Example:
        >>> prompt = build_documentation_prompt(
        ...     code_content="def hello(): pass",
        ...     documentation_analysis="Missing docstring",
        ...     complex_sections="None",
        ... )
        >>> "hello" in prompt
        True
    """
    # Include or exclude sections
    guidelines_text = DOCUMENTATION_GUIDELINES if include_guidelines else ""
    module_text = MODULE_DOCSTRING_TEMPLATE if include_module_template else ""
    function_text = FUNCTION_DOCSTRING_TEMPLATE if include_function_template else ""

    # Format additional context
    if additional_context:
        context_text = f"## Additional Context\n\n{additional_context}"
    else:
        context_text = "(No additional context provided)"

    # Build the complete prompt
    prompt = DOCUMENTATION_GENERATION_TEMPLATE.format(
        code_content=code_content,
        documentation_analysis=documentation_analysis,
        complex_sections=complex_sections,
        documentation_guidelines=guidelines_text,
        module_template=module_text,
        function_template=function_text,
        additional_context=context_text,
    )

    return prompt


def build_documentation_retry_prompt(
    original_prompt: str,
    error_message: str,
    previous_code: str,
) -> str:
    """Build a retry prompt when documentation generation fails validation.

    Args:
        original_prompt: The original documentation generation prompt.
        error_message: The error message from validation.
        previous_code: The code that failed validation.

    Returns:
        Modified prompt for retry with error context.

    Example:
        >>> retry_prompt = build_documentation_retry_prompt(
        ...     original_prompt="Generate documentation...",
        ...     error_message="Line 5: unexpected indent",
        ...     previous_code="def broken(",
        ... )
        >>> "syntax" in retry_prompt.lower()
        True
    """
    retry_template = """{original_prompt}

## Previous Attempt Failed

The previous documentation generation attempt failed with a syntax error:

**Error:** {error_message}

**Previous Code (excerpt):**
```python
{previous_code_excerpt}
```

Please fix the syntax error and generate valid Python code with documentation.
Ensure all brackets, parentheses, quotes, and indentation are correct.
Pay special attention to:
- Proper string quoting in docstrings (use triple quotes)
- Correct indentation for docstrings
- Balanced parentheses and brackets
- Valid Python syntax throughout
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
