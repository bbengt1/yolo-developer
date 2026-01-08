"""Unit tests for question generation enhancements (Story 4.4).

Tests cover:
- AnswerFormat enum values and serialization
- Enhanced ResolutionPrompt with format fields
- Question quality validation
- Question prioritization
- Backward compatibility
"""

from __future__ import annotations

import pytest

from yolo_developer.seed.ambiguity import (
    Ambiguity,
    AmbiguitySeverity,
    AmbiguityType,
    AnswerFormat,
    ResolutionPrompt,
    calculate_question_priority,
    prioritize_questions,
    validate_question_quality,
)


class TestAnswerFormat:
    """Test AnswerFormat enum."""

    def test_answer_format_values(self) -> None:
        """Test that all expected AnswerFormat values exist."""
        assert AnswerFormat.BOOLEAN.value == "boolean"
        assert AnswerFormat.NUMERIC.value == "numeric"
        assert AnswerFormat.CHOICE.value == "choice"
        assert AnswerFormat.FREE_TEXT.value == "free_text"
        assert AnswerFormat.DATE.value == "date"
        assert AnswerFormat.LIST.value == "list"

    def test_answer_format_is_string_enum(self) -> None:
        """Test that AnswerFormat is a string enum for JSON serialization."""
        assert isinstance(AnswerFormat.BOOLEAN.value, str)
        assert str(AnswerFormat.BOOLEAN) == "AnswerFormat.BOOLEAN"
        # Should work with JSON serialization via .value
        import json
        serialized = json.dumps({"format": AnswerFormat.BOOLEAN.value})
        assert '"boolean"' in serialized


class TestEnhancedResolutionPrompt:
    """Test enhanced ResolutionPrompt with answer format fields."""

    def test_resolution_prompt_with_answer_format(self) -> None:
        """Test ResolutionPrompt with explicit answer_format."""
        prompt = ResolutionPrompt(
            question="How many users should the system support?",
            suggestions=("100", "1000", "10000"),
            default="1000",
            answer_format=AnswerFormat.NUMERIC,
            format_hint="Enter a number (e.g., 100, 1000)",
        )
        assert prompt.answer_format == AnswerFormat.NUMERIC
        assert prompt.format_hint == "Enter a number (e.g., 100, 1000)"
        assert prompt.validation_pattern is None

    def test_resolution_prompt_with_validation_pattern(self) -> None:
        """Test ResolutionPrompt with validation pattern."""
        prompt = ResolutionPrompt(
            question="When is the deadline?",
            suggestions=("2026-06-01", "2026-12-31"),
            answer_format=AnswerFormat.DATE,
            format_hint="Enter a date (YYYY-MM-DD)",
            validation_pattern=r"^\d{4}-\d{2}-\d{2}$",
        )
        assert prompt.answer_format == AnswerFormat.DATE
        assert prompt.validation_pattern == r"^\d{4}-\d{2}-\d{2}$"

    def test_resolution_prompt_backward_compatibility(self) -> None:
        """Test that existing code without new fields still works."""
        # Old-style creation without new fields
        prompt = ResolutionPrompt(
            question="What do you mean?",
            suggestions=("Option A", "Option B"),
        )
        # Should have defaults for new fields
        assert prompt.answer_format == AnswerFormat.FREE_TEXT
        assert prompt.format_hint is None
        assert prompt.validation_pattern is None

    def test_resolution_prompt_to_dict_includes_new_fields(self) -> None:
        """Test that to_dict() includes the new fields."""
        prompt = ResolutionPrompt(
            question="Is this feature required?",
            suggestions=("Yes", "No"),
            answer_format=AnswerFormat.BOOLEAN,
            format_hint="Answer yes or no",
        )
        result = prompt.to_dict()
        assert result["question"] == "Is this feature required?"
        assert result["suggestions"] == ["Yes", "No"]
        assert result["answer_format"] == "boolean"
        assert result["format_hint"] == "Answer yes or no"
        assert result["validation_pattern"] is None

    def test_resolution_prompt_to_dict_backward_compatible(self) -> None:
        """Test to_dict() still works for prompts without new fields."""
        prompt = ResolutionPrompt(
            question="Test question?",
            suggestions=("A", "B"),
            default="A",
        )
        result = prompt.to_dict()
        # Must include all fields including defaults
        assert "question" in result
        assert "suggestions" in result
        assert "default" in result
        assert "answer_format" in result
        assert "format_hint" in result
        assert "validation_pattern" in result

    def test_resolution_prompt_frozen(self) -> None:
        """Test that ResolutionPrompt is still frozen (immutable)."""
        prompt = ResolutionPrompt(
            question="Test?",
            suggestions=(),
            answer_format=AnswerFormat.CHOICE,
        )
        with pytest.raises(AttributeError):
            prompt.question = "Modified"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            prompt.answer_format = AnswerFormat.BOOLEAN  # type: ignore[misc]

    def test_all_answer_formats_have_sensible_defaults(self) -> None:
        """Test creating prompts with each AnswerFormat value."""
        formats = [
            AnswerFormat.BOOLEAN,
            AnswerFormat.NUMERIC,
            AnswerFormat.CHOICE,
            AnswerFormat.FREE_TEXT,
            AnswerFormat.DATE,
            AnswerFormat.LIST,
        ]
        for fmt in formats:
            prompt = ResolutionPrompt(
                question=f"Test question for {fmt.value}?",
                suggestions=(),
                answer_format=fmt,
            )
            assert prompt.answer_format == fmt


class TestQuestionQualityValidation:
    """Test validate_question_quality function."""

    def test_valid_question_passes(self) -> None:
        """Test that a well-formed question passes validation."""
        is_valid, suggestions = validate_question_quality(
            "How many concurrent users should the system support?"
        )
        assert is_valid is True
        assert suggestions == []

    def test_question_with_please_clarify_fails(self) -> None:
        """Test that vague 'please clarify' phrase is flagged."""
        is_valid, suggestions = validate_question_quality(
            "Please clarify what you mean by fast."
        )
        assert is_valid is False
        assert any("vague phrase" in s.lower() for s in suggestions)

    def test_question_with_more_information_fails(self) -> None:
        """Test that 'more information' phrase is flagged."""
        is_valid, suggestions = validate_question_quality(
            "Can you provide more information about the requirements?"
        )
        assert is_valid is False
        assert any("vague phrase" in s.lower() for s in suggestions)

    def test_question_with_elaborate_fails(self) -> None:
        """Test that 'elaborate' phrase is flagged."""
        is_valid, suggestions = validate_question_quality(
            "Could you elaborate on the scalability requirements?"
        )
        assert is_valid is False
        assert any("vague phrase" in s.lower() for s in suggestions)

    def test_question_too_short_fails(self) -> None:
        """Test that very short questions are flagged."""
        is_valid, suggestions = validate_question_quality("What?")
        assert is_valid is False
        assert any("too short" in s.lower() for s in suggestions)

    def test_empty_question_fails(self) -> None:
        """Test that empty questions are flagged."""
        is_valid, suggestions = validate_question_quality("")
        assert is_valid is False
        assert len(suggestions) > 0

    def test_whitespace_only_question_fails(self) -> None:
        """Test that whitespace-only questions are flagged."""
        is_valid, _suggestions = validate_question_quality("   ")
        assert is_valid is False

    def test_minimum_length_boundary(self) -> None:
        """Test the minimum length boundary (>10 chars)."""
        # Exactly 10 chars should fail
        is_valid_10, _ = validate_question_quality("1234567890")
        assert is_valid_10 is False

        # 11 chars should pass length check (may still fail other checks)
        _is_valid_11, suggestions = validate_question_quality("12345678901")
        # Length check should pass
        assert not any("too short" in s.lower() for s in suggestions)

    def test_multiple_vague_phrases_lists_all(self) -> None:
        """Test that multiple vague phrases are all identified."""
        is_valid, suggestions = validate_question_quality(
            "Please clarify and provide more information about what you need."
        )
        assert is_valid is False
        # Should have at least one suggestion about vague phrases
        assert len(suggestions) >= 1

    def test_case_insensitive_vague_phrase_detection(self) -> None:
        """Test that vague phrase detection is case insensitive."""
        is_valid, _ = validate_question_quality(
            "PLEASE CLARIFY what you mean."
        )
        assert is_valid is False

    def test_actionable_question_with_specifics_passes(self) -> None:
        """Test that specific, actionable questions pass."""
        questions = [
            "What is the maximum response time in milliseconds?",
            "Should authentication use OAuth2 or JWT?",
            "How many items per page should the pagination return?",
            "Is two-factor authentication required for admin users?",
        ]
        for question in questions:
            is_valid, suggestions = validate_question_quality(question)
            assert is_valid is True, f"Question should pass: {question}"
            assert suggestions == []


class TestQuestionPrioritization:
    """Test question prioritization functions."""

    def _create_ambiguity(
        self,
        amb_type: AmbiguityType,
        severity: AmbiguitySeverity,
        source_text: str = "test",
    ) -> Ambiguity:
        """Helper to create test ambiguities."""
        return Ambiguity(
            ambiguity_type=amb_type,
            severity=severity,
            source_text=source_text,
            location="line 1",
            description="Test description",
        )

    def test_calculate_priority_high_undefined(self) -> None:
        """Test priority for HIGH severity + UNDEFINED type (highest)."""
        amb = self._create_ambiguity(AmbiguityType.UNDEFINED, AmbiguitySeverity.HIGH)
        priority = calculate_question_priority(amb)
        # HIGH=30 + UNDEFINED=25 = 55
        assert priority == 55

    def test_calculate_priority_high_scope(self) -> None:
        """Test priority for HIGH severity + SCOPE type."""
        amb = self._create_ambiguity(AmbiguityType.SCOPE, AmbiguitySeverity.HIGH)
        priority = calculate_question_priority(amb)
        # HIGH=30 + SCOPE=20 = 50
        assert priority == 50

    def test_calculate_priority_medium_technical(self) -> None:
        """Test priority for MEDIUM severity + TECHNICAL type."""
        amb = self._create_ambiguity(AmbiguityType.TECHNICAL, AmbiguitySeverity.MEDIUM)
        priority = calculate_question_priority(amb)
        # MEDIUM=20 + TECHNICAL=15 = 35
        assert priority == 35

    def test_calculate_priority_low_priority_type(self) -> None:
        """Test priority for LOW severity + PRIORITY type (lowest)."""
        amb = self._create_ambiguity(AmbiguityType.PRIORITY, AmbiguitySeverity.LOW)
        priority = calculate_question_priority(amb)
        # LOW=10 + PRIORITY=5 = 15
        assert priority == 15

    def test_severity_weights_are_correct(self) -> None:
        """Test that severity weights match spec: HIGH=30, MEDIUM=20, LOW=10."""
        high = self._create_ambiguity(AmbiguityType.SCOPE, AmbiguitySeverity.HIGH)
        med = self._create_ambiguity(AmbiguityType.SCOPE, AmbiguitySeverity.MEDIUM)
        low = self._create_ambiguity(AmbiguityType.SCOPE, AmbiguitySeverity.LOW)

        # Same type, different severity - difference should be 10
        assert calculate_question_priority(high) - calculate_question_priority(med) == 10
        assert calculate_question_priority(med) - calculate_question_priority(low) == 10

    def test_type_weights_ordering(self) -> None:
        """Test type weights: UNDEFINED > SCOPE > TECHNICAL > DEPENDENCY > PRIORITY."""
        sev = AmbiguitySeverity.MEDIUM  # Same severity for all
        undefined = self._create_ambiguity(AmbiguityType.UNDEFINED, sev)
        scope = self._create_ambiguity(AmbiguityType.SCOPE, sev)
        technical = self._create_ambiguity(AmbiguityType.TECHNICAL, sev)
        dependency = self._create_ambiguity(AmbiguityType.DEPENDENCY, sev)
        priority = self._create_ambiguity(AmbiguityType.PRIORITY, sev)

        scores = [
            calculate_question_priority(undefined),
            calculate_question_priority(scope),
            calculate_question_priority(technical),
            calculate_question_priority(dependency),
            calculate_question_priority(priority),
        ]
        # Scores should be strictly decreasing
        assert scores == sorted(scores, reverse=True)

    def test_prioritize_questions_ordering(self) -> None:
        """Test that prioritize_questions returns highest priority first."""
        amb_low = self._create_ambiguity(AmbiguityType.PRIORITY, AmbiguitySeverity.LOW, "low_item")
        amb_med = self._create_ambiguity(AmbiguityType.TECHNICAL, AmbiguitySeverity.MEDIUM, "med_item")
        amb_high = self._create_ambiguity(AmbiguityType.UNDEFINED, AmbiguitySeverity.HIGH, "high_item")

        # Pass in unsorted order
        unsorted = [amb_low, amb_med, amb_high]
        sorted_ambs = prioritize_questions(unsorted)

        # Should be sorted by priority (highest first)
        assert sorted_ambs[0].source_text == "high_item"
        assert sorted_ambs[1].source_text == "med_item"
        assert sorted_ambs[2].source_text == "low_item"

    def test_prioritize_questions_deterministic_tie_breaking(self) -> None:
        """Test deterministic tie-breaking by source_text."""
        amb_a = self._create_ambiguity(AmbiguityType.SCOPE, AmbiguitySeverity.HIGH, "apple")
        amb_b = self._create_ambiguity(AmbiguityType.SCOPE, AmbiguitySeverity.HIGH, "banana")
        amb_c = self._create_ambiguity(AmbiguityType.SCOPE, AmbiguitySeverity.HIGH, "cherry")

        # All have same priority, should be sorted by source_text alphabetically
        unsorted = [amb_c, amb_a, amb_b]
        sorted_ambs = prioritize_questions(unsorted)

        assert sorted_ambs[0].source_text == "apple"
        assert sorted_ambs[1].source_text == "banana"
        assert sorted_ambs[2].source_text == "cherry"

    def test_prioritize_questions_empty_list(self) -> None:
        """Test prioritize_questions with empty list."""
        sorted_ambs = prioritize_questions([])
        assert sorted_ambs == []

    def test_prioritize_questions_single_item(self) -> None:
        """Test prioritize_questions with single item."""
        amb = self._create_ambiguity(AmbiguityType.SCOPE, AmbiguitySeverity.HIGH)
        sorted_ambs = prioritize_questions([amb])
        assert len(sorted_ambs) == 1
        assert sorted_ambs[0] == amb

    def test_prioritize_questions_returns_list(self) -> None:
        """Test that prioritize_questions returns a list (not tuple)."""
        amb = self._create_ambiguity(AmbiguityType.SCOPE, AmbiguitySeverity.HIGH)
        result = prioritize_questions([amb])
        assert isinstance(result, list)


class TestAmbiguityResultPrioritization:
    """Test AmbiguityResult prioritization features (Story 4.4 Task 6)."""

    def _create_ambiguity(
        self,
        amb_type: AmbiguityType,
        severity: AmbiguitySeverity,
        source_text: str = "test",
    ) -> Ambiguity:
        """Helper to create test ambiguities."""
        return Ambiguity(
            ambiguity_type=amb_type,
            severity=severity,
            source_text=source_text,
            location="line 1",
            description="Test description",
        )

    def test_prioritized_ambiguities_property(self) -> None:
        """Test that prioritized_ambiguities returns sorted tuple."""
        from yolo_developer.seed.ambiguity import AmbiguityResult

        amb_low = self._create_ambiguity(AmbiguityType.PRIORITY, AmbiguitySeverity.LOW, "low")
        amb_high = self._create_ambiguity(AmbiguityType.UNDEFINED, AmbiguitySeverity.HIGH, "high")
        amb_med = self._create_ambiguity(AmbiguityType.TECHNICAL, AmbiguitySeverity.MEDIUM, "med")

        result = AmbiguityResult(
            ambiguities=(amb_low, amb_med, amb_high),  # Unsorted
            overall_confidence=0.8,
            resolution_prompts=(),
        )

        prioritized = result.prioritized_ambiguities
        assert isinstance(prioritized, tuple)
        assert len(prioritized) == 3
        # Should be sorted: high > med > low
        assert prioritized[0].source_text == "high"
        assert prioritized[1].source_text == "med"
        assert prioritized[2].source_text == "low"

    def test_prioritized_ambiguities_empty(self) -> None:
        """Test prioritized_ambiguities with empty result."""
        from yolo_developer.seed.ambiguity import AmbiguityResult

        result = AmbiguityResult(
            ambiguities=(),
            overall_confidence=1.0,
            resolution_prompts=(),
        )

        assert result.prioritized_ambiguities == ()

    def test_get_highest_priority_ambiguity(self) -> None:
        """Test get_highest_priority_ambiguity returns top priority."""
        from yolo_developer.seed.ambiguity import AmbiguityResult

        amb_low = self._create_ambiguity(AmbiguityType.PRIORITY, AmbiguitySeverity.LOW, "low")
        amb_high = self._create_ambiguity(AmbiguityType.UNDEFINED, AmbiguitySeverity.HIGH, "high")

        result = AmbiguityResult(
            ambiguities=(amb_low, amb_high),  # Unsorted
            overall_confidence=0.8,
            resolution_prompts=(),
        )

        top = result.get_highest_priority_ambiguity()
        assert top is not None
        assert top.source_text == "high"

    def test_get_highest_priority_ambiguity_empty(self) -> None:
        """Test get_highest_priority_ambiguity returns None when empty."""
        from yolo_developer.seed.ambiguity import AmbiguityResult

        result = AmbiguityResult(
            ambiguities=(),
            overall_confidence=1.0,
            resolution_prompts=(),
        )

        assert result.get_highest_priority_ambiguity() is None

    def test_get_priority_score(self) -> None:
        """Test get_priority_score returns correct score."""
        from yolo_developer.seed.ambiguity import AmbiguityResult

        amb = self._create_ambiguity(AmbiguityType.UNDEFINED, AmbiguitySeverity.HIGH)

        result = AmbiguityResult(
            ambiguities=(amb,),
            overall_confidence=0.85,
            resolution_prompts=(),
        )

        score = result.get_priority_score(amb)
        # HIGH=30 + UNDEFINED=25 = 55
        assert score == 55

    def test_to_dict_includes_priority_scores(self) -> None:
        """Test to_dict() includes priority_scores list."""
        from yolo_developer.seed.ambiguity import AmbiguityResult, ResolutionPrompt

        amb1 = self._create_ambiguity(AmbiguityType.SCOPE, AmbiguitySeverity.HIGH, "scope_issue")
        amb2 = self._create_ambiguity(AmbiguityType.PRIORITY, AmbiguitySeverity.LOW, "priority_issue")

        prompt1 = ResolutionPrompt(question="Q1?", suggestions=())
        prompt2 = ResolutionPrompt(question="Q2?", suggestions=())

        result = AmbiguityResult(
            ambiguities=(amb1, amb2),
            overall_confidence=0.75,
            resolution_prompts=(prompt1, prompt2),
        )

        result_dict = result.to_dict()

        assert "priority_scores" in result_dict
        assert len(result_dict["priority_scores"]) == 2
        # amb1: HIGH=30 + SCOPE=20 = 50
        assert result_dict["priority_scores"][0] == 50
        # amb2: LOW=10 + PRIORITY=5 = 15
        assert result_dict["priority_scores"][1] == 15

    def test_to_dict_empty_priority_scores(self) -> None:
        """Test to_dict() with empty ambiguities has empty priority_scores."""
        from yolo_developer.seed.ambiguity import AmbiguityResult

        result = AmbiguityResult(
            ambiguities=(),
            overall_confidence=1.0,
            resolution_prompts=(),
        )

        result_dict = result.to_dict()
        assert result_dict["priority_scores"] == []


class TestValidateFormatResponse:
    """Test _validate_format_response() function from CLI (Story 4.4 code review fix)."""

    def test_boolean_yes_variations_normalized(self) -> None:
        """Test that yes/y/true/1 are normalized to 'yes'."""
        from yolo_developer.cli.commands.seed import _validate_format_response

        prompt = ResolutionPrompt(
            question="Is this required?",
            suggestions=(),
            answer_format=AnswerFormat.BOOLEAN,
        )

        # All these should normalize to "yes"
        assert _validate_format_response("yes", prompt) == "yes"
        assert _validate_format_response("y", prompt) == "yes"
        assert _validate_format_response("true", prompt) == "yes"
        assert _validate_format_response("1", prompt) == "yes"
        assert _validate_format_response("YES", prompt) == "yes"
        assert _validate_format_response("Y", prompt) == "yes"
        assert _validate_format_response("TRUE", prompt) == "yes"

    def test_boolean_no_variations_normalized(self) -> None:
        """Test that no/n/false/0 are normalized to 'no'."""
        from yolo_developer.cli.commands.seed import _validate_format_response

        prompt = ResolutionPrompt(
            question="Is this required?",
            suggestions=(),
            answer_format=AnswerFormat.BOOLEAN,
        )

        # All these should normalize to "no"
        assert _validate_format_response("no", prompt) == "no"
        assert _validate_format_response("n", prompt) == "no"
        assert _validate_format_response("false", prompt) == "no"
        assert _validate_format_response("0", prompt) == "no"
        assert _validate_format_response("NO", prompt) == "no"
        assert _validate_format_response("N", prompt) == "no"
        assert _validate_format_response("FALSE", prompt) == "no"

    def test_boolean_non_standard_kept_as_is(self) -> None:
        """Test that non-standard boolean input is kept as-is with warning."""
        from yolo_developer.cli.commands.seed import _validate_format_response

        prompt = ResolutionPrompt(
            question="Is this required?",
            suggestions=(),
            answer_format=AnswerFormat.BOOLEAN,
        )

        # Non-standard values should be kept as-is
        assert _validate_format_response("maybe", prompt) == "maybe"
        assert _validate_format_response("yep", prompt) == "yep"

    def test_numeric_valid_integers(self) -> None:
        """Test that valid integers pass numeric validation."""
        from yolo_developer.cli.commands.seed import _validate_format_response

        prompt = ResolutionPrompt(
            question="How many users?",
            suggestions=(),
            answer_format=AnswerFormat.NUMERIC,
        )

        assert _validate_format_response("100", prompt) == "100"
        assert _validate_format_response("1000", prompt) == "1000"

    def test_numeric_with_commas(self) -> None:
        """Test that numbers with commas are valid."""
        from yolo_developer.cli.commands.seed import _validate_format_response

        prompt = ResolutionPrompt(
            question="How many users?",
            suggestions=(),
            answer_format=AnswerFormat.NUMERIC,
        )

        # Commas are stripped for validation but kept in response
        assert _validate_format_response("1,000", prompt) == "1,000"
        assert _validate_format_response("1,000,000", prompt) == "1,000,000"

    def test_numeric_floats(self) -> None:
        """Test that floats pass numeric validation."""
        from yolo_developer.cli.commands.seed import _validate_format_response

        prompt = ResolutionPrompt(
            question="What percentage?",
            suggestions=(),
            answer_format=AnswerFormat.NUMERIC,
        )

        assert _validate_format_response("99.9", prompt) == "99.9"
        assert _validate_format_response("0.5", prompt) == "0.5"

    def test_numeric_invalid_text_kept(self) -> None:
        """Test that invalid numeric input is kept with warning."""
        from yolo_developer.cli.commands.seed import _validate_format_response

        prompt = ResolutionPrompt(
            question="How many users?",
            suggestions=(),
            answer_format=AnswerFormat.NUMERIC,
        )

        # Non-numeric should be kept with warning logged
        assert _validate_format_response("many", prompt) == "many"
        assert _validate_format_response("about 100", prompt) == "about 100"

    def test_date_valid_format(self) -> None:
        """Test that valid YYYY-MM-DD dates pass."""
        from yolo_developer.cli.commands.seed import _validate_format_response

        prompt = ResolutionPrompt(
            question="What's the deadline?",
            suggestions=(),
            answer_format=AnswerFormat.DATE,
        )

        assert _validate_format_response("2026-01-15", prompt) == "2026-01-15"
        assert _validate_format_response("2026-12-31", prompt) == "2026-12-31"

    def test_date_invalid_format_kept(self) -> None:
        """Test that invalid date formats are kept with warning."""
        from yolo_developer.cli.commands.seed import _validate_format_response

        prompt = ResolutionPrompt(
            question="What's the deadline?",
            suggestions=(),
            answer_format=AnswerFormat.DATE,
        )

        # These should be kept but would show warning
        assert _validate_format_response("2026-1-15", prompt) == "2026-1-15"
        assert _validate_format_response("01/15/2026", prompt) == "01/15/2026"
        assert _validate_format_response("Jan 15, 2026", prompt) == "Jan 15, 2026"

    def test_free_text_passes_through(self) -> None:
        """Test that FREE_TEXT format passes any input through."""
        from yolo_developer.cli.commands.seed import _validate_format_response

        prompt = ResolutionPrompt(
            question="Describe the feature",
            suggestions=(),
            answer_format=AnswerFormat.FREE_TEXT,
        )

        assert _validate_format_response("any text here", prompt) == "any text here"
        assert _validate_format_response("123", prompt) == "123"

    def test_choice_passes_through(self) -> None:
        """Test that CHOICE format passes any input through."""
        from yolo_developer.cli.commands.seed import _validate_format_response

        prompt = ResolutionPrompt(
            question="Select option",
            suggestions=("A", "B", "C"),
            answer_format=AnswerFormat.CHOICE,
        )

        assert _validate_format_response("A", prompt) == "A"
        assert _validate_format_response("custom", prompt) == "custom"

    def test_list_passes_through(self) -> None:
        """Test that LIST format passes any input through."""
        from yolo_developer.cli.commands.seed import _validate_format_response

        prompt = ResolutionPrompt(
            question="Enter items",
            suggestions=(),
            answer_format=AnswerFormat.LIST,
        )

        assert _validate_format_response("a, b, c", prompt) == "a, b, c"
