"""Implementation generation tests for Dev agent (Story 8.1, AC3).

Tests for _generate_implementation and _generate_tests functions.
"""

from __future__ import annotations

from yolo_developer.agents.dev.node import _generate_implementation, _generate_tests
from yolo_developer.agents.dev.types import CodeFile


class TestGenerateImplementation:
    """Tests for _generate_implementation function."""

    def test_generates_implementation_artifact(self) -> None:
        """Test generates ImplementationArtifact for story."""
        story = {"id": "story-001", "title": "User Authentication"}
        artifact = _generate_implementation(story)
        assert artifact.story_id == "story-001"
        assert artifact.implementation_status == "completed"

    def test_generates_code_file(self) -> None:
        """Test generates at least one code file."""
        story = {"id": "story-001", "title": "User Auth"}
        artifact = _generate_implementation(story)
        assert len(artifact.code_files) >= 1
        code_file = artifact.code_files[0]
        assert code_file.file_type == "source"
        assert "story_001" in code_file.file_path

    def test_generates_test_files(self) -> None:
        """Test generates test files for code files."""
        story = {"id": "story-001", "title": "User Auth"}
        artifact = _generate_implementation(story)
        assert len(artifact.test_files) >= 1
        test_file = artifact.test_files[0]
        assert test_file.test_type == "unit"

    def test_handles_story_without_id(self) -> None:
        """Test handles story without id gracefully."""
        story = {"title": "No ID Story"}
        artifact = _generate_implementation(story)
        assert artifact.story_id == "unknown"
        assert artifact.implementation_status == "completed"

    def test_handles_story_without_title(self) -> None:
        """Test handles story without title gracefully."""
        story = {"id": "story-001"}
        artifact = _generate_implementation(story)
        assert artifact.story_id == "story-001"
        assert "Untitled Story" in artifact.notes

    def test_includes_notes(self) -> None:
        """Test includes implementation notes."""
        story = {"id": "story-001", "title": "Test Story"}
        artifact = _generate_implementation(story)
        assert artifact.notes != ""
        assert "stub" in artifact.notes.lower() or "Story 8.2" in artifact.notes


class TestGenerateTests:
    """Tests for _generate_tests function."""

    def test_generates_test_for_source_file(self) -> None:
        """Test generates test file for source code file."""
        story = {"id": "story-001", "title": "Test Story"}
        code_file = CodeFile(
            file_path="src/module.py",
            content="def hello(): pass",
            file_type="source",
        )
        tests = _generate_tests(story, [code_file])
        assert len(tests) == 1
        assert tests[0].test_type == "unit"
        assert "test_" in tests[0].file_path

    def test_skips_non_source_files(self) -> None:
        """Test does not generate tests for non-source files."""
        story = {"id": "story-001", "title": "Test Story"}
        config_file = CodeFile(
            file_path="config.yaml",
            content="key: value",
            file_type="config",
        )
        tests = _generate_tests(story, [config_file])
        assert len(tests) == 0

    def test_generates_one_test_per_story(self) -> None:
        """Test generates exactly one test file per story (stub behavior).

        The current stub implementation generates a single test file per story,
        regardless of how many source files are provided. This is intentional -
        full per-file test generation will be implemented in Story 8.3.
        """
        story = {"id": "story-001", "title": "Test Story"}
        code_files = [
            CodeFile(file_path="src/a.py", content="pass", file_type="source"),
            CodeFile(file_path="src/b.py", content="pass", file_type="source"),
        ]
        tests = _generate_tests(story, code_files)
        # Stub generates one test file per story (based on story module name)
        # Full per-file generation in Story 8.3
        assert len(tests) == 1
        assert "story_001" in tests[0].file_path

    def test_handles_empty_code_files(self) -> None:
        """Test handles empty code files list."""
        story = {"id": "story-001", "title": "Test Story"}
        tests = _generate_tests(story, [])
        assert tests == []
