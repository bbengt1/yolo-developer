"""Implementation generation tests for Dev agent (Story 8.1, 8.2, AC3).

Tests for _generate_implementation, _generate_stub_implementation, and _generate_tests functions.
Updated for Story 8.2 with async implementation and context parameter.
"""

from __future__ import annotations

import pytest

from yolo_developer.agents.dev.node import (
    _generate_implementation,
    _generate_stub_implementation,
    _generate_tests,
)
from yolo_developer.agents.dev.types import CodeFile


class TestGenerateStubImplementation:
    """Tests for _generate_stub_implementation function (fallback)."""

    @pytest.mark.asyncio
    async def test_generates_implementation_artifact(self) -> None:
        """Test generates ImplementationArtifact for story."""
        story = {"id": "story-001", "title": "User Authentication"}
        artifact = await _generate_stub_implementation(story)
        assert artifact.story_id == "story-001"
        assert artifact.implementation_status == "completed"

    @pytest.mark.asyncio
    async def test_generates_code_file(self) -> None:
        """Test generates at least one code file."""
        story = {"id": "story-001", "title": "User Auth"}
        artifact = await _generate_stub_implementation(story)
        assert len(artifact.code_files) >= 1
        code_file = artifact.code_files[0]
        assert code_file.file_type == "source"
        assert "story_001" in code_file.file_path

    @pytest.mark.asyncio
    async def test_generates_test_files(self) -> None:
        """Test generates test files for code files."""
        story = {"id": "story-001", "title": "User Auth"}
        artifact = await _generate_stub_implementation(story)
        assert len(artifact.test_files) >= 1
        test_file = artifact.test_files[0]
        assert test_file.test_type == "unit"

    @pytest.mark.asyncio
    async def test_handles_story_without_id(self) -> None:
        """Test handles story without id gracefully."""
        story = {"title": "No ID Story"}
        artifact = await _generate_stub_implementation(story)
        assert artifact.story_id == "unknown"
        assert artifact.implementation_status == "completed"

    @pytest.mark.asyncio
    async def test_handles_story_without_title(self) -> None:
        """Test handles story without title gracefully."""
        story = {"id": "story-001"}
        artifact = await _generate_stub_implementation(story)
        assert artifact.story_id == "story-001"
        assert "Untitled Story" in artifact.notes

    @pytest.mark.asyncio
    async def test_includes_notes(self) -> None:
        """Test includes implementation notes."""
        story = {"id": "story-001", "title": "Test Story"}
        artifact = await _generate_stub_implementation(story)
        assert artifact.notes != ""
        assert "stub" in artifact.notes.lower() or "fallback" in artifact.notes.lower()


class TestGenerateImplementation:
    """Tests for async _generate_implementation function."""

    @pytest.mark.asyncio
    async def test_generates_implementation_artifact(self) -> None:
        """Test generates ImplementationArtifact for story."""
        story = {"id": "story-001", "title": "User Authentication"}
        context: dict = {"patterns": [], "constraints": [], "conventions": {}}
        artifact = await _generate_implementation(story, context)
        assert artifact.story_id == "story-001"
        assert artifact.implementation_status == "completed"

    @pytest.mark.asyncio
    async def test_generates_code_file(self) -> None:
        """Test generates at least one code file."""
        story = {"id": "story-001", "title": "User Auth"}
        context: dict = {"patterns": [], "constraints": [], "conventions": {}}
        artifact = await _generate_implementation(story, context)
        assert len(artifact.code_files) >= 1
        code_file = artifact.code_files[0]
        assert code_file.file_type == "source"
        assert "story_001" in code_file.file_path

    @pytest.mark.asyncio
    async def test_generates_test_files(self) -> None:
        """Test generates test files for code files."""
        story = {"id": "story-001", "title": "User Auth"}
        context: dict = {"patterns": [], "constraints": [], "conventions": {}}
        artifact = await _generate_implementation(story, context)
        assert len(artifact.test_files) >= 1
        test_file = artifact.test_files[0]
        assert test_file.test_type == "unit"

    @pytest.mark.asyncio
    async def test_handles_story_without_id(self) -> None:
        """Test handles story without id gracefully."""
        story = {"title": "No ID Story"}
        context: dict = {"patterns": [], "constraints": [], "conventions": {}}
        artifact = await _generate_implementation(story, context)
        assert artifact.story_id == "unknown"
        assert artifact.implementation_status == "completed"

    @pytest.mark.asyncio
    async def test_handles_story_without_title(self) -> None:
        """Test handles story without title gracefully."""
        story = {"id": "story-001"}
        context: dict = {"patterns": [], "constraints": [], "conventions": {}}
        artifact = await _generate_implementation(story, context)
        assert artifact.story_id == "story-001"
        assert "Untitled Story" in artifact.notes or "fallback" in artifact.notes.lower()

    @pytest.mark.asyncio
    async def test_includes_notes(self) -> None:
        """Test includes implementation notes."""
        story = {"id": "story-001", "title": "Test Story"}
        context: dict = {"patterns": [], "constraints": [], "conventions": {}}
        artifact = await _generate_implementation(story, context)
        assert artifact.notes != ""

    @pytest.mark.asyncio
    async def test_uses_stub_when_no_router(self) -> None:
        """Test falls back to stub when no router provided."""
        story = {"id": "story-001", "title": "Test Story"}
        context: dict = {"patterns": [], "constraints": [], "conventions": {}}
        # No router = None, so should use stub
        artifact = await _generate_implementation(story, context, router=None)
        assert artifact.story_id == "story-001"
        assert "stub" in artifact.notes.lower() or "fallback" in artifact.notes.lower()


class TestGenerateTests:
    """Tests for _generate_tests async function (Story 8.3)."""

    @pytest.mark.asyncio
    async def test_generates_test_for_source_file(self) -> None:
        """Test generates test file for source code file."""
        story = {"id": "story-001", "title": "Test Story"}
        code_file = CodeFile(
            file_path="src/module.py",
            content="def hello(): pass",
            file_type="source",
        )
        tests = await _generate_tests(story, [code_file])
        assert len(tests) == 1
        assert tests[0].test_type == "unit"
        assert "test_" in tests[0].file_path

    @pytest.mark.asyncio
    async def test_skips_non_source_files(self) -> None:
        """Test does not generate tests for non-source files."""
        story = {"id": "story-001", "title": "Test Story"}
        config_file = CodeFile(
            file_path="config.yaml",
            content="key: value",
            file_type="config",
        )
        tests = await _generate_tests(story, [config_file])
        assert len(tests) == 0

    @pytest.mark.asyncio
    async def test_generates_one_test_per_source_file(self) -> None:
        """Test generates one test file per source file (Story 8.3 behavior).

        Story 8.3 updated the behavior to generate one test file per source
        file, rather than one per story. Each source file gets its own test.
        """
        story = {"id": "story-001", "title": "Test Story"}
        code_files = [
            CodeFile(file_path="src/a.py", content="def foo(): pass", file_type="source"),
            CodeFile(file_path="src/b.py", content="def bar(): pass", file_type="source"),
        ]
        tests = await _generate_tests(story, code_files)
        # Story 8.3: One test file per source file
        assert len(tests) == 2
        assert "test_a" in tests[0].file_path
        assert "test_b" in tests[1].file_path

    @pytest.mark.asyncio
    async def test_handles_empty_code_files(self) -> None:
        """Test handles empty code files list."""
        story = {"id": "story-001", "title": "Test Story"}
        tests = await _generate_tests(story, [])
        assert tests == []
