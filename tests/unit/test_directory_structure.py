"""Unit tests for directory structure creation (Story 1.2)."""

import tempfile
from pathlib import Path

from yolo_developer.cli.commands.init import (
    create_conftest,
    create_directory_structure,
    create_mocks_stub,
)


class TestSourceModulesCreated:
    """Tests that all 12 source modules are created per architecture spec."""

    EXPECTED_MODULES = [
        "cli",
        "sdk",
        "mcp",
        "agents",
        "orchestrator",
        "memory",
        "gates",
        "seed",
        "llm",
        "audit",
        "config",
        "utils",
    ]

    def test_all_source_modules_created(self) -> None:
        """Verify all 12 source modules exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            create_directory_structure(project_path)

            src_dir = project_path / "src" / "yolo_developer"
            for module in self.EXPECTED_MODULES:
                assert (src_dir / module).is_dir(), f"Module {module} not created"

    def test_all_source_modules_have_init(self) -> None:
        """Verify all source modules have __init__.py."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            create_directory_structure(project_path)

            src_dir = project_path / "src" / "yolo_developer"
            for module in self.EXPECTED_MODULES:
                init_file = src_dir / module / "__init__.py"
                assert init_file.is_file(), f"Missing __init__.py in {module}"


class TestAgentsPromptsSubdirectory:
    """Tests for agents/prompts subdirectory."""

    def test_prompts_directory_exists(self) -> None:
        """Verify agents/prompts subdirectory exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            create_directory_structure(project_path)

            prompts_dir = project_path / "src" / "yolo_developer" / "agents" / "prompts"
            assert prompts_dir.is_dir()

    def test_prompts_has_init(self) -> None:
        """Verify agents/prompts has __init__.py."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            create_directory_structure(project_path)

            init_file = (
                project_path / "src" / "yolo_developer" / "agents" / "prompts" / "__init__.py"
            )
            assert init_file.is_file()


class TestGatesGatesSubdirectory:
    """Tests for gates/gates subdirectory."""

    def test_gates_gates_directory_exists(self) -> None:
        """Verify gates/gates subdirectory exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            create_directory_structure(project_path)

            gates_gates = project_path / "src" / "yolo_developer" / "gates" / "gates"
            assert gates_gates.is_dir()

    def test_gates_gates_has_init(self) -> None:
        """Verify gates/gates has __init__.py."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            create_directory_structure(project_path)

            init_file = (
                project_path / "src" / "yolo_developer" / "gates" / "gates" / "__init__.py"
            )
            assert init_file.is_file()


class TestTestDirectoriesCreated:
    """Tests for test directory structure."""

    EXPECTED_TEST_DIRS = [
        "unit",
        "unit/agents",
        "unit/gates",
        "unit/memory",
        "unit/seed",
        "unit/config",
        "integration",
        "e2e",
        "fixtures",
        "fixtures/seeds",
        "fixtures/states",
    ]

    def test_all_test_directories_created(self) -> None:
        """Verify all test directories exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            create_directory_structure(project_path)

            tests_dir = project_path / "tests"
            for dir_path in self.EXPECTED_TEST_DIRS:
                assert (tests_dir / dir_path).is_dir(), f"Test dir {dir_path} not created"

    def test_unit_subdirs_have_init(self) -> None:
        """Verify unit test subdirectories have __init__.py."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            create_directory_structure(project_path)

            unit_subdirs = ["agents", "gates", "memory", "seed", "config"]
            for subdir in unit_subdirs:
                init_file = project_path / "tests" / "unit" / subdir / "__init__.py"
                assert init_file.is_file(), f"Missing __init__.py in tests/unit/{subdir}"


class TestInitFilesCreated:
    """Tests for __init__.py file creation."""

    def test_src_yolo_developer_has_init(self) -> None:
        """Verify src/yolo_developer has __init__.py."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            create_directory_structure(project_path)

            init_file = project_path / "src" / "yolo_developer" / "__init__.py"
            assert init_file.is_file()

    def test_tests_has_init(self) -> None:
        """Verify tests directory has __init__.py."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            create_directory_structure(project_path)

            init_file = project_path / "tests" / "__init__.py"
            assert init_file.is_file()

    def test_init_files_are_valid_python(self) -> None:
        """Verify __init__.py files are valid Python."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            create_directory_structure(project_path)

            # Check a sample of __init__.py files
            init_files = [
                project_path / "src" / "yolo_developer" / "__init__.py",
                project_path / "src" / "yolo_developer" / "sdk" / "__init__.py",
                project_path / "tests" / "__init__.py",
            ]
            for init_file in init_files:
                content = init_file.read_text()
                # Should be valid Python (no syntax errors)
                compile(content, str(init_file), "exec")


class TestConftestCreation:
    """Tests for conftest.py creation."""

    def test_creates_conftest(self) -> None:
        """Test that conftest.py is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            (project_path / "tests").mkdir(parents=True)
            create_conftest(project_path)

            conftest_file = project_path / "tests" / "conftest.py"
            assert conftest_file.is_file()

    def test_conftest_is_valid_python(self) -> None:
        """Test that conftest.py is valid Python."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            (project_path / "tests").mkdir(parents=True)
            create_conftest(project_path)

            conftest_file = project_path / "tests" / "conftest.py"
            content = conftest_file.read_text()
            compile(content, str(conftest_file), "exec")

    def test_conftest_not_overwritten(self) -> None:
        """Test that existing conftest.py is not overwritten."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            (project_path / "tests").mkdir(parents=True)

            # Create existing conftest
            existing_content = "# Custom conftest\n"
            (project_path / "tests" / "conftest.py").write_text(existing_content)

            create_conftest(project_path)

            content = (project_path / "tests" / "conftest.py").read_text()
            assert content == existing_content


class TestMocksStubCreation:
    """Tests for mocks.py stub creation."""

    def test_creates_mocks_stub(self) -> None:
        """Test that mocks.py stub is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            (project_path / "tests" / "fixtures").mkdir(parents=True)
            create_mocks_stub(project_path)

            mocks_file = project_path / "tests" / "fixtures" / "mocks.py"
            assert mocks_file.is_file()

    def test_mocks_is_valid_python(self) -> None:
        """Test that mocks.py is valid Python."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            (project_path / "tests" / "fixtures").mkdir(parents=True)
            create_mocks_stub(project_path)

            mocks_file = project_path / "tests" / "fixtures" / "mocks.py"
            content = mocks_file.read_text()
            compile(content, str(mocks_file), "exec")

    def test_mocks_has_mock_llm_response(self) -> None:
        """Test that mocks.py contains MockLLMResponse class."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            (project_path / "tests" / "fixtures").mkdir(parents=True)
            create_mocks_stub(project_path)

            mocks_file = project_path / "tests" / "fixtures" / "mocks.py"
            content = mocks_file.read_text()
            assert "class MockLLMResponse" in content

    def test_mocks_not_overwritten(self) -> None:
        """Test that existing mocks.py is not overwritten."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            (project_path / "tests" / "fixtures").mkdir(parents=True)

            # Create existing mocks
            existing_content = "# Custom mocks\n"
            (project_path / "tests" / "fixtures" / "mocks.py").write_text(existing_content)

            create_mocks_stub(project_path)

            content = (project_path / "tests" / "fixtures" / "mocks.py").read_text()
            assert content == existing_content


class TestPyTypedMarker:
    """Tests for py.typed marker in directory structure."""

    def test_py_typed_location_in_structure(self) -> None:
        """Verify py.typed should be in src/yolo_developer/."""
        # This test verifies the expected location per architecture
        # The actual creation is tested in TestCreatePyTyped
        from yolo_developer.cli.commands.init import create_py_typed

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            (project_path / "src" / "yolo_developer").mkdir(parents=True)
            create_py_typed(project_path)

            py_typed = project_path / "src" / "yolo_developer" / "py.typed"
            assert py_typed.is_file()


class TestDirectoryStructureMatchesArchitecture:
    """Tests that verify the directory structure matches architecture spec exactly."""

    def test_no_extra_source_modules(self) -> None:
        """Verify no unexpected source modules are created."""
        expected_items = {
            "__init__.py",
            "cli",
            "sdk",
            "mcp",
            "agents",
            "orchestrator",
            "memory",
            "gates",
            "seed",
            "llm",
            "audit",
            "config",
            "utils",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            create_directory_structure(project_path)

            src_dir = project_path / "src" / "yolo_developer"
            actual_items = {item.name for item in src_dir.iterdir()}

            # Should only contain expected items
            unexpected = actual_items - expected_items
            assert not unexpected, f"Unexpected items in source: {unexpected}"

    def test_cli_commands_subdirectory(self) -> None:
        """Verify cli/commands subdirectory exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            create_directory_structure(project_path)

            commands_dir = project_path / "src" / "yolo_developer" / "cli" / "commands"
            assert commands_dir.is_dir()
            assert (commands_dir / "__init__.py").is_file()

    def test_fixtures_subdirectories(self) -> None:
        """Verify fixtures has seeds/ and states/ subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            create_directory_structure(project_path)

            fixtures_dir = project_path / "tests" / "fixtures"
            assert (fixtures_dir / "seeds").is_dir()
            assert (fixtures_dir / "states").is_dir()
