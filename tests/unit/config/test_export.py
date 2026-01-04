"""Tests for configuration export and import functionality.

Tests cover:
- Exporting configuration to YAML files
- Excluding secrets (API keys) from exports
- Importing configuration from YAML files
- Validation during import
- Error handling for missing files
"""

from __future__ import annotations

from pathlib import Path

import pytest


class TestExportConfig:
    """Tests for export_config function."""

    def test_export_creates_yaml_file(self, tmp_path: Path) -> None:
        """Export should create a YAML file at the specified path."""
        from yolo_developer.config import YoloConfig
        from yolo_developer.config.export import export_config

        config = YoloConfig(project_name="test-project")
        output = tmp_path / "exported.yaml"

        export_config(config, output)

        assert output.exists()
        assert output.is_file()

    def test_exported_yaml_is_valid_and_loadable(self, tmp_path: Path) -> None:
        """Exported YAML should be valid and loadable by load_config."""
        from yolo_developer.config import YoloConfig, load_config
        from yolo_developer.config.export import export_config

        config = YoloConfig(project_name="test-project")
        output = tmp_path / "exported.yaml"

        export_config(config, output)

        # Should be loadable without errors
        loaded = load_config(output)
        assert loaded.project_name == config.project_name

    def test_export_preserves_project_name(self, tmp_path: Path) -> None:
        """Exported config should preserve the project name."""
        from yolo_developer.config import YoloConfig, load_config
        from yolo_developer.config.export import export_config

        config = YoloConfig(project_name="my-awesome-project")
        output = tmp_path / "exported.yaml"

        export_config(config, output)

        loaded = load_config(output)
        assert loaded.project_name == "my-awesome-project"

    def test_export_preserves_llm_settings(self, tmp_path: Path) -> None:
        """Exported config should preserve LLM model settings."""
        from yolo_developer.config import LLMConfig, YoloConfig, load_config
        from yolo_developer.config.export import export_config

        config = YoloConfig(
            project_name="test-project",
            llm=LLMConfig(
                cheap_model="gpt-3.5-turbo",
                premium_model="gpt-4",
                best_model="gpt-4-turbo",
            ),
        )
        output = tmp_path / "exported.yaml"

        export_config(config, output)

        loaded = load_config(output)
        assert loaded.llm.cheap_model == "gpt-3.5-turbo"
        assert loaded.llm.premium_model == "gpt-4"
        assert loaded.llm.best_model == "gpt-4-turbo"

    def test_export_preserves_quality_settings(self, tmp_path: Path) -> None:
        """Exported config should preserve quality threshold settings."""
        from yolo_developer.config import QualityConfig, YoloConfig, load_config
        from yolo_developer.config.export import export_config

        config = YoloConfig(
            project_name="test-project",
            quality=QualityConfig(
                test_coverage_threshold=0.95,
                confidence_threshold=0.85,
            ),
        )
        output = tmp_path / "exported.yaml"

        export_config(config, output)

        loaded = load_config(output)
        assert loaded.quality.test_coverage_threshold == 0.95
        assert loaded.quality.confidence_threshold == 0.85

    def test_export_preserves_memory_settings(self, tmp_path: Path) -> None:
        """Exported config should preserve memory settings."""
        from yolo_developer.config import MemoryConfig, YoloConfig, load_config
        from yolo_developer.config.export import export_config

        config = YoloConfig(
            project_name="test-project",
            memory=MemoryConfig(
                persist_path="/custom/path",
                vector_store_type="chromadb",
                graph_store_type="json",
            ),
        )
        output = tmp_path / "exported.yaml"

        export_config(config, output)

        loaded = load_config(output)
        assert loaded.memory.persist_path == "/custom/path"
        assert loaded.memory.vector_store_type == "chromadb"
        assert loaded.memory.graph_store_type == "json"


class TestExportDirectoryCreation:
    """Tests for export directory creation behavior."""

    def test_export_creates_parent_directories(self, tmp_path: Path) -> None:
        """Export should create parent directories if they don't exist."""
        from yolo_developer.config import YoloConfig
        from yolo_developer.config.export import export_config

        config = YoloConfig(project_name="test-project")
        output = tmp_path / "nested" / "deep" / "dir" / "exported.yaml"

        # Parent directories don't exist
        assert not output.parent.exists()

        export_config(config, output)

        assert output.exists()
        assert output.parent.exists()


class TestExportExcludesSecrets:
    """Tests for secret exclusion during export."""

    def test_export_excludes_openai_api_key(self, tmp_path: Path) -> None:
        """OpenAI API key should not appear in exported file."""
        import os

        from yolo_developer.config import load_config
        from yolo_developer.config.export import export_config

        # Create config with API key via env var
        yaml_file = tmp_path / "input.yaml"
        yaml_file.write_text("project_name: test-project\n")

        env_backup = os.environ.get("YOLO_LLM__OPENAI_API_KEY")
        try:
            os.environ["YOLO_LLM__OPENAI_API_KEY"] = "sk-test-secret-key"
            config = load_config(yaml_file)

            output = tmp_path / "exported.yaml"
            export_config(config, output)

            content = output.read_text()
            assert "openai_api_key" not in content
            assert "sk-test-secret-key" not in content
        finally:
            if env_backup is not None:
                os.environ["YOLO_LLM__OPENAI_API_KEY"] = env_backup
            else:
                os.environ.pop("YOLO_LLM__OPENAI_API_KEY", None)

    def test_export_excludes_anthropic_api_key(self, tmp_path: Path) -> None:
        """Anthropic API key should not appear in exported file."""
        import os

        from yolo_developer.config import load_config
        from yolo_developer.config.export import export_config

        # Create config with API key via env var
        yaml_file = tmp_path / "input.yaml"
        yaml_file.write_text("project_name: test-project\n")

        env_backup = os.environ.get("YOLO_LLM__ANTHROPIC_API_KEY")
        try:
            os.environ["YOLO_LLM__ANTHROPIC_API_KEY"] = "sk-ant-test-secret"
            config = load_config(yaml_file)

            output = tmp_path / "exported.yaml"
            export_config(config, output)

            content = output.read_text()
            assert "anthropic_api_key" not in content
            assert "sk-ant-test-secret" not in content
        finally:
            if env_backup is not None:
                os.environ["YOLO_LLM__ANTHROPIC_API_KEY"] = env_backup
            else:
                os.environ.pop("YOLO_LLM__ANTHROPIC_API_KEY", None)

    def test_export_includes_header_comment_about_secrets(self, tmp_path: Path) -> None:
        """Exported file should include header comment explaining API key setup."""
        from yolo_developer.config import YoloConfig
        from yolo_developer.config.export import export_config

        config = YoloConfig(project_name="test-project")
        output = tmp_path / "exported.yaml"

        export_config(config, output)

        content = output.read_text()
        assert "YOLO_LLM__OPENAI_API_KEY" in content
        assert "YOLO_LLM__ANTHROPIC_API_KEY" in content
        # Should be in comments
        lines = content.split("\n")
        api_key_lines = [line for line in lines if "API_KEY" in line]
        assert all(line.strip().startswith("#") for line in api_key_lines)


class TestImportConfig:
    """Tests for import_config function."""

    def test_import_reads_source_and_writes_target(self, tmp_path: Path) -> None:
        """Import should read source file and write to target path."""
        from yolo_developer.config.export import import_config

        # Create source file
        source = tmp_path / "source.yaml"
        source.write_text("project_name: imported-project\n")

        target = tmp_path / "target.yaml"

        import_config(source, target)

        assert target.exists()
        content = target.read_text()
        assert "imported-project" in content

    def test_import_defaults_target_to_yolo_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Import without target should write to yolo.yaml in current directory."""
        from yolo_developer.config.export import import_config

        # Change to tmp_path
        monkeypatch.chdir(tmp_path)

        # Create source file
        source = tmp_path / "source.yaml"
        source.write_text("project_name: default-target-test\n")

        import_config(source)

        default_target = tmp_path / "yolo.yaml"
        assert default_target.exists()
        content = default_target.read_text()
        assert "default-target-test" in content

    def test_import_validates_config_before_writing(self, tmp_path: Path) -> None:
        """Import should validate configuration before writing to target."""
        from yolo_developer.config.export import import_config
        from yolo_developer.config.loader import ConfigurationError

        # Create invalid source (missing required project_name)
        source = tmp_path / "invalid.yaml"
        source.write_text("llm:\n  cheap_model: gpt-4\n")

        target = tmp_path / "target.yaml"

        with pytest.raises(ConfigurationError) as exc_info:
            import_config(source, target)

        assert (
            "project_name" in str(exc_info.value).lower()
            or "invalid" in str(exc_info.value).lower()
        )
        # Target should not be created on validation failure
        assert not target.exists()


class TestImportDirectoryCreation:
    """Tests for import directory creation behavior."""

    def test_import_creates_parent_directories(self, tmp_path: Path) -> None:
        """Import should create parent directories if they don't exist."""
        from yolo_developer.config.export import import_config

        # Create source file
        source = tmp_path / "source.yaml"
        source.write_text("project_name: test-project\n")

        target = tmp_path / "nested" / "deep" / "dir" / "imported.yaml"

        # Parent directories don't exist
        assert not target.parent.exists()

        import_config(source, target)

        assert target.exists()
        assert target.parent.exists()

    def test_import_adds_header_comment(self, tmp_path: Path) -> None:
        """Imported file should include header comment explaining API key setup."""
        from yolo_developer.config.export import import_config

        # Create source file without header
        source = tmp_path / "source.yaml"
        source.write_text("project_name: test-project\n")

        target = tmp_path / "target.yaml"
        import_config(source, target)

        content = target.read_text()
        assert "YOLO_LLM__OPENAI_API_KEY" in content
        assert "YOLO_LLM__ANTHROPIC_API_KEY" in content


class TestImportErrorHandling:
    """Tests for import error handling."""

    def test_import_missing_file_raises_configuration_error(self, tmp_path: Path) -> None:
        """Import should raise ConfigurationError for missing source file."""
        from yolo_developer.config.export import import_config
        from yolo_developer.config.loader import ConfigurationError

        missing = tmp_path / "nonexistent.yaml"

        with pytest.raises(ConfigurationError) as exc_info:
            import_config(missing)

        assert "not found" in str(exc_info.value).lower()

    def test_import_missing_file_error_includes_full_path(self, tmp_path: Path) -> None:
        """Error message should include the full absolute path."""
        from yolo_developer.config.export import import_config
        from yolo_developer.config.loader import ConfigurationError

        missing = tmp_path / "nonexistent.yaml"

        with pytest.raises(ConfigurationError) as exc_info:
            import_config(missing)

        # Should contain the absolute path
        assert str(missing.absolute()) in str(exc_info.value)

    def test_import_invalid_yaml_raises_configuration_error(self, tmp_path: Path) -> None:
        """Import should raise ConfigurationError for invalid YAML."""
        from yolo_developer.config.export import import_config
        from yolo_developer.config.loader import ConfigurationError

        # Create invalid YAML
        source = tmp_path / "invalid.yaml"
        source.write_text("invalid: yaml: content:\n  - [broken")

        target = tmp_path / "target.yaml"

        with pytest.raises(ConfigurationError):
            import_config(source, target)


class TestRoundTrip:
    """Tests for export/import round-trip preservation."""

    def test_round_trip_preserves_all_values(self, tmp_path: Path) -> None:
        """Export then import should preserve all configuration values."""
        from yolo_developer.config import (
            LLMConfig,
            MemoryConfig,
            QualityConfig,
            YoloConfig,
            load_config,
        )
        from yolo_developer.config.export import export_config, import_config

        # Create config with all custom values
        original = YoloConfig(
            project_name="round-trip-test",
            llm=LLMConfig(
                cheap_model="custom-cheap",
                premium_model="custom-premium",
                best_model="custom-best",
            ),
            quality=QualityConfig(
                test_coverage_threshold=0.75,
                confidence_threshold=0.88,
            ),
            memory=MemoryConfig(
                persist_path="/custom/memory/path",
                vector_store_type="chromadb",
                graph_store_type="json",
            ),
        )

        # Export
        exported = tmp_path / "exported.yaml"
        export_config(original, exported)

        # Import to new location
        imported_path = tmp_path / "imported.yaml"
        import_config(exported, imported_path)

        # Load imported config
        loaded = load_config(imported_path)

        # Verify all values
        assert loaded.project_name == original.project_name
        assert loaded.llm.cheap_model == original.llm.cheap_model
        assert loaded.llm.premium_model == original.llm.premium_model
        assert loaded.llm.best_model == original.llm.best_model
        assert loaded.quality.test_coverage_threshold == original.quality.test_coverage_threshold
        assert loaded.quality.confidence_threshold == original.quality.confidence_threshold
        assert loaded.memory.persist_path == original.memory.persist_path
        assert loaded.memory.vector_store_type == original.memory.vector_store_type
        assert loaded.memory.graph_store_type == original.memory.graph_store_type
