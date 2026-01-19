"""Tests for CLI display utilities (Story 12.1).

Tests cover:
- success_panel outputs green panel
- error_panel outputs red panel
- info_panel outputs blue panel
- warning_panel outputs yellow panel
- coming_soon outputs warning panel with command name
- create_table returns configured Table
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rich.table import Table


class TestSuccessPanel:
    """Tests for success_panel function."""

    @patch("yolo_developer.cli.display.console")
    def test_success_panel_prints_green_panel(self, mock_console: MagicMock) -> None:
        """Test success_panel outputs a panel with green styling."""
        from yolo_developer.cli.display import success_panel

        success_panel("Operation completed")

        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args
        panel = call_args[0][0]
        assert panel.title == "Success"
        assert panel.border_style == "green"

    @patch("yolo_developer.cli.display.console")
    def test_success_panel_with_custom_title(self, mock_console: MagicMock) -> None:
        """Test success_panel with custom title."""
        from yolo_developer.cli.display import success_panel

        success_panel("Done!", title="Complete")

        call_args = mock_console.print.call_args
        panel = call_args[0][0]
        assert panel.title == "Complete"


class TestErrorPanel:
    """Tests for error_panel function."""

    @patch("yolo_developer.cli.display.console")
    def test_error_panel_prints_red_panel(self, mock_console: MagicMock) -> None:
        """Test error_panel outputs a panel with red styling."""
        from yolo_developer.cli.display import error_panel

        error_panel("Something went wrong")

        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args
        panel = call_args[0][0]
        assert panel.title == "Error"
        assert panel.border_style == "red"

    @patch("yolo_developer.cli.display.console")
    def test_error_panel_with_custom_title(self, mock_console: MagicMock) -> None:
        """Test error_panel with custom title."""
        from yolo_developer.cli.display import error_panel

        error_panel("Failed", title="Fatal Error")

        call_args = mock_console.print.call_args
        panel = call_args[0][0]
        assert panel.title == "Fatal Error"


class TestInfoPanel:
    """Tests for info_panel function."""

    @patch("yolo_developer.cli.display.console")
    def test_info_panel_prints_blue_panel(self, mock_console: MagicMock) -> None:
        """Test info_panel outputs a panel with blue styling."""
        from yolo_developer.cli.display import info_panel

        info_panel("Here is some information")

        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args
        panel = call_args[0][0]
        assert panel.title == "Info"
        assert panel.border_style == "blue"

    @patch("yolo_developer.cli.display.console")
    def test_info_panel_with_custom_title(self, mock_console: MagicMock) -> None:
        """Test info_panel with custom title."""
        from yolo_developer.cli.display import info_panel

        info_panel("Details", title="Details")

        call_args = mock_console.print.call_args
        panel = call_args[0][0]
        assert panel.title == "Details"


class TestWarningPanel:
    """Tests for warning_panel function."""

    @patch("yolo_developer.cli.display.console")
    def test_warning_panel_prints_yellow_panel(self, mock_console: MagicMock) -> None:
        """Test warning_panel outputs a panel with yellow styling."""
        from yolo_developer.cli.display import warning_panel

        warning_panel("Be careful!")

        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args
        panel = call_args[0][0]
        assert panel.title == "Warning"
        assert panel.border_style == "yellow"

    @patch("yolo_developer.cli.display.console")
    def test_warning_panel_with_custom_title(self, mock_console: MagicMock) -> None:
        """Test warning_panel with custom title."""
        from yolo_developer.cli.display import warning_panel

        warning_panel("Caution", title="Caution")

        call_args = mock_console.print.call_args
        panel = call_args[0][0]
        assert panel.title == "Caution"


class TestComingSoon:
    """Tests for coming_soon function."""

    @patch("yolo_developer.cli.display.console")
    def test_coming_soon_displays_command_name(self, mock_console: MagicMock) -> None:
        """Test coming_soon displays the command name in the message."""
        from yolo_developer.cli.display import coming_soon

        coming_soon("run")

        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args
        panel = call_args[0][0]
        assert panel.title == "Coming Soon"
        assert panel.border_style == "yellow"
        # Check renderable contains the command name
        renderable_str = str(panel.renderable)
        assert "run" in renderable_str

    @patch("yolo_developer.cli.display.console")
    def test_coming_soon_various_commands(self, mock_console: MagicMock) -> None:
        """Test coming_soon works with various command names."""
        from yolo_developer.cli.display import coming_soon

        for cmd in ["status", "logs", "tune", "config"]:
            mock_console.reset_mock()
            coming_soon(cmd)
            mock_console.print.assert_called_once()


class TestCreateTable:
    """Tests for create_table function."""

    def test_create_table_returns_table(self) -> None:
        """Test create_table returns a Rich Table."""
        from yolo_developer.cli.display import create_table

        table = create_table("Test Table", [("Column1", "cyan"), ("Column2", "green")])

        assert isinstance(table, Table)
        assert table.title == "Test Table"

    def test_create_table_has_correct_columns(self) -> None:
        """Test create_table configures columns correctly."""
        from yolo_developer.cli.display import create_table

        columns = [("Name", "cyan"), ("Value", "green"), ("Status", "yellow")]
        table = create_table("Results", columns)

        assert len(table.columns) == 3
        assert table.columns[0].header == "Name"
        assert table.columns[1].header == "Value"
        assert table.columns[2].header == "Status"

    def test_create_table_has_header_style(self) -> None:
        """Test create_table sets bold header style."""
        from yolo_developer.cli.display import create_table

        table = create_table("Test", [("Col", "cyan")])

        assert table.show_header is True
        assert table.header_style == "bold"

    def test_create_table_empty_columns(self) -> None:
        """Test create_table with empty columns list."""
        from yolo_developer.cli.display import create_table

        table = create_table("Empty", [])

        assert isinstance(table, Table)
        assert len(table.columns) == 0


class TestModuleExports:
    """Tests for module-level exports."""

    def test_console_is_exported(self) -> None:
        """Test that console is exported from display module."""
        from rich.console import Console

        from yolo_developer.cli.display import console

        assert isinstance(console, Console)

    def test_all_functions_importable(self) -> None:
        """Test all display functions can be imported."""
        from yolo_developer.cli.display import (
            coming_soon,
            console,
            create_table,
            error_panel,
            info_panel,
            success_panel,
            warning_panel,
        )

        assert callable(success_panel)
        assert callable(error_panel)
        assert callable(info_panel)
        assert callable(warning_panel)
        assert callable(coming_soon)
        assert callable(create_table)
        assert console is not None
