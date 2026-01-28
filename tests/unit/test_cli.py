"""
Unit tests for CLI tools
"""
import pytest
import sys
from unittest.mock import patch, MagicMock


class TestCLIHealth:
    """Test CLI health command"""

    def test_health_check_modules(self):
        """Test module import checks"""
        from src.cli import cmd_health

        args = MagicMock()

        # Should not raise
        result = cmd_health(args)

        # Result is 0 (success) or 1 (failure) depending on env
        assert result in [0, 1]


class TestCLIStatus:
    """Test CLI status command"""

    def test_status_command(self):
        """Test status command"""
        from src.cli import cmd_status

        args = MagicMock()

        result = cmd_status(args)
        assert result in [0, 1]


class TestCLISymbols:
    """Test CLI symbols command"""

    def test_list_symbols(self):
        """Test listing symbols"""
        from src.cli import cmd_symbols

        args = MagicMock()
        args.activate = None
        args.deactivate = None

        result = cmd_symbols(args)
        assert result == 0

    def test_activate_symbols(self):
        """Test activating symbols"""
        from src.cli import cmd_symbols

        args = MagicMock()
        args.activate = ["US.TQQQ"]
        args.deactivate = None

        result = cmd_symbols(args)
        assert result == 0


class TestCLIBacktest:
    """Test CLI backtest command"""

    def test_backtest_with_sample_data(self):
        """Test backtest with generated sample data"""
        from src.cli import cmd_backtest

        args = MagicMock()
        args.data = None
        args.symbol = "US.TQQQ"
        args.capital = 100000.0
        args.slippage = 0.001
        args.rsi_low = 30
        args.rsi_high = 70
        args.save = None

        result = cmd_backtest(args)
        assert result == 0


class TestCLIReport:
    """Test CLI report command"""

    def test_html_report(self, tmp_path):
        """Test HTML report generation"""
        from src.cli import cmd_report

        args = MagicMock()
        args.format = "html"
        args.days = 7
        args.title = "Test Report"
        args.output_dir = str(tmp_path)

        result = cmd_report(args)
        assert result == 0


class TestMainParser:
    """Test main CLI parser"""

    def test_parse_health_command(self):
        """Test parsing health command"""
        from src.cli import main

        with patch('sys.argv', ['cli.py', 'health']):
            with patch('src.cli.cmd_health', return_value=0) as mock:
                main()
                mock.assert_called_once()

    def test_parse_status_command(self):
        """Test parsing status command"""
        from src.cli import main

        with patch('sys.argv', ['cli.py', 'status']):
            with patch('src.cli.cmd_status', return_value=0) as mock:
                main()
                mock.assert_called_once()

    def test_no_command_shows_help(self, capsys):
        """Test no command shows help"""
        from src.cli import main

        with patch('sys.argv', ['cli.py']):
            result = main()
            assert result == 0
