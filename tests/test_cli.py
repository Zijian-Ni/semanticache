"""Tests for the CLI module."""

from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from semanticache.cli import app

runner = CliRunner()


class TestCLIStats:
    def test_stats_table_output(self) -> None:
        result = runner.invoke(app, ["stats"])
        assert result.exit_code == 0
        assert "SemantiCache Statistics" in result.output

    def test_stats_json_output(self) -> None:
        result = runner.invoke(app, ["stats", "--format", "json"])
        assert result.exit_code == 0
        assert "total_requests" in result.output


class TestCLIClear:
    def test_clear_with_yes_flag(self) -> None:
        result = runner.invoke(app, ["clear", "--yes"])
        assert result.exit_code == 0
        assert "Cleared" in result.output

    def test_clear_aborted(self) -> None:
        result = runner.invoke(app, ["clear"], input="n\n")
        assert result.exit_code == 0
        assert "Aborted" in result.output


class TestCLIBenchmark:
    def test_benchmark_runs(self) -> None:
        result = runner.invoke(app, ["benchmark", "--entries", "10", "--dim", "64"])
        assert result.exit_code == 0
        assert "Benchmark Results" in result.output

    def test_benchmark_default(self) -> None:
        result = runner.invoke(app, ["benchmark", "--entries", "50"])
        assert result.exit_code == 0


class TestCLIServe:
    def test_serve_missing_deps(self) -> None:
        with patch.dict("sys.modules", {"uvicorn": None}):
            # Just verify the command exists; it will fail without uvicorn
            result = runner.invoke(app, ["serve", "--help"])
            assert result.exit_code == 0
            assert "dashboard" in result.output.lower() or "server" in result.output.lower()
