"""End-to-end tests: the package as a user meets it, through the shell.

Driven with typer's `CliRunner`, which invokes the real command tree in process. These
are the tests that would catch a broken entry point, a renamed option, or a command
that computes correctly and then prints nothing.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bsm.cli import app
from bsm.pricing import call_price
from bsm.store import ScenarioStore

runner = CliRunner()


class TestPriceCommand:
    def test_matches_the_library(self) -> None:
        result = runner.invoke(
            app,
            ["price", "-s", "100", "-K", "100", "-r", "5", "-t", "1", "-v", "20"],
        )
        assert result.exit_code == 0
        expected = float(call_price(100.0, 100.0, 0.05, 0.0, 1.0, 0.2))
        assert float(result.stdout.strip()) == pytest.approx(expected, abs=1e-6)

    def test_percentages_are_converted_at_the_boundary(self) -> None:
        """`-v 20` must mean 20%, not 2000%. A missing division shows here."""
        result = runner.invoke(app, ["price", "-v", "20"])
        as_decimal = runner.invoke(app, ["price", "-v", "2000"])
        assert float(result.stdout.strip()) == pytest.approx(10.450584, abs=1e-5)
        assert float(as_decimal.stdout.strip()) > 99.0

    def test_put_differs_from_call(self) -> None:
        call = runner.invoke(app, ["price", "-k", "call"]).stdout.strip()
        put = runner.invoke(app, ["price", "-k", "put"]).stdout.strip()
        assert float(call) != float(put)

    def test_rejects_an_invalid_kind(self) -> None:
        assert runner.invoke(app, ["price", "-k", "straddle"]).exit_code != 0

    def test_rejects_a_negative_strike(self) -> None:
        result = runner.invoke(app, ["price", "-K", "-100"])
        assert result.exit_code != 0


class TestGreeksCommand:
    def test_lists_all_five(self) -> None:
        result = runner.invoke(app, ["greeks"])
        assert result.exit_code == 0
        for greek in ("delta", "gamma", "vega", "theta", "rho"):
            assert greek in result.stdout

    def test_quotes_in_conventional_units(self) -> None:
        """Vega per volatility point, theta per day, rho per basis point."""
        result = runner.invoke(app, ["greeks"])
        assert "volatility point" in result.stdout
        assert "calendar day" in result.stdout
        assert "basis point" in result.stdout


class TestImpliedCommand:
    def test_round_trips_a_price_from_the_price_command(self) -> None:
        priced = runner.invoke(app, ["price", "-v", "35"]).stdout.strip()
        result = runner.invoke(app, ["implied", priced])
        assert result.exit_code == 0
        assert float(result.stdout.strip().rstrip("%")) == pytest.approx(35.0, abs=1e-3)

    def test_impossible_price_exits_nonzero(self) -> None:
        """The library raises rather than returning a sentinel; the shell sees it."""
        result = runner.invoke(app, ["implied", "500"])
        assert result.exit_code == 1
        assert "bound" in result.stdout

    def test_expired_contract_exits_nonzero(self) -> None:
        assert runner.invoke(app, ["implied", "20", "-t", "0"]).exit_code == 1


class TestSurfaceCommand:
    def test_prints_a_table_by_default(self) -> None:
        result = runner.invoke(app, ["surface", "--spot-steps", "3"])
        assert result.exit_code == 0
        assert "volatility" in result.stdout

    def test_writes_csv(self, tmp_path: Path) -> None:
        destination = tmp_path / "surface.csv"
        result = runner.invoke(
            app,
            [
                "surface",
                "--spot-steps",
                "3",
                "--volatility-steps",
                "2",
                "--csv",
                str(destination),
            ],
        )
        assert result.exit_code == 0
        with destination.open(encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        assert len(rows) == 3
        assert len(rows[0]) == 4

    def test_writes_an_image(self, tmp_path: Path) -> None:
        destination = tmp_path / "surface.png"
        result = runner.invoke(
            app, ["surface", "--spot-steps", "3", "--image", str(destination)]
        )
        assert result.exit_code == 0
        assert destination.stat().st_size > 1_000

    def test_records_to_the_store(self, tmp_path: Path) -> None:
        database = tmp_path / "scenarios.db"
        result = runner.invoke(
            app, ["surface", "--spot-steps", "3", "--database", str(database)]
        )
        assert result.exit_code == 0
        with ScenarioStore(database) as store:
            assert len(store.scenarios()) == 1

    def test_recording_twice_records_once(self, tmp_path: Path) -> None:
        database = tmp_path / "scenarios.db"
        for _ in range(2):
            runner.invoke(
                app, ["surface", "--spot-steps", "3", "--database", str(database)]
            )
        with ScenarioStore(database) as store:
            assert len(store.scenarios()) == 1


class TestStoreCommands:
    def test_list_on_an_empty_database(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["store", "list", "--database", str(tmp_path / "s.db")]
        )
        assert result.exit_code == 0
        assert "no scenarios" in result.stdout

    def test_full_lifecycle(self, tmp_path: Path) -> None:
        """Record, list, show, clear — the sequence a user would actually run."""
        database = str(tmp_path / "scenarios.db")

        recorded = runner.invoke(
            app, ["surface", "--spot-steps", "3", "--database", database]
        )
        digest = recorded.stdout.strip().split()[-1]

        listed = runner.invoke(app, ["store", "list", "--database", database])
        assert digest in listed.stdout

        shown = runner.invoke(app, ["store", "show", digest, "--database", database])
        assert shown.exit_code == 0
        assert "volatility" in shown.stdout

        cleared = runner.invoke(
            app, ["store", "clear", "--database", database, "--force"]
        )
        assert cleared.exit_code == 0

        empty = runner.invoke(app, ["store", "list", "--database", database])
        assert "no scenarios" in empty.stdout

    def test_show_unknown_fingerprint_exits_nonzero(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["store", "show", "deadbeefcafe", "--database", str(tmp_path / "s.db")],
        )
        assert result.exit_code == 1

    def test_clear_asks_before_deleting(self, tmp_path: Path) -> None:
        database = str(tmp_path / "scenarios.db")
        runner.invoke(app, ["surface", "--spot-steps", "3", "--database", database])

        declined = runner.invoke(
            app, ["store", "clear", "--database", database], input="n\n"
        )
        assert declined.exit_code != 0

        with ScenarioStore(database) as store:
            assert len(store.scenarios()) == 1


class TestTopLevel:
    def test_version(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "bsm-pricer" in result.stdout

    def test_bare_invocation_shows_help(self) -> None:
        result = runner.invoke(app, [])
        assert "price" in result.stdout
        assert "greeks" in result.stdout
        assert "implied" in result.stdout
        assert "surface" in result.stdout

    def test_console_entry_point_runs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`main` is what `pip install` wires to the `bsm` command."""
        from bsm.cli import main

        monkeypatch.setattr("sys.argv", ["bsm", "--version"])
        with pytest.raises(SystemExit) as exit_info:
            main()
        assert exit_info.value.code == 0

    def test_every_command_has_help(self) -> None:
        for command in ("price", "greeks", "implied", "surface"):
            result = runner.invoke(app, [command, "--help"])
            assert result.exit_code == 0
