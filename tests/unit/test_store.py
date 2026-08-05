"""Tests for the scenario log.

Two things carry the design and are tested hardest: that identity comes from a
fingerprint rather than float equality, and that the foreign key is real rather than
declared and ignored.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from bsm.grids import Grid, price_surface
from bsm.store import Scenario, ScenarioStore, fingerprint
from bsm.types import OptionKind

VOLS = np.array([0.1, 0.2, 0.3])
SPOTS = np.array([90.0, 100.0, 110.0, 120.0])

TERMS = dict(strike=100.0, rate=0.05, dividend_yield=0.0, maturity=1.0)


def make_grid(kind: OptionKind = OptionKind.CALL) -> Grid:
    return price_surface(kind, volatilities=VOLS, spots=SPOTS, **TERMS)


@pytest.fixture
def store() -> Iterator[ScenarioStore]:
    """An ephemeral store, closed on teardown.

    The close is not tidiness. A connection left to the garbage collector emits
    `ResourceWarning` on Python 3.13, which `filterwarnings = ["error"]` turns into a
    failure — attributed to whichever test happened to be running when collection
    fired, rather than to the one that leaked. Python 3.12 does not warn, so this was
    invisible until the CI matrix ran 3.13.
    """
    with ScenarioStore(":memory:") as scenario_store:
        yield scenario_store


class TestFingerprint:
    def test_is_deterministic(self) -> None:
        first = fingerprint(OptionKind.CALL, 100.0, 0.05, 0.0, 1.0, VOLS, SPOTS)
        second = fingerprint(OptionKind.CALL, 100.0, 0.05, 0.0, 1.0, VOLS, SPOTS)
        assert first == second

    def test_length(self) -> None:
        assert (
            len(fingerprint(OptionKind.CALL, 100.0, 0.05, 0.0, 1.0, VOLS, SPOTS)) == 12
        )

    def test_survives_float_accumulation(self) -> None:
        """The whole reason for hashing a canonical form instead of keying on floats."""
        assert 0.1 + 0.2 != 0.3
        exact = fingerprint(OptionKind.CALL, 100.0, 0.3, 0.0, 1.0, VOLS, SPOTS)
        accumulated = fingerprint(
            OptionKind.CALL, 100.0, 0.1 + 0.2, 0.0, 1.0, VOLS, SPOTS
        )
        assert exact == accumulated

    def test_negative_zero_matches_zero(self) -> None:
        positive = fingerprint(OptionKind.CALL, 100.0, 0.05, 0.0, 1.0, VOLS, SPOTS)
        negative = fingerprint(OptionKind.CALL, 100.0, 0.05, -0.0, 1.0, VOLS, SPOTS)
        assert positive == negative

    @pytest.mark.parametrize(
        "changed",
        ["kind", "strike", "rate", "dividend_yield", "maturity", "rows", "columns"],
    )
    def test_every_parameter_affects_identity(self, changed: str) -> None:
        base = dict(
            kind=OptionKind.CALL,
            strike=100.0,
            rate=0.05,
            dividend_yield=0.0,
            maturity=1.0,
            rows=VOLS,
            columns=SPOTS,
        )
        altered = dict(base)
        altered[changed] = {
            "kind": OptionKind.PUT,
            "strike": 105.0,
            "rate": 0.06,
            "dividend_yield": 0.01,
            "maturity": 2.0,
            "rows": VOLS * 2.0,
            "columns": SPOTS * 2.0,
        }[changed]
        assert fingerprint(**base) != fingerprint(**altered)  # type: ignore[arg-type]

    def test_differences_below_the_canonical_resolution_collide(self) -> None:
        """Documented consequence: ten decimal places is the resolution of identity."""
        coarse = fingerprint(OptionKind.CALL, 100.0, 0.05, 0.0, 1.0, VOLS, SPOTS)
        finer = fingerprint(OptionKind.CALL, 100.0 + 1e-13, 0.05, 0.0, 1.0, VOLS, SPOTS)
        assert coarse == finer


class TestRecordAndLoad:
    def test_round_trip_is_exact(self, store: ScenarioStore) -> None:
        """SQLite REAL is an IEEE-754 double, so this is equality, not closeness."""
        grid = make_grid()
        scenario = store.record(grid, kind=OptionKind.CALL, **TERMS)
        reloaded = store.grid(scenario.fingerprint)

        assert np.array_equal(reloaded.values, grid.values)
        assert np.array_equal(reloaded.rows, grid.rows)
        assert np.array_equal(reloaded.columns, grid.columns)
        assert reloaded.row_label == grid.row_label
        assert reloaded.column_label == grid.column_label

    def test_shape_survives(self, store: ScenarioStore) -> None:
        """A non-square grid, so a transpose in the round trip would show."""
        grid = make_grid()
        assert grid.shape == (3, 4)
        scenario = store.record(grid, kind=OptionKind.CALL, **TERMS)
        assert store.grid(scenario.fingerprint).shape == (3, 4)

    def test_scenario_carries_its_terms(self, store: ScenarioStore) -> None:
        scenario = store.record(make_grid(), kind=OptionKind.PUT, **TERMS)
        assert isinstance(scenario, Scenario)
        assert scenario.kind is OptionKind.PUT
        assert scenario.strike == 100.0
        assert scenario.recorded_at.endswith("+00:00")

    def test_recording_twice_is_idempotent(self, store: ScenarioStore) -> None:
        grid = make_grid()
        first = store.record(grid, kind=OptionKind.CALL, **TERMS)
        second = store.record(grid, kind=OptionKind.CALL, **TERMS)
        assert first == second
        assert len(store.scenarios()) == 1

    def test_calls_and_puts_are_separate_scenarios(self, store: ScenarioStore) -> None:
        store.record(make_grid(OptionKind.CALL), kind=OptionKind.CALL, **TERMS)
        store.record(make_grid(OptionKind.PUT), kind=OptionKind.PUT, **TERMS)
        assert len(store.scenarios()) == 2

    def test_unknown_fingerprint_raises(self, store: ScenarioStore) -> None:
        with pytest.raises(KeyError, match="no scenario"):
            store.grid("deadbeefcafe")

    def test_scenario_lookup_returns_none_when_absent(
        self, store: ScenarioStore
    ) -> None:
        assert store.scenario("deadbeefcafe") is None

    def test_empty_store_lists_nothing(self, store: ScenarioStore) -> None:
        assert store.scenarios() == []


class TestReferentialIntegrity:
    def test_foreign_keys_are_actually_enforced(self, store: ScenarioStore) -> None:
        """Declared without the pragma, SQLite parses the constraint and ignores it."""
        connection = store._connection
        with pytest.raises(sqlite3.IntegrityError):
            with connection:
                connection.execute(
                    "INSERT INTO grid_points VALUES (?, ?, ?, ?, ?, ?)",
                    ("orphan000000", 0, 0, 0.1, 100.0, 1.0),
                )

    def test_delete_cascades_to_cells(self, store: ScenarioStore) -> None:
        scenario = store.record(make_grid(), kind=OptionKind.CALL, **TERMS)
        assert store.delete(scenario.fingerprint) is True

        remaining = store._connection.execute(
            "SELECT COUNT(*) FROM grid_points WHERE fingerprint = ?",
            (scenario.fingerprint,),
        ).fetchone()[0]
        assert remaining == 0

    def test_delete_reports_absence(self, store: ScenarioStore) -> None:
        assert store.delete("deadbeefcafe") is False

    def test_clear_empties_both_tables(self, store: ScenarioStore) -> None:
        store.record(make_grid(), kind=OptionKind.CALL, **TERMS)
        store.clear()
        assert store.scenarios() == []
        assert (
            store._connection.execute("SELECT COUNT(*) FROM grid_points").fetchone()[0]
            == 0
        )


class TestPersistence:
    def test_survives_reopening(self, tmp_path: Path) -> None:
        database = tmp_path / "scenarios.db"
        grid = make_grid()

        with ScenarioStore(database) as store:
            scenario = store.record(grid, kind=OptionKind.CALL, **TERMS)

        with ScenarioStore(database) as reopened:
            assert np.array_equal(
                reopened.grid(scenario.fingerprint).values, grid.values
            )

    def test_context_manager_closes(self, tmp_path: Path) -> None:
        with ScenarioStore(tmp_path / "s.db") as store:
            pass
        with pytest.raises(sqlite3.ProgrammingError):
            store.scenarios()

    def test_schema_creation_is_idempotent(self, tmp_path: Path) -> None:
        database = tmp_path / "scenarios.db"
        ScenarioStore(database).close()
        ScenarioStore(database).close()
