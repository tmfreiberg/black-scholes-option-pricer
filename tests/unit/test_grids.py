"""Tests for two-dimensional sensitivity surfaces."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from bsm.grids import Grid, price_surface
from bsm.pricing import call_price, put_price
from bsm.types import OptionKind

VOLS = np.linspace(0.1, 0.5, 5)
SPOTS = np.linspace(80.0, 120.0, 9)


def surface(kind: OptionKind = OptionKind.CALL) -> Grid:
    return price_surface(
        kind,
        volatilities=VOLS,
        spots=SPOTS,
        strike=100.0,
        rate=0.05,
        dividend_yield=0.0,
        maturity=1.0,
    )


class TestOrientation:
    def test_shape_is_rows_by_columns(self) -> None:
        assert surface().shape == (len(VOLS), len(SPOTS))

    def test_axes_are_stored_unchanged(self) -> None:
        grid = surface()
        assert np.array_equal(grid.rows, VOLS)
        assert np.array_equal(grid.columns, SPOTS)

    def test_labels(self) -> None:
        grid = surface()
        assert (grid.row_label, grid.column_label) == ("volatility", "spot")

    @pytest.mark.parametrize("i", range(5))
    @pytest.mark.parametrize("j", range(9))
    def test_every_cell_matches_a_scalar_call(self, i: int, j: int) -> None:
        """Guards against a transpose, which a square grid would hide."""
        grid = surface()
        expected = call_price(float(SPOTS[j]), 100.0, 0.05, 0.0, 1.0, float(VOLS[i]))
        assert grid.values[i, j] == expected

    def test_non_square_grid_is_not_transposed(self) -> None:
        grid = surface()
        assert grid.values.shape[0] != grid.values.shape[1]


class TestMonotonicity:
    def test_call_increases_along_both_axes(self) -> None:
        values = surface(OptionKind.CALL).values
        assert np.all(np.diff(values, axis=0) > 0.0)
        assert np.all(np.diff(values, axis=1) > 0.0)

    def test_put_increases_in_volatility_and_decreases_in_spot(self) -> None:
        values = surface(OptionKind.PUT).values
        assert np.all(np.diff(values, axis=0) > 0.0)
        assert np.all(np.diff(values, axis=1) < 0.0)

    def test_put_surface_matches_scalar_calls(self) -> None:
        grid = surface(OptionKind.PUT)
        assert grid.values[2, 4] == put_price(
            float(SPOTS[4]), 100.0, 0.05, 0.0, 1.0, float(VOLS[2])
        )


class TestValidation:
    def test_mismatched_shape_rejected(self) -> None:
        with pytest.raises(ValueError, match="expected"):
            Grid(
                rows=np.array([1.0, 2.0]),
                columns=np.array([1.0, 2.0, 3.0]),
                values=np.zeros((3, 2)),
                row_label="a",
                column_label="b",
            )

    def test_two_dimensional_axis_rejected(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional"):
            price_surface(
                OptionKind.CALL,
                volatilities=VOLS.reshape(-1, 1),
                spots=SPOTS,
                strike=100.0,
                rate=0.05,
                dividend_yield=0.0,
                maturity=1.0,
            )

    def test_two_dimensional_spots_rejected(self) -> None:
        with pytest.raises(ValueError, match="spots must be one-dimensional"):
            price_surface(
                OptionKind.CALL,
                volatilities=VOLS,
                spots=SPOTS.reshape(-1, 1),
                strike=100.0,
                rate=0.05,
                dividend_yield=0.0,
                maturity=1.0,
            )

    def test_grid_is_frozen(self) -> None:
        assert not hasattr(surface(), "__dict__")


class TestCsv:
    def test_round_trips_through_csv(self, tmp_path: Path) -> None:
        grid = surface()
        destination = tmp_path / "surface.csv"
        grid.to_csv(destination)

        with destination.open(encoding="utf-8") as handle:
            rows = list(csv.reader(handle))

        assert rows[0][0] == "volatility\\spot"
        assert [float(x) for x in rows[0][1:]] == pytest.approx(SPOTS.tolist())
        assert len(rows) == len(VOLS) + 1

        recovered = np.array([[float(x) for x in row[1:]] for row in rows[1:]])
        assert np.array_equal(recovered, grid.values)

    def test_row_labels_are_written(self, tmp_path: Path) -> None:
        destination = tmp_path / "surface.csv"
        surface().to_csv(destination)
        with destination.open(encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        assert [float(row[0]) for row in rows[1:]] == pytest.approx(VOLS.tolist())

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        destination = tmp_path / "surface.csv"
        destination.write_text("stale content that must not survive\n")
        surface().to_csv(destination)
        assert "stale" not in destination.read_text()
