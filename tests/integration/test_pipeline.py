"""Integration tests: modules used together, as a caller would.

The unit suites check each module against an oracle. These check that the pieces
compose — that a price produced by one module is consumed correctly by the next, and
that the identities linking them survive the round trip.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from bsm import (
    Contract,
    Market,
    OptionKind,
    greeks,
    implied_volatility_for,
    parity_residual,
    price,
    price_bounds,
    price_surface,
)
from bsm.daycount import tenor_to_years
from bsm.greeks import vega
from bsm.grids import Grid
from bsm.pricing import call_price


class TestPriceToImpliedVolatilityAndBack:
    @pytest.mark.parametrize("kind", [OptionKind.CALL, OptionKind.PUT])
    @pytest.mark.parametrize("volatility", [0.05, 0.2, 0.6, 1.5])
    def test_full_round_trip(self, kind: OptionKind, volatility: float) -> None:
        contract = Contract(strike=100.0, maturity=1.0, kind=kind)
        market = Market(
            spot=105.0, rate=0.04, volatility=volatility, dividend_yield=0.02
        )

        observed = price(contract, market)
        recovered = implied_volatility_for(observed, contract, market)
        repriced = price(contract, Market(105.0, 0.04, recovered, 0.02))

        assert recovered == pytest.approx(volatility, abs=1e-9)
        assert repriced == pytest.approx(observed, abs=1e-12)

    def test_round_trip_from_a_calendar_tenor(self) -> None:
        """Dates in, implied volatility out, with no hand-computed year fraction."""
        from datetime import date

        maturity = tenor_to_years(date(2026, 8, 5), months=3)
        contract = Contract(strike=100.0, maturity=maturity, kind=OptionKind.CALL)
        market = Market(spot=100.0, rate=0.05, volatility=0.22)

        observed = price(contract, market)
        assert implied_volatility_for(observed, contract, market) == pytest.approx(
            0.22, abs=1e-9
        )


class TestGreeksAgainstPriceDifferences:
    def test_delta_predicts_a_small_move(self) -> None:
        """A first-order prediction should be right to second order."""
        contract = Contract(100.0, 1.0, OptionKind.CALL)
        market = Market(100.0, 0.05, 0.2)

        base = price(contract, market)
        result = greeks(contract, market)
        step = 0.01
        predicted = base + result.delta * step
        actual = price(contract, Market(100.0 + step, 0.05, 0.2))

        assert predicted == pytest.approx(actual, abs=1e-6)

    def test_delta_and_gamma_predict_a_larger_move(self) -> None:
        contract = Contract(100.0, 1.0, OptionKind.CALL)
        market = Market(100.0, 0.05, 0.2)

        base = price(contract, market)
        result = greeks(contract, market)
        step = 1.0
        predicted = base + result.delta * step + 0.5 * result.gamma * step**2
        actual = price(contract, Market(100.0 + step, 0.05, 0.2))

        assert predicted == pytest.approx(actual, rel=1e-4)

    def test_vega_predicts_a_volatility_move(self) -> None:
        contract = Contract(100.0, 1.0, OptionKind.CALL)
        market = Market(100.0, 0.05, 0.2)

        base = price(contract, market)
        step = 0.001
        predicted = base + float(vega(100.0, 100.0, 0.05, 0.0, 1.0, 0.2)) * step
        actual = price(contract, Market(100.0, 0.05, 0.2 + step))

        assert predicted == pytest.approx(actual, abs=1e-4)


class TestSurfaceConsistency:
    def test_every_cell_inverts_to_its_own_volatility(self) -> None:
        """The strongest end-to-end check available: grid, then invert each cell."""
        volatilities = np.linspace(0.1, 0.8, 6)
        spots = np.linspace(85.0, 115.0, 7)
        grid = price_surface(
            OptionKind.CALL,
            volatilities=volatilities,
            spots=spots,
            strike=100.0,
            rate=0.05,
            dividend_yield=0.01,
            maturity=1.0,
        )

        for i, volatility in enumerate(volatilities):
            for j, spot in enumerate(spots):
                contract = Contract(100.0, 1.0, OptionKind.CALL)
                market = Market(float(spot), 0.05, float(volatility), 0.01)
                recovered = implied_volatility_for(
                    float(grid.values[i, j]), contract, market
                )
                assert recovered == pytest.approx(float(volatility), abs=1e-8)

    def test_surface_stays_within_the_no_arbitrage_bounds(self) -> None:
        volatilities = np.linspace(0.05, 2.0, 10)
        spots = np.linspace(50.0, 200.0, 12)
        grid = price_surface(
            OptionKind.PUT,
            volatilities=volatilities,
            spots=spots,
            strike=100.0,
            rate=0.05,
            dividend_yield=0.0,
            maturity=1.0,
        )
        lower, upper = price_bounds(
            OptionKind.PUT, spots.reshape(1, -1), 100.0, 0.05, 0.0, 1.0
        )
        assert np.all(grid.values >= lower - 1e-9)
        assert np.all(grid.values <= upper + 1e-9)

    def test_call_and_put_surfaces_satisfy_parity(self) -> None:
        volatilities = np.linspace(0.1, 1.0, 5)
        spots = np.linspace(80.0, 120.0, 9)
        calls = price_surface(
            OptionKind.CALL,
            volatilities=volatilities,
            spots=spots,
            strike=100.0,
            rate=0.05,
            dividend_yield=0.02,
            maturity=1.0,
        )
        puts = price_surface(
            OptionKind.PUT,
            volatilities=volatilities,
            spots=spots,
            strike=100.0,
            rate=0.05,
            dividend_yield=0.02,
            maturity=1.0,
        )

        residual = parity_residual(
            calls.values, puts.values, spots.reshape(1, -1), 100.0, 0.05, 0.02, 1.0
        )
        assert np.all(np.abs(residual) < 1e-10)


class TestGoldenSurface:
    """A regression over the whole pipeline, recorded as data not as pixels."""

    VOLATILITIES = np.linspace(0.1, 0.5, 5)
    SPOTS = np.linspace(80.0, 120.0, 9)

    def build(self) -> Grid:
        return price_surface(
            OptionKind.CALL,
            volatilities=self.VOLATILITIES,
            spots=self.SPOTS,
            strike=100.0,
            rate=0.05,
            dividend_yield=0.0,
            maturity=1.0,
        )

    def test_surface_is_reproducible(self, tmp_path: Path) -> None:
        first, second = tmp_path / "a.csv", tmp_path / "b.csv"
        self.build().to_csv(first)
        self.build().to_csv(second)
        assert first.read_text() == second.read_text()

    def test_csv_reloads_bit_for_bit(self, tmp_path: Path) -> None:
        """Full precision survives the write, so the file is a usable golden."""
        grid = self.build()
        destination = tmp_path / "surface.csv"
        grid.to_csv(destination)

        with destination.open(encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        reloaded = np.array([[float(x) for x in row[1:]] for row in rows[1:]])

        assert np.array_equal(reloaded, grid.values)

    def test_known_cell_values(self) -> None:
        """Pins the composed pipeline, not just its parts."""
        grid = self.build()
        assert grid.values[1, 4] == pytest.approx(
            float(call_price(100.0, 100.0, 0.05, 0.0, 1.0, 0.2)), abs=0.0
        )
