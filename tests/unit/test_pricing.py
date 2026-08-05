"""Unit tests for the pricing formulas.

Reference values are computed independently in the test rather than copied from the
implementation, so the suite would catch a formula that is self-consistently wrong.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bsm.normal import cdf
from bsm.pricing import call_price, d1_d2, forward_price, price, put_price
from bsm.types import Contract, Market, OptionKind


def reference_call(s: float, k: float, r: float, q: float, t: float, v: float) -> float:
    """Independent scalar implementation, written straight from the textbook."""
    if v * math.sqrt(t) == 0.0:
        return math.exp(-r * t) * max(s * math.exp((r - q) * t) - k, 0.0)
    d1 = (math.log(s / k) + (r - q + 0.5 * v * v) * t) / (v * math.sqrt(t))
    d2 = d1 - v * math.sqrt(t)

    def n(x: float) -> float:
        return 0.5 * math.erfc(-x / math.sqrt(2.0))

    return s * math.exp(-q * t) * n(d1) - k * math.exp(-r * t) * n(d2)


class TestForward:
    def test_no_yield(self) -> None:
        assert forward_price(100.0, 0.05, 0.0, 1.0) == pytest.approx(
            100.0 * math.exp(0.05)
        )

    def test_yield_equal_to_rate(self) -> None:
        assert forward_price(100.0, 0.05, 0.05, 1.0) == 100.0

    def test_zero_maturity(self) -> None:
        assert forward_price(100.0, 0.05, 0.0, 0.0) == 100.0


class TestD1D2:
    def test_known_values(self) -> None:
        d1, d2 = d1_d2(100.0, 100.0, 0.05, 0.0, 1.0, 0.2)
        assert d1 == pytest.approx(0.35)
        assert d2 == pytest.approx(0.15)

    def test_difference_is_total_volatility(self) -> None:
        d1, d2 = d1_d2(97.0, 103.0, 0.03, 0.01, 0.75, 0.28)
        assert d1 - d2 == pytest.approx(0.28 * math.sqrt(0.75))

    @pytest.mark.parametrize(
        ("spot", "maturity", "volatility", "expected"),
        [
            (120.0, 0.0, 0.2, math.inf),
            (80.0, 0.0, 0.2, -math.inf),
            (120.0, 1.0, 0.0, math.inf),
            (80.0, 1.0, 0.0, -math.inf),
        ],
    )
    def test_degenerate_limits(
        self, spot: float, maturity: float, volatility: float, expected: float
    ) -> None:
        d1, d2 = d1_d2(spot, 100.0, 0.05, 0.0, maturity, volatility)
        assert d1 == expected
        assert d2 == expected

    def test_zero_volatility_uses_forward_not_spot(self) -> None:
        """Spot below strike but forward above it: the limit must follow the forward."""
        d1, _ = d1_d2(99.0, 100.0, 0.20, 0.0, 1.0, 0.0)
        assert forward_price(99.0, 0.20, 0.0, 1.0) > 100.0
        assert d1 == math.inf


class TestCallAndPut:
    def test_textbook_value(self) -> None:
        assert call_price(100.0, 100.0, 0.05, 0.0, 1.0, 0.2) == pytest.approx(
            10.450583572185565, abs=1e-12
        )

    @pytest.mark.parametrize(
        ("s", "k", "r", "q", "t", "v"),
        [
            (100.0, 100.0, 0.05, 0.00, 1.00, 0.20),
            (42.0, 40.0, 0.10, 0.00, 0.50, 0.20),
            (49.0, 50.0, 0.05, 0.00, 0.3846, 0.20),
            (930.0, 900.0, 0.08, 0.03, 0.25, 0.20),
            (100.0, 150.0, 0.01, 0.02, 3.00, 0.55),
            (100.0, 60.0, -0.01, 0.00, 0.10, 0.15),
        ],
    )
    def test_against_independent_implementation(
        self, s: float, k: float, r: float, q: float, t: float, v: float
    ) -> None:
        assert call_price(s, k, r, q, t, v) == pytest.approx(
            reference_call(s, k, r, q, t, v), abs=1e-12
        )

    @pytest.mark.parametrize(
        ("spot", "expected"),
        [(120.0, 20.0), (100.0, 0.0), (80.0, 0.0)],
    )
    def test_call_at_expiry_is_payoff(self, spot: float, expected: float) -> None:
        assert call_price(spot, 100.0, 0.05, 0.0, 0.0, 0.2) == expected

    @pytest.mark.parametrize(
        ("spot", "expected"),
        [(80.0, 20.0), (100.0, 0.0), (120.0, 0.0)],
    )
    def test_put_at_expiry_is_payoff(self, spot: float, expected: float) -> None:
        assert put_price(spot, 100.0, 0.05, 0.0, 0.0, 0.2) == expected

    def test_zero_volatility_is_discounted_intrinsic(self) -> None:
        forward = forward_price(100.0, 0.05, 0.0, 1.0)
        expected = math.exp(-0.05) * max(forward - 100.0, 0.0)
        assert call_price(100.0, 100.0, 0.05, 0.0, 1.0, 0.0) == pytest.approx(
            expected, abs=1e-12
        )

    def test_deep_out_of_the_money_does_not_produce_nan(self) -> None:
        """The Hart clipping exists so this returns zero rather than NaN."""
        value = call_price(1.0, 1000.0, 0.05, 0.0, 0.01, 0.05)
        assert value == 0.0

    def test_dividend_yield_lowers_a_call(self) -> None:
        without = call_price(100.0, 100.0, 0.05, 0.00, 1.0, 0.2)
        with_yield = call_price(100.0, 100.0, 0.05, 0.03, 1.0, 0.2)
        assert with_yield < without


class TestPutCallParity:
    """C - P = S*exp(-q*tau) - K*exp(-r*tau).

    The put is computed directly rather than derived from the call, so parity is a
    genuine check on two independent expressions.
    """

    @pytest.mark.parametrize(
        ("s", "k", "r", "q", "t", "v"),
        [
            (100.0, 100.0, 0.05, 0.00, 1.00, 0.20),
            (80.0, 120.0, 0.02, 0.01, 2.50, 0.45),
            (150.0, 100.0, 0.07, 0.04, 0.25, 0.15),
            (100.0, 100.0, 0.05, 0.00, 1.00, 0.00),
            (120.0, 100.0, 0.05, 0.00, 0.00, 0.20),
        ],
    )
    def test_parity(
        self, s: float, k: float, r: float, q: float, t: float, v: float
    ) -> None:
        left = call_price(s, k, r, q, t, v) - put_price(s, k, r, q, t, v)
        right = s * math.exp(-q * t) - k * math.exp(-r * t)
        assert left == pytest.approx(right, abs=1e-10)


class TestVectorisation:
    """The array path must agree exactly with the scalar path, not merely closely."""

    def test_scalar_returns_float(self) -> None:
        assert isinstance(call_price(100.0, 100.0, 0.05, 0.0, 1.0, 0.2), float)

    def test_array_returns_array(self) -> None:
        result = call_price(np.array([90.0, 110.0]), 100.0, 0.05, 0.0, 1.0, 0.2)
        assert isinstance(result, np.ndarray)

    def test_elementwise_agreement(self) -> None:
        spots = np.linspace(50.0, 150.0, 101)
        vectorised = call_price(spots, 100.0, 0.05, 0.0, 1.0, 0.2)
        elementwise = np.array(
            [call_price(float(s), 100.0, 0.05, 0.0, 1.0, 0.2) for s in spots]
        )
        assert np.array_equal(vectorised, elementwise)

    def test_two_dimensional_broadcast(self) -> None:
        """The shape a sensitivity surface needs: spot down, volatility across."""
        spots = np.linspace(80.0, 120.0, 7).reshape(-1, 1)
        vols = np.linspace(0.1, 0.5, 5).reshape(1, -1)
        surface = call_price(spots, 100.0, 0.05, 0.0, 1.0, vols)
        assert surface.shape == (7, 5)
        assert surface[3, 2] == pytest.approx(
            call_price(float(spots[3, 0]), 100.0, 0.05, 0.0, 1.0, float(vols[0, 2]))
        )

    def test_degenerate_cells_within_an_array(self) -> None:
        """A grid containing an expiring column must not poison its neighbours."""
        maturities = np.array([0.0, 0.5, 1.0])
        values = call_price(120.0, 100.0, 0.05, 0.0, maturities, 0.2)
        assert values[0] == 20.0
        assert np.all(np.isfinite(values))


class TestPriceEntryPoint:
    def test_call(self) -> None:
        contract = Contract(100.0, 1.0, OptionKind.CALL)
        market = Market(100.0, 0.05, 0.2)
        assert price(contract, market) == pytest.approx(10.450583572185565, abs=1e-12)

    def test_put(self) -> None:
        contract = Contract(100.0, 1.0, OptionKind.PUT)
        market = Market(100.0, 0.05, 0.2)
        assert price(contract, market) == pytest.approx(5.573526022256971, abs=1e-12)

    def test_returns_plain_float(self) -> None:
        value = price(Contract(100.0, 1.0, OptionKind.CALL), Market(100.0, 0.05, 0.2))
        assert type(value) is float

    def test_dispatches_on_kind(self) -> None:
        market = Market(100.0, 0.05, 0.2)
        call = price(Contract(100.0, 1.0, OptionKind.CALL), market)
        put = price(Contract(100.0, 1.0, OptionKind.PUT), market)
        assert call != put

    def test_cdf_is_used_consistently(self) -> None:
        """Guard against the price and d1_d2 drifting apart."""
        d1, d2 = d1_d2(100.0, 100.0, 0.05, 0.0, 1.0, 0.2)
        manual = 100.0 * cdf(d1) - 100.0 * math.exp(-0.05) * cdf(d2)
        assert manual == pytest.approx(
            call_price(100.0, 100.0, 0.05, 0.0, 1.0, 0.2), abs=1e-14
        )
