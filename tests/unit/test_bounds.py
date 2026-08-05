"""Tests for the model-free no-arbitrage relationships."""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from bsm.bounds import parity_residual, price_bounds
from bsm.pricing import call_price, put_price
from bsm.types import OptionKind

positive = st.floats(min_value=1.0, max_value=1_000.0, allow_nan=False)
rates = st.floats(min_value=-0.05, max_value=0.25, allow_nan=False)
yields = st.floats(min_value=0.0, max_value=0.15, allow_nan=False)
maturities = st.floats(min_value=0.0, max_value=10.0, allow_nan=False)
volatilities = st.floats(min_value=0.0, max_value=3.0, allow_nan=False)
market_inputs = st.tuples(positive, positive, rates, yields, maturities, volatilities)

Inputs = tuple[float, float, float, float, float, float]


class TestBounds:
    def test_call_bounds_at_the_money(self) -> None:
        lower, upper = price_bounds(OptionKind.CALL, 100.0, 100.0, 0.05, 0.0, 1.0)
        assert lower == pytest.approx(100.0 - 100.0 * math.exp(-0.05))
        assert upper == 100.0

    def test_put_bounds_at_the_money(self) -> None:
        lower, upper = price_bounds(OptionKind.PUT, 100.0, 100.0, 0.05, 0.0, 1.0)
        assert lower == 0.0
        assert upper == pytest.approx(100.0 * math.exp(-0.05))

    def test_bounds_pinch_shut_at_expiry(self) -> None:
        lower, _ = price_bounds(OptionKind.CALL, 120.0, 100.0, 0.05, 0.0, 0.0)
        assert lower == 20.0

    def test_lower_bound_is_attained_at_zero_volatility(self) -> None:
        lower, _ = price_bounds(OptionKind.CALL, 100.0, 100.0, 0.05, 0.0, 1.0)
        assert call_price(100.0, 100.0, 0.05, 0.0, 1.0, 0.0) == pytest.approx(
            lower, abs=1e-12
        )

    def test_upper_bound_is_approached_from_below(self) -> None:
        _, upper = price_bounds(OptionKind.CALL, 100.0, 100.0, 0.05, 0.0, 1.0)
        assert call_price(100.0, 100.0, 0.05, 0.0, 1.0, 5.0) < upper

    def test_upper_bound_is_the_limit(self) -> None:
        """At large volatility the price reaches the bound to within rounding.

        The bound is a supremum rather than a maximum, but double precision cannot see
        the difference once volatility is large enough, so this asserts convergence
        rather than strict inequality.
        """
        _, upper = price_bounds(OptionKind.CALL, 100.0, 100.0, 0.05, 0.0, 1.0)
        assert call_price(100.0, 100.0, 0.05, 0.0, 1.0, 50.0) == pytest.approx(
            upper, rel=1e-12
        )

    @given(market_inputs)
    def test_black_scholes_prices_respect_the_bounds(self, args: Inputs) -> None:
        """Exactly, not approximately: the pricer clips into these bounds."""
        s, k, r, q, t, v = args

        call_lower, call_upper = price_bounds(OptionKind.CALL, s, k, r, q, t)
        call = float(call_price(*args))
        assert float(call_lower) <= call <= float(call_upper)

        put_lower, put_upper = price_bounds(OptionKind.PUT, s, k, r, q, t)
        put = float(put_price(*args))
        assert float(put_lower) <= put <= float(put_upper)


class TestParity:
    def test_black_scholes_satisfies_parity(self) -> None:
        args = (100.0, 100.0, 0.05, 0.0, 1.0)
        residual = parity_residual(call_price(*args, 0.2), put_price(*args, 0.2), *args)
        assert abs(float(residual)) < 1e-12

    @given(market_inputs)
    def test_parity_holds_everywhere(self, args: Inputs) -> None:
        s, k, r, q, t, v = args
        residual = parity_residual(call_price(*args), put_price(*args), s, k, r, q, t)
        assert abs(float(residual)) < 1e-8

    def test_residual_is_signed_and_scaled_like_a_price(self) -> None:
        """An overpriced call gives a positive residual of the size of the error."""
        args = (100.0, 100.0, 0.05, 0.0, 1.0)
        fair_put = float(put_price(*args, 0.2))
        fair_call = float(call_price(*args, 0.2))
        assert parity_residual(fair_call + 0.25, fair_put, *args) == pytest.approx(
            0.25, abs=1e-10
        )
