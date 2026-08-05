"""Property-based tests: invariants that must hold for all admissible inputs.

These are the tests that catch a formula which is right on the textbook cases and wrong
in a corner. Every strategy below generates only inputs the functions declare as valid
preconditions; behaviour outside them is undefined by design.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from bsm.pricing import call_price, forward_price, put_price

positive = st.floats(min_value=1.0, max_value=1_000.0, allow_nan=False)
rates = st.floats(min_value=-0.05, max_value=0.25, allow_nan=False)
yields = st.floats(min_value=0.0, max_value=0.15, allow_nan=False)
maturities = st.floats(min_value=0.0, max_value=10.0, allow_nan=False)
volatilities = st.floats(min_value=0.0, max_value=3.0, allow_nan=False)

market_inputs = st.tuples(positive, positive, rates, yields, maturities, volatilities)

Inputs = tuple[float, float, float, float, float, float]


@given(market_inputs)
def test_put_call_parity(args: Inputs) -> None:
    s, k, r, q, t, v = args
    left = call_price(*args) - put_price(*args)
    right = s * math.exp(-q * t) - k * math.exp(-r * t)
    assert left == pytest.approx(right, abs=1e-8, rel=1e-10)


@given(market_inputs)
def test_prices_are_non_negative(args: Inputs) -> None:
    assert call_price(*args) >= 0.0
    assert put_price(*args) >= 0.0


@given(market_inputs)
def test_no_arbitrage_bounds_recomputed_independently(args: Inputs) -> None:
    """The bounds hold, with the bound recomputed here from `math.exp`.

    The allowance is a few units in the last place and nothing more. It is needed not
    because the pricer strays outside its bounds — it clips into them, and
    `test_bounds.py` asserts that with no tolerance at all against the package's own
    `price_bounds` — but because `math.exp` and `numpy.exp` are each correctly rounded
    to within an ulp and need not agree on which way. Recomputing the bound here rather
    than importing it is the point of this test; the ulp allowance is the price of that
    independence.
    """
    s, k, r, q, t, v = args
    discounted_spot = s * math.exp(-q * t)
    discounted_strike = k * math.exp(-r * t)
    slack = 4.0 * sys.float_info.epsilon * max(discounted_spot, discounted_strike)

    call = call_price(*args)
    assert call >= max(discounted_spot - discounted_strike, 0.0) - slack
    assert call <= discounted_spot + slack

    put = put_price(*args)
    assert put >= max(discounted_strike - discounted_spot, 0.0) - slack
    assert put <= discounted_strike + slack


@given(market_inputs, st.floats(min_value=1.01, max_value=3.0))
def test_call_increases_with_spot(args: Inputs, factor: float) -> None:
    s, k, r, q, t, v = args
    lower = call_price(s, k, r, q, t, v)
    upper = call_price(s * factor, k, r, q, t, v)
    assert upper >= lower - 1e-12


@given(market_inputs, st.floats(min_value=1.01, max_value=3.0))
def test_put_decreases_with_spot(args: Inputs, factor: float) -> None:
    s, k, r, q, t, v = args
    lower = put_price(s, k, r, q, t, v)
    upper = put_price(s * factor, k, r, q, t, v)
    assert upper <= lower + 1e-12


@given(market_inputs, st.floats(min_value=0.001, max_value=2.0))
def test_prices_increase_with_volatility(args: Inputs, extra: float) -> None:
    """Vega is positive. This is what makes implied volatility well posed."""
    s, k, r, q, t, v = args
    assume(t > 1e-6)
    assert call_price(s, k, r, q, t, v + extra) >= call_price(*args) - 1e-12
    assert put_price(s, k, r, q, t, v + extra) >= put_price(*args) - 1e-12


@given(market_inputs)
def test_call_decreases_with_strike(args: Inputs) -> None:
    s, k, r, q, t, v = args
    assert call_price(s, k * 1.5, r, q, t, v) <= call_price(*args) + 1e-12


@given(market_inputs)
def test_expiry_gives_payoff(args: Inputs) -> None:
    s, k, r, q, _, v = args
    assert call_price(s, k, r, q, 0.0, v) == pytest.approx(max(s - k, 0.0), abs=1e-12)
    assert put_price(s, k, r, q, 0.0, v) == pytest.approx(max(k - s, 0.0), abs=1e-12)


@given(market_inputs)
def test_zero_volatility_is_discounted_forward_intrinsic(args: Inputs) -> None:
    s, k, r, q, t, _ = args
    forward = forward_price(s, r, q, t)
    expected = math.exp(-r * t) * max(forward - k, 0.0)
    assert call_price(s, k, r, q, t, 0.0) == pytest.approx(
        expected, abs=1e-8, rel=1e-10
    )


@given(market_inputs)
def test_scalar_and_array_paths_agree_exactly(args: Inputs) -> None:
    """Bitwise equality, not approximate: it is literally the same code path."""
    scalar = call_price(*args)
    array = call_price(*(np.array([value]) for value in args))
    assert array[0] == scalar


@settings(max_examples=50)
@given(market_inputs)
def test_prices_are_finite(args: Inputs) -> None:
    assert math.isfinite(call_price(*args))
    assert math.isfinite(put_price(*args))
