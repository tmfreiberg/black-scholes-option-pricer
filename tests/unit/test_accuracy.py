"""Absolute accuracy of the pricing pipeline, against a 50-digit reference.

The CDF is checked against `math.erfc` in test_normal; that bounds the error in a
probability. This module bounds the error in a *price*, which is the quantity anyone
actually uses, and which amplifies the CDF error by roughly the spot level.

mpmath evaluates the Black-Scholes formula in 50-digit arithmetic, so the reference is
exact for these purposes and the measured error is genuinely ours.
"""

from __future__ import annotations

import mpmath as mp
import pytest

from bsm.pricing import call_price, put_price
from bsm.types import OptionKind

# Measured, not chosen: the worst error observed across the cases below is about
# 2.4e-14 at spot 100. Stated as an absolute bound at that scale; it grows roughly in
# proportion to spot, because it is the CDF's ~1e-15 error multiplied through.
ABSOLUTE_TOLERANCE_AT_UNIT_SPOT = 1e-15


def reference_price(
    kind: OptionKind, s: float, k: float, r: float, q: float, t: float, v: float
) -> float:
    """Black-Scholes-Merton price in 50-digit arithmetic."""
    with mp.workdps(50):
        S, K, R, Q, T, V = (mp.mpf(str(x)) for x in (s, k, r, q, t, v))
        total = V * mp.sqrt(T)
        d1 = (mp.log(S / K) + (R - Q + V**2 / 2) * T) / total
        d2 = d1 - total

        def phi(x: mp.mpf) -> mp.mpf:
            return mp.erfc(-x / mp.sqrt(2)) / 2

        if kind is OptionKind.CALL:
            value = S * mp.exp(-Q * T) * phi(d1) - K * mp.exp(-R * T) * phi(d2)
        else:
            value = K * mp.exp(-R * T) * phi(-d2) - S * mp.exp(-Q * T) * phi(-d1)
        return float(value)


CASES = [
    (100.0, 100.0, 0.05, 0.00, 1.00, 0.20),
    (42.0, 40.0, 0.10, 0.00, 0.50, 0.20),
    (930.0, 900.0, 0.08, 0.03, 0.25, 0.20),
    (100.0, 150.0, 0.01, 0.02, 3.00, 0.55),
    (100.0, 60.0, -0.01, 0.00, 0.10, 0.15),
    (1.0, 1.0, 0.03, 0.00, 2.00, 0.80),
    (5000.0, 4800.0, 0.04, 0.015, 0.75, 0.18),
]


@pytest.mark.parametrize("kind", [OptionKind.CALL, OptionKind.PUT])
@pytest.mark.parametrize(("s", "k", "r", "q", "t", "v"), CASES)
def test_price_matches_high_precision_reference(
    kind: OptionKind, s: float, k: float, r: float, q: float, t: float, v: float
) -> None:
    formula = call_price if kind is OptionKind.CALL else put_price
    computed = float(formula(s, k, r, q, t, v))
    exact = reference_price(kind, s, k, r, q, t, v)
    assert abs(computed - exact) <= ABSOLUTE_TOLERANCE_AT_UNIT_SPOT * s


def test_stated_tolerance_is_a_measurement() -> None:
    """The bound should be tight, not a comfortable margin hiding a regression."""
    worst = 0.0
    for kind in (OptionKind.CALL, OptionKind.PUT):
        formula = call_price if kind is OptionKind.CALL else put_price
        for s, k, r, q, t, v in CASES:
            computed = float(formula(s, k, r, q, t, v))
            exact = reference_price(kind, s, k, r, q, t, v)
            worst = max(worst, abs(computed - exact) / s)
    assert worst <= ABSOLUTE_TOLERANCE_AT_UNIT_SPOT
    assert worst >= ABSOLUTE_TOLERANCE_AT_UNIT_SPOT / 100.0
