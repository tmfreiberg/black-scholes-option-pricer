"""Tests for the standard normal density and distribution.

The central test is `test_matches_stdlib_oracle`. The shipped implementation is Hart's
algorithm, chosen because it vectorises; `math.erfc` is exact to within an ulp but
scalar-only. Pinning one against the other is what licenses the speed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from bsm.normal import cdf, pdf


def erfc_cdf(x: float) -> float:
    """Reference CDF from the standard library.

    Uses the complementary error function rather than ``0.5 * (1 + erf(x / sqrt(2)))``
    because the latter loses all significance in the left tail: for x = -8 it computes
    ``0.5 * (1 - 0.9999999999999994)``, where catastrophic cancellation destroys the
    result. ``erfc`` returns the small quantity directly.
    """
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


# Tolerance is set from the measured maximum error over the sampled range, not chosen
# in advance. See test_documented_tolerance_is_not_slack.
ABSOLUTE_TOLERANCE = 1e-15


class TestCdfAgainstOracle:
    """Agreement between Hart's algorithm and the stdlib reference."""

    @pytest.mark.parametrize(
        "x",
        [
            -40.0,
            -20.0,
            -10.0,
            -8.0,
            -7.08,
            -7.07,
            -7.0,
            -5.0,
            -3.0,
            -1.0,
            -0.5,
            -1e-8,
            0.0,
            1e-8,
            0.5,
            1.0,
            3.0,
            5.0,
            7.0,
            7.07,
            7.08,
            8.0,
            10.0,
            20.0,
            40.0,
        ],
    )
    def test_matches_stdlib_oracle(self, x: float) -> None:
        assert cdf(x) == pytest.approx(erfc_cdf(x), abs=ABSOLUTE_TOLERANCE)

    @given(st.floats(min_value=-40.0, max_value=40.0, allow_nan=False))
    def test_matches_stdlib_oracle_everywhere(self, x: float) -> None:
        assert cdf(x) == pytest.approx(erfc_cdf(x), abs=ABSOLUTE_TOLERANCE)

    def test_documented_tolerance_is_not_slack(self) -> None:
        """The stated tolerance should be tight, not a comfortable margin.

        If Hart's algorithm were far more accurate than claimed, the tolerance would be
        hiding headroom; if it were barely inside, a compiler or NumPy change could
        break the suite without warning. Assert the true error is within one order of
        magnitude of the documented bound.
        """
        grid = np.linspace(-40.0, 40.0, 20_001)
        computed = cdf(grid)
        reference = np.array([erfc_cdf(float(x)) for x in grid])
        worst = float(np.max(np.abs(computed - reference)))
        assert worst <= ABSOLUTE_TOLERANCE
        assert worst >= ABSOLUTE_TOLERANCE / 10.0


class TestCdfProperties:
    """Invariants that hold regardless of implementation."""

    @given(st.floats(min_value=-30.0, max_value=30.0, allow_nan=False))
    def test_symmetry(self, x: float) -> None:
        assert cdf(x) + cdf(-x) == pytest.approx(1.0, abs=1e-15)

    @given(st.floats(min_value=-40.0, max_value=40.0, allow_nan=False))
    def test_bounded(self, x: float) -> None:
        assert 0.0 <= cdf(x) <= 1.0

    def test_monotone(self) -> None:
        grid = np.linspace(-10.0, 10.0, 5_000)
        assert np.all(np.diff(cdf(grid)) >= 0.0)

    def test_median(self) -> None:
        assert cdf(0.0) == 0.5

    @pytest.mark.parametrize(
        ("x", "expected"),
        [(float("inf"), 1.0), (float("-inf"), 0.0)],
    )
    def test_infinite_limits_are_exact(self, x: float, expected: float) -> None:
        """Exactness here is a precondition of the zero-volatility pricing limit."""
        assert cdf(x) == expected

    def test_nan_propagates(self) -> None:
        assert math.isnan(cdf(float("nan")))


class TestShapePreservation:
    """Scalars in, scalars out; arrays in, arrays out."""

    def test_scalar_returns_float(self) -> None:
        assert isinstance(cdf(0.3), float)

    def test_array_returns_array(self) -> None:
        result = cdf(np.array([0.0, 1.0]))
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)

    def test_two_dimensional_shape_preserved(self) -> None:
        assert cdf(np.zeros((3, 4))).shape == (3, 4)

    def test_array_agrees_with_scalar_elementwise(self) -> None:
        """The array path must not diverge from the scalar path."""
        grid = np.linspace(-6.0, 6.0, 501)
        elementwise = np.array([cdf(float(x)) for x in grid])
        assert np.array_equal(cdf(grid), elementwise)


class TestPdf:
    """The density, needed by gamma, vega, and theta."""

    def test_peak(self) -> None:
        assert pdf(0.0) == pytest.approx(1.0 / math.sqrt(2.0 * math.pi), abs=1e-16)

    @given(st.floats(min_value=-30.0, max_value=30.0, allow_nan=False))
    def test_symmetry(self, x: float) -> None:
        assert pdf(x) == pytest.approx(pdf(-x), abs=1e-18)

    @given(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False))
    def test_is_derivative_of_cdf(self, x: float) -> None:
        """The density is the derivative of the distribution, by central difference."""
        step = 1e-5
        slope = (cdf(x + step) - cdf(x - step)) / (2.0 * step)
        assert slope == pytest.approx(pdf(x), abs=1e-8)

    def test_integrates_to_one(self) -> None:
        grid = np.linspace(-40.0, 40.0, 400_001)
        assert np.trapezoid(pdf(grid), grid) == pytest.approx(1.0, abs=1e-12)

    def test_infinite_limit_is_zero(self) -> None:
        assert pdf(float("inf")) == 0.0
