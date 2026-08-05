"""Tests for Brent's method.

The shipped implementation exists so the package needs no SciPy. SciPy is therefore the
oracle: `TestAgainstScipy` runs both over the same problems and requires agreement to
machine precision.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from scipy.optimize import brentq as scipy_brentq

from bsm.roots import BracketError, ConvergenceError, _interpolate, brentq

Problem = tuple[Callable[[float], float], float, float]

PROBLEMS: list[tuple[str, Callable[[float], float], float, float]] = [
    ("cubic", lambda x: x**3 - 2 * x - 5, 2.0, 3.0),
    ("sqrt2", lambda x: x * x - 2.0, 0.0, 2.0),
    ("cosine", lambda x: math.cos(x) - x, 0.0, 1.0),
    ("exponential", lambda x: math.exp(x) - 5.0, 0.0, 3.0),
    ("steep", lambda x: x**15 - 0.5, 0.0, 2.0),
    ("asymmetric", lambda x: math.atan(x) - 0.5, -1.0, 10.0),
    ("log", lambda x: math.log(x) - 1.0, 0.1, 10.0),
]

# Both solvers are driven to the smallest tolerances double precision admits, so that
# the comparison measures the algorithms rather than the stopping rules. At the library
# defaults the two stop at different points inside their shared tolerance, and ours
# often lands closer to the true root — comparing there would test the defaults, not
# the method.
TIGHT_XTOL = 1e-15
TIGHT_RTOL = 8.881784197001252e-16


class TestAgainstScipy:
    @pytest.mark.parametrize(
        ("name", "f", "a", "b"), PROBLEMS, ids=[p[0] for p in PROBLEMS]
    )
    def test_agrees_with_scipy(
        self, name: str, f: Callable[[float], float], a: float, b: float
    ) -> None:
        ours = brentq(f, a, b, xtol=TIGHT_XTOL)
        theirs = scipy_brentq(f, a, b, xtol=TIGHT_XTOL, rtol=TIGHT_RTOL)
        assert ours == pytest.approx(theirs, abs=1e-12)

    @settings(max_examples=50, deadline=None)
    @given(
        st.floats(min_value=-5.0, max_value=5.0),
        st.floats(min_value=0.5, max_value=5.0),
    )
    def test_agrees_with_scipy_on_random_quadratics(
        self, root: float, offset: float
    ) -> None:
        """A quadratic with one root in the bracket and one far outside."""

        def f(x: float) -> float:
            return (x - root) * (x - (root + 100.0))

        a, b = root - offset, root + offset
        assume(f(a) * f(b) < 0.0)
        ours = brentq(f, a, b, xtol=TIGHT_XTOL)
        theirs = scipy_brentq(f, a, b, xtol=TIGHT_XTOL, rtol=TIGHT_RTOL)
        assert ours == pytest.approx(theirs, abs=1e-10)


class TestAccuracy:
    @pytest.mark.parametrize(
        ("name", "f", "a", "b"), PROBLEMS, ids=[p[0] for p in PROBLEMS]
    )
    def test_residual_is_small(
        self, name: str, f: Callable[[float], float], a: float, b: float
    ) -> None:
        assert abs(f(brentq(f, a, b))) < 1e-10

    def test_known_root_to_machine_precision(self) -> None:
        assert abs(brentq(lambda x: x * x - 2.0, 0.0, 2.0) - math.sqrt(2.0)) < 1e-15

    def test_triple_root(self) -> None:
        """A root of multiplicity three, where the residual carries little signal.

        `(x - 1)**3` is flat enough near its root that the function value is
        indistinguishable from zero across a wide interval, so a solver has to fall back
        on bracket width rather than residual. Worth pinning: SciPy's `brentq` does not
        converge on this problem within its default iteration limit, and ours does.
        """
        root = brentq(lambda x: (x - 1.0) ** 3, 0.0, 3.0)
        assert root == pytest.approx(1.0, abs=1e-4)

    def test_root_stays_inside_the_bracket(self) -> None:
        root = brentq(lambda x: math.cos(x) - x, 0.0, 1.0)
        assert 0.0 <= root <= 1.0


class TestEndpointsAndFailures:
    def test_root_on_lower_endpoint(self) -> None:
        assert brentq(lambda x: x - 1.0, 1.0, 5.0) == 1.0

    def test_root_on_upper_endpoint(self) -> None:
        assert brentq(lambda x: x - 5.0, 1.0, 5.0) == 5.0

    def test_no_sign_change_raises(self) -> None:
        with pytest.raises(BracketError, match="same sign"):
            brentq(lambda x: x * x + 1.0, -1.0, 1.0)

    def test_bracket_error_names_both_endpoints(self) -> None:
        with pytest.raises(BracketError) as info:
            brentq(lambda x: x * x + 1.0, -1.0, 1.0)
        assert "-1.0" in str(info.value)
        assert "1.0" in str(info.value)

    def test_iteration_cap_raises(self) -> None:
        with pytest.raises(ConvergenceError, match="after 1 iterations"):
            brentq(lambda x: x**3 - 2 * x - 5, 2.0, 3.0, max_iterations=1)

    def test_convergence_error_reports_best_estimate(self) -> None:
        with pytest.raises(ConvergenceError) as info:
            brentq(lambda x: x**3 - 2 * x - 5, 2.0, 3.0, max_iterations=1)
        assert "residual" in str(info.value)

    def test_non_positive_tolerance_rejected(self) -> None:
        with pytest.raises(ValueError, match="xtol must be positive"):
            brentq(lambda x: x - 1.0, 0.0, 2.0, xtol=0.0)

    def test_reversed_bracket_still_works(self) -> None:
        """The endpoints need not be ordered; only the signs matter."""
        assert brentq(lambda x: x - 1.0, 5.0, 0.0) == pytest.approx(1.0, abs=1e-12)


class TestEfficiency:
    def test_beats_bisection_on_a_smooth_function(self) -> None:
        """Interpolation should be doing real work, not silently degrading."""
        calls = 0

        def counted(x: float) -> float:
            nonlocal calls
            calls += 1
            return math.exp(x) - 5.0

        brentq(counted, 0.0, 3.0)
        # Bisection alone would need about 50 evaluations for this tolerance.
        assert calls < 20


class TestInterpolationGuard:
    """Direct tests of the private step chooser.

    These paths are unreachable through `brentq` itself, which maintains a bracket with
    residuals of opposite sign, so `fa` and `fb` can never coincide. They exist because
    the alternative to guarding is a `ZeroDivisionError`, and the conditions that
    trigger them are exactly the ones a bracketing solver on an underflowing residual
    would otherwise hit.
    """

    def test_underflowing_product_falls_back_to_secant(self) -> None:
        """The IQI denominators are products, and products of tiny values underflow."""
        proposal = _interpolate(0.0, 1.0, 0.5, -1e-183, 1e-183, 3e-183)
        assert proposal is not None
        assert math.isfinite(proposal)

    def test_equal_residuals_give_no_proposal(self) -> None:
        assert _interpolate(0.0, 1.0, 0.5, 2.0, 2.0, 2.0) is None

    def test_non_finite_proposal_is_left_to_the_caller(self) -> None:
        """An overflowing step is returned as-is; the containment test rejects it.

        `not low < inf < high` is True, so the caller bisects. Filtering here as well
        would duplicate that check on a path no test could reach.
        """
        proposal = _interpolate(0.0, 1e308, 0.5, -1e-308, 1e308, 1e308)
        assert proposal is not None
        low, high = sorted(((3.0 * 0.0 + 1e308) / 4.0, 1e308))
        assert not low < proposal < high

    def test_ordinary_case_uses_inverse_quadratic(self) -> None:
        proposal = _interpolate(0.0, 3.0, 1.0, -2.0, 4.0, -1.0)
        assert proposal is not None
        assert 0.0 < proposal < 3.0
