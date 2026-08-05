"""Brent's method for finding a root of a scalar function.

Implied volatility is an inverse problem: the price is a known, strictly increasing
function of volatility, and the task is to run it backwards. That makes a bracketing
root-finder the right tool, and Brent's method the right root-finder — it keeps the
guaranteed convergence of bisection while taking the fast steps of inverse quadratic
interpolation wherever those are safe.

The method is implemented here rather than imported from
:func:`scipy.optimize.brentq` so that the package carries no SciPy dependency. SciPy is
still used, as a *test* oracle: :mod:`tests.unit.test_roots` runs both implementations
over randomised problems and asserts they agree to machine precision.

Why not Newton's method, which needs no bracket? Because vega vanishes for deep
out-of-the-money options, and Newton divides by it. A bracketing method cannot diverge,
and the bracket for implied volatility is available for free from the no-arbitrage
bounds in :mod:`bsm.bounds`.
"""

from __future__ import annotations

from collections.abc import Callable

__all__ = ["BracketError", "ConvergenceError", "RootFindingError", "brentq"]


def _opposite_signs(x: float, y: float) -> bool:
    """Report whether two nonzero values straddle zero.

    Compares signs directly rather than testing ``x * y < 0``. The product formulation
    is the textbook one and is wrong in floating point: two values of order 1e-210 have
    a product of order 1e-420, which underflows to zero, so a genuine sign change reads
    as no sign change. Residuals that small arise routinely when inverting the price of
    a deep out-of-the-money option.

    Parameters
    ----------
    x, y
        Values to compare. Neither should be zero; callers handle exact roots before
        reaching here.

    Returns
    -------
    bool
        True if exactly one of the two is positive.
    """
    return (x > 0.0) != (y > 0.0)


def _interpolate(
    a: float, b: float, c: float, fa: float, fb: float, fc: float
) -> float | None:
    """Propose a step by inverse quadratic interpolation, or a secant step.

    Returns ``None`` when neither is usable, which tells the caller to bisect.

    The guards are not decorative. Each inverse-quadratic denominator is a *product* of
    two residual differences, so when the residuals span a wide dynamic range — one of
    order 1e-134 and another of order 1e-183, which is ordinary when inverting the price
    of a deep out-of-the-money option — the product underflows to zero even though
    neither factor is zero. The textbook formulation divides by it anyway.

    Parameters
    ----------
    a, b, c
        The three most recent abscissae, with ``b`` the current best estimate.
    fa, fb, fc
        Their residuals.

    Returns
    -------
    float or None
        A proposal, or ``None`` if every interpolation is degenerate. The proposal is
        not guaranteed finite: an overflowing step is left to the caller, whose
        containment test rejects infinities and NaN along with any other value outside
        the bracket, and bisects instead.
    """
    if fa != fc and fb != fc:
        first = (fa - fb) * (fa - fc)
        second = (fb - fa) * (fb - fc)
        third = (fc - fa) * (fc - fb)
        if first != 0.0 and second != 0.0 and third != 0.0:
            return a * fb * fc / first + b * fa * fc / second + c * fa * fb / third

    if fb != fa:
        return b - fb * (b - a) / (fb - fa)

    return None


class RootFindingError(ValueError):
    """Base class for failures in :func:`brentq`."""


class BracketError(RootFindingError):
    """The supplied interval does not bracket a root.

    Raised when the function takes the same sign at both endpoints. Brent's method
    cannot proceed, and no amount of iteration would help: the guarantee it offers is
    conditional on a sign change existing in the interval.
    """


class ConvergenceError(RootFindingError):
    """The iteration limit was reached before the tolerance was met."""


def brentq(
    f: Callable[[float], float],
    lower: float,
    upper: float,
    *,
    xtol: float = 2e-12,
    rtol: float = 8.881784197001252e-16,
    max_iterations: int = 100,
) -> float:
    """Find a root of ``f`` in ``[lower, upper]`` by Brent's method.

    Parameters
    ----------
    f
        Continuous function of one variable. Must take opposite signs at the two
        endpoints, or a root must sit exactly on one of them.
    lower, upper
        Endpoints of the bracketing interval.
    xtol
        Absolute tolerance on the root. Must be positive.
    rtol
        Relative tolerance. The default is four times machine epsilon, which is the
        smallest value that is meaningful in double precision.
    max_iterations
        Iteration cap. Brent's method converges superlinearly in practice and is
        bounded below by bisection, so reaching this limit indicates a pathological
        function rather than a slow one.

    Returns
    -------
    float
        A point ``x`` with ``f(x)`` approximately zero, located to within
        ``xtol + rtol * abs(x)``.

    Raises
    ------
    BracketError
        If ``f(lower)`` and ``f(upper)`` share a sign.
    ConvergenceError
        If ``max_iterations`` is exhausted.

    Notes
    -----
    The algorithm maintains a bracket ``[a, b]`` with ``b`` the better estimate, and on
    each step proposes either inverse quadratic interpolation through three points or a
    secant step through two. The proposal is rejected in favour of bisection whenever
    it falls outside the bracket or fails to make sufficient progress, which is what
    preserves the worst-case guarantee while allowing the fast case to be fast.

    Examples
    --------
    A root of a cubic:

    >>> round(brentq(lambda x: x**3 - 2 * x - 5, 2.0, 3.0), 12)
    2.094551481542

    The square root of two, to machine precision:

    >>> import math
    >>> abs(brentq(lambda x: x * x - 2.0, 0.0, 2.0) - math.sqrt(2.0)) < 1e-15
    True

    A root sitting exactly on an endpoint is returned without iterating:

    >>> brentq(lambda x: x - 1.0, 1.0, 5.0)
    1.0

    Residuals far below the underflow threshold of their own product are handled, which
    a ``f(a) * f(b) < 0`` sign test would not be:

    >>> root = brentq(lambda x: 1e-210 * (x - 2.0), 0.0, 5.0)
    >>> abs(root - 2.0) < 1e-9
    True

    An interval with no sign change is rejected rather than searched:

    >>> brentq(lambda x: x * x + 1.0, -1.0, 1.0)
    Traceback (most recent call last):
        ...
    bsm.roots.BracketError: f(-1.0) = 2 and f(1.0) = 2 have the same sign
    """
    if xtol <= 0.0:
        raise ValueError(f"xtol must be positive, got {xtol!r}")

    a, b = lower, upper
    fa, fb = f(a), f(b)

    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if not _opposite_signs(fa, fb):
        raise BracketError(
            f"f({a!r}) = {fa:g} and f({b!r}) = {fb:g} have the same sign"
        )

    # Keep b as the endpoint with the smaller residual: it is the current best estimate
    # and the value that will eventually be returned.
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c, fc = a, fa
    d = b - a
    bisected = True

    for _ in range(max_iterations):
        tolerance = xtol + rtol * abs(b)
        if fb == 0.0 or abs(b - a) < tolerance:
            return b

        interpolated = _interpolate(a, b, c, fa, fb, fc)

        # Reject the proposal unless it lies in the outer three quarters of the bracket
        # and is at least halving the step. Each of these conditions, if violated,
        # admits a sequence that stalls; bisection is the safe alternative.
        low, high = sorted(((3.0 * a + b) / 4.0, b))
        if (
            interpolated is None
            or not low < interpolated < high
            or (bisected and abs(interpolated - b) >= abs(b - c) / 2.0)
            or (not bisected and abs(interpolated - b) >= abs(c - d) / 2.0)
            or (bisected and abs(b - c) < tolerance)
            or (not bisected and abs(c - d) < tolerance)
        ):
            proposal = 0.5 * (a + b)
            bisected = True
        else:
            proposal = interpolated
            bisected = False

        f_proposal = f(proposal)
        if f_proposal == 0.0:
            return proposal

        d, c, fc = c, b, fb

        if _opposite_signs(fa, f_proposal):
            b, fb = proposal, f_proposal
        else:
            a, fa = proposal, f_proposal

        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

    raise ConvergenceError(
        f"no root located to within {xtol!r} after {max_iterations} iterations; "
        f"best estimate {b!r} has residual {fb!r}"
    )
