r"""Standard normal density and cumulative distribution function.

Black-Scholes needs :math:`\Phi`, the standard normal CDF, and nothing else from
statistics. SciPy provides it as ``scipy.special.ndtr``, but SciPy is a large
dependency to carry for one function, and it is particularly expensive under Pyodide
where the published document loads the package into the reader's browser. So it is
implemented here.

The implementation is Hart's algorithm, in the form given by Graeme West, *Better
Approximations to Cumulative Normal Functions* (Wilmott Magazine, 2005). It evaluates
the upper tail :math:`\Phi(-|x|)` by one of two routes depending on magnitude: a ratio
of polynomials of degree 6 and 7 near the centre, and a continued-fraction expansion of
the Mills ratio in the tail. Both are pure arithmetic, so both vectorise over NumPy
arrays with no Python-level loop.

Why not simply wrap :func:`math.erfc`? Because it is scalar-only, so filling a grid
would mean a Python loop over every cell. Instead ``math.erfc`` serves as the *test
oracle*: :mod:`tests.unit.test_normal` asserts that this implementation agrees with it
to within a documented tolerance across the whole range. The fast path is the shipped
one; the stdlib is what proves it right.

Notes
-----
Arguments are clipped to magnitude 40 before evaluation. This is not a shortcut but a
correctness requirement: unclipped, ``exp(-inf) * polynomial(inf)`` evaluates to
``0 * inf``, which is NaN. Clipping makes :math:`\Phi(\pm\infty)` return exactly 1
and 0, which the pricing formulas rely on for their zero-volatility and zero-maturity
limits. Nothing is lost numerically: :math:`\Phi(-40)` is below the smallest normal
double, so every clipped value already rounds to zero.
"""

from __future__ import annotations

import math
from typing import overload

import numpy as np

from bsm._numeric import FloatArray, Numeric, as_array, scalarize

__all__ = ["cdf", "pdf"]

_SQRT_2PI: float = math.sqrt(2.0 * math.pi)

# Split between the polynomial-ratio and continued-fraction regimes, at 5*sqrt(2).
_TAIL_SPLIT: float = 7.07106781186547

# Beyond this magnitude the upper tail is below the smallest normal double.
_CLIP: float = 40.0

# Hart coefficients, highest degree first.
_NUMERATOR: tuple[float, ...] = (
    0.0352624965998911,
    0.700383064443688,
    6.37396220353165,
    33.912866078383,
    112.079291497871,
    221.213596169931,
    220.206867912376,
)
_DENOMINATOR: tuple[float, ...] = (
    0.0883883476483184,
    1.75566716318264,
    16.064177579207,
    86.7807322029461,
    296.564248779674,
    637.333633378831,
    793.826512519948,
    440.413735824752,
)


def _horner(coefficients: tuple[float, ...], x: FloatArray) -> FloatArray:
    """Evaluate a polynomial by Horner's method.

    Parameters
    ----------
    coefficients
        Coefficients in descending order of degree.
    x
        Points of evaluation.

    Returns
    -------
    FloatArray
        The polynomial evaluated at ``x``.
    """
    result = np.full_like(x, coefficients[0])
    for coefficient in coefficients[1:]:
        result = result * x + coefficient
    return result


def _upper_tail(x_abs: FloatArray) -> FloatArray:
    r"""Evaluate :math:`\Phi(-|x|)` for non-negative ``x_abs``.

    Both regimes are computed for every element and selected between with
    :func:`numpy.where`. Evaluating the unused branch is harmless — neither can divide
    by zero on non-negative input — and it keeps the function branch-free and therefore
    vectorised.

    Parameters
    ----------
    x_abs
        Non-negative magnitudes, already clipped to ``_CLIP``.

    Returns
    -------
    FloatArray
        Upper tail probabilities in :math:`[0, 1/2]`.
    """
    density = np.exp(-0.5 * x_abs * x_abs)

    polynomial = density * _horner(_NUMERATOR, x_abs) / _horner(_DENOMINATOR, x_abs)

    # Continued fraction for the Mills ratio: x + 1/(x + 2/(x + 3/(x + 4/(x + 0.65)))).
    fraction = x_abs + 0.65
    for term in (4.0, 3.0, 2.0, 1.0):
        fraction = x_abs + term / fraction
    tail = density / (fraction * _SQRT_2PI)

    return np.where(x_abs < _TAIL_SPLIT, polynomial, tail)


@overload
def cdf(x: float) -> float: ...


@overload
def cdf(x: FloatArray) -> FloatArray: ...


def cdf(x: Numeric) -> Numeric:
    r"""Evaluate the standard normal cumulative distribution function.

    Parameters
    ----------
    x
        Scalar or array of quantiles.

    Returns
    -------
    Numeric
        :math:`\Phi(x) = \mathbb{P}(Z \le x)` for :math:`Z` standard normal, with
        the same shape as ``x``.

    Examples
    --------
    >>> cdf(0.0)
    0.5

    Symmetry holds to machine precision:

    >>> cdf(1.5) + cdf(-1.5)
    1.0

    The one-sigma and two-sigma probabilities are the familiar ones:

    >>> round(cdf(1.0) - cdf(-1.0), 6)
    0.682689
    >>> round(cdf(2.0) - cdf(-2.0), 6)
    0.9545

    Arrays are returned as arrays:

    >>> cdf(np.array([-1.0, 0.0, 1.0])).round(6)
    array([0.158655, 0.5     , 0.841345])

    Infinities give exact limits, which the pricing formulas depend on:

    >>> cdf(float("inf")), cdf(float("-inf"))
    (1.0, 0.0)
    """
    values = as_array(x)
    magnitude = np.clip(np.abs(values), 0.0, _CLIP)
    tail = _upper_tail(magnitude)
    result = np.where(values > 0.0, 1.0 - tail, tail)
    return scalarize(result, x)


@overload
def pdf(x: float) -> float: ...


@overload
def pdf(x: FloatArray) -> FloatArray: ...


def pdf(x: Numeric) -> Numeric:
    r"""Evaluate the standard normal probability density function.

    Parameters
    ----------
    x
        Scalar or array of points.

    Returns
    -------
    Numeric
        :math:`\varphi(x) = e^{-x^2/2} / \sqrt{2\pi}`, with the same shape as ``x``.

    Notes
    -----
    Needed by gamma, vega, and theta in :mod:`bsm.greeks`, and exported because the
    density is worth plotting alongside the distribution in the accompanying document.

    Examples
    --------
    >>> round(pdf(0.0), 6)
    0.398942
    >>> round(pdf(0.0) - 1 / math.sqrt(2 * math.pi), 15)
    0.0
    >>> pdf(float("inf"))
    0.0
    """
    values = as_array(x)
    magnitude = np.clip(np.abs(values), 0.0, _CLIP)
    result = np.exp(-0.5 * magnitude * magnitude) / _SQRT_2PI
    return scalarize(result, x)
