r"""No-arbitrage bounds and put-call parity.

These are the relationships an option price must satisfy whatever model produced it.
They are model-free: each follows from a portfolio that would otherwise offer a
riskless profit, so they hold for Black-Scholes prices, for market prices, and for
prices from any model worth taking seriously.

They serve two purposes here. As tests, they are the strongest available check on the
pricing formulas, because they are derived from arbitrage rather than from the formula
being checked. As code, they supply the bracket for implied volatility: the price is
increasing in :math:`\sigma` from the lower bound at :math:`\sigma = 0` toward the
upper bound as :math:`\sigma \to \infty`, so an observed price strictly inside the
bounds has exactly one implied volatility, and one outside has none.

For a European call with the usual notation,

.. math::

    \max(S e^{-q\tau} - K e^{-r\tau},\ 0) \le C \le S e^{-q\tau},

and for a put,

.. math::

    \max(K e^{-r\tau} - S e^{-q\tau},\ 0) \le P \le K e^{-r\tau}.

The lower bounds are the discounted intrinsic value on the forward, attained at zero
volatility. The upper bounds are the limits as volatility grows without bound: a call
is never worth more than the (dividend-discounted) stock, and a put never more than the
discounted strike.
"""

from __future__ import annotations

import numpy as np

from bsm._numeric import Numeric, as_array, scalarize
from bsm.types import OptionKind

__all__ = ["parity_residual", "price_bounds"]


def price_bounds(
    kind: OptionKind,
    spot: Numeric,
    strike: Numeric,
    rate: Numeric,
    dividend_yield: Numeric,
    maturity: Numeric,
) -> tuple[Numeric, Numeric]:
    r"""Return the no-arbitrage bounds on an option price.

    Parameters
    ----------
    kind
        Call or put.
    spot
        Current price of the underlying, strictly positive.
    strike
        Exercise price, strictly positive.
    rate
        Continuously compounded risk-free rate, as a decimal.
    dividend_yield
        Continuous dividend yield, as a decimal.
    maturity
        Time to expiry in years, non-negative.

    Returns
    -------
    tuple of Numeric
        ``(lower, upper)``. The lower bound is attained at zero volatility; the upper
        is approached as volatility grows without bound and is not attained.

    Examples
    --------
    An at-the-money call with a year to run:

    >>> lower, upper = price_bounds(OptionKind.CALL, 100.0, 100.0, 0.05, 0.0, 1.0)
    >>> round(lower, 6), round(upper, 6)
    (4.877058, 100.0)

    The Black-Scholes price sits strictly inside:

    >>> from bsm.pricing import call_price
    >>> lower < call_price(100.0, 100.0, 0.05, 0.0, 1.0, 0.2) < upper
    True

    At expiry the bounds pinch shut onto the payoff, which is why implied volatility is
    undefined there:

    >>> price_bounds(OptionKind.PUT, 80.0, 100.0, 0.05, 0.0, 0.0)
    (20.0, 100.0)
    """
    s, k, r, q, t = (
        as_array(value) for value in (spot, strike, rate, dividend_yield, maturity)
    )
    arguments = (spot, strike, rate, dividend_yield, maturity)

    discounted_spot = s * np.exp(-q * t)
    discounted_strike = k * np.exp(-r * t)

    if kind is OptionKind.CALL:
        lower = np.maximum(discounted_spot - discounted_strike, 0.0)
        upper = discounted_spot
    else:
        lower = np.maximum(discounted_strike - discounted_spot, 0.0)
        upper = discounted_strike

    return scalarize(lower, *arguments), scalarize(upper, *arguments)


def parity_residual(
    call: Numeric,
    put: Numeric,
    spot: Numeric,
    strike: Numeric,
    rate: Numeric,
    dividend_yield: Numeric,
    maturity: Numeric,
) -> Numeric:
    r"""Measure the departure from put-call parity.

    Parity states that :math:`C - P = S e^{-q\tau} - K e^{-r\tau}`, because a long call
    and a short put replicate a forward. This function returns the left side minus the
    right, which is zero to rounding for any consistent pair of prices.

    Parameters
    ----------
    call, put
        Prices of a call and a put on the same underlying, with the same strike and
        expiry.
    spot
        Current price of the underlying, strictly positive.
    strike
        Exercise price, strictly positive.
    rate
        Continuously compounded risk-free rate, as a decimal.
    dividend_yield
        Continuous dividend yield, as a decimal.
    maturity
        Time to expiry in years, non-negative.

    Returns
    -------
    Numeric
        The residual. Nonzero values are quoted in the same units as the prices, so the
        magnitude is directly interpretable as a mispricing.

    Notes
    -----
    Returned as a signed quantity rather than a boolean so that a caller can apply a
    tolerance appropriate to its own context: a test wants machine precision, whereas a
    check against quoted market prices must accommodate the bid-ask spread.

    Examples
    --------
    Black-Scholes prices satisfy parity to rounding:

    >>> from bsm.pricing import call_price, put_price
    >>> args = (100.0, 100.0, 0.05, 0.0, 1.0)
    >>> residual = parity_residual(
    ...     call_price(*args, 0.2), put_price(*args, 0.2), *args
    ... )
    >>> abs(residual) < 1e-12
    True

    An inconsistent pair reports the size of the inconsistency:

    >>> round(parity_residual(11.0, 5.573526, 100.0, 100.0, 0.05, 0.0, 1.0), 4)
    0.5494
    """
    arguments = (call, put, spot, strike, rate, dividend_yield, maturity)
    c, p, s, k, r, q, t = (as_array(value) for value in arguments)
    residual = (c - p) - (s * np.exp(-q * t) - k * np.exp(-r * t))
    return scalarize(residual, *arguments)
