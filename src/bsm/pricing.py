r"""European option prices under Black-Scholes-Merton.

The formula priced here is Merton's generalisation carrying a continuous dividend
yield :math:`q`; setting :math:`q = 0` recovers the original 1973 Black-Scholes
result. For a call with spot :math:`S`, strike :math:`K`, rate :math:`r`, yield
:math:`q`, maturity :math:`\tau` and volatility :math:`\sigma`,

.. math::

    C = S e^{-q\tau} \Phi(d_1) - K e^{-r\tau} \Phi(d_2),

.. math::

    d_1 = \frac{\ln(S/K) + (r - q + \sigma^2/2)\tau}{\sigma\sqrt{\tau}},
    \qquad d_2 = d_1 - \sigma\sqrt{\tau},

and the put follows from put-call parity, or equivalently by reflecting both
:math:`d`'s.

Degenerate inputs
-----------------
Two boundary cases matter and are handled without a special-cased branch in the price
functions themselves. When :math:`\sigma\sqrt{\tau} = 0` — either an expired
contract or a riskless underlying — the ratio defining :math:`d_1` is undefined, but
its limit is not: both :math:`d`'s tend to :math:`+\infty` when the forward exceeds
the strike and :math:`-\infty` when it does not. :func:`d1_d2` returns those limits
directly, so :math:`\Phi` evaluates to exactly 1 or 0 and the price formula collapses
to the discounted intrinsic value on its own.

This is why :func:`bsm.normal.cdf` is careful about infinities. The alternative — an
``if`` on maturity inside each price function — does not vectorise, and has to be
written once for the call and again for the put.

Clamping to the no-arbitrage bounds
-----------------------------------
Both price functions clip their result into
:math:`[\max(Se^{-q\tau} - Ke^{-r\tau},\ 0),\ Se^{-q\tau}]` for a call, and the
analogous interval for a put. These bounds are model-free consequences of arbitrage, so
a price outside them is always a rounding artefact rather than a result.

Such artefacts do occur. Far out of the money the two terms of the formula agree to
within rounding and their difference can come out a denormal below zero; far in the
money it can land one unit in the last place below the lower bound. Both are of no
financial consequence and both break code downstream that is entitled to assume the
bounds hold — the implied-volatility solver rejects a price below its own bracket,
which is the correct behaviour for a genuinely impossible quote and the wrong one for a
rounding error of 1e-322. Clipping costs two operations and removes the whole class.

Preconditions
-------------
These functions are the unvalidated core described in :mod:`bsm.types`. They assume
``spot > 0``, ``strike > 0``, ``maturity >= 0`` and ``volatility >= 0``, and check
none of it, because they are called once per cell of a grid. Build a
:class:`~bsm.types.Contract` and a :class:`~bsm.types.Market` and call :func:`price`
to get those checks.
"""

from __future__ import annotations

from typing import overload

import numpy as np

from bsm._numeric import FloatArray, Numeric, as_array, scalarize
from bsm.normal import cdf
from bsm.types import Contract, Market, OptionKind

__all__ = ["call_price", "d1_d2", "forward_price", "price", "put_price"]

_MIN_TOTAL_VOLATILITY: float = 1e-100
r"""Below this, total volatility is treated as exactly zero.

At :math:`\sigma\sqrt{\tau} = 10^{-100}` an option's entire time value is of order
:math:`10^{-100}` times the forward — far below the resolution of a double relative to
the intrinsic value, and unimaginably below the tick size of any traded instrument. So
nothing is lost by taking the limit, and something is gained: dividing by a subnormal
denominator overflows to infinity and emits a warning, which is how this threshold came
to exist.
"""


# Every public formula below carries two overloads. The first matches only when all six
# arguments are Python floats and promises a float; the second matches otherwise and
# promises an array. Type checkers therefore know that `call_price(90.0, ...)` is a
# float while `call_price(spots, ...)` is an array, without the caller casting. The
# alternative — a bare `Numeric` return — pushes an `isinstance` check onto every
# caller who wants to index the result.
#
# The all-float signature is a subtype of the general one, so mypy flags the pair as an
# unsafe overlap. It is not unsafe here: `scalarize` returns a float precisely when
# every input was scalar, which is the relationship the two overloads encode and which
# mypy has no way to verify. Each first overload therefore silences that one code, and
# only that one.


@overload
def forward_price(  # type: ignore[overload-overlap]
    spot: float, rate: float, dividend_yield: float, maturity: float
) -> float: ...


@overload
def forward_price(
    spot: Numeric, rate: Numeric, dividend_yield: Numeric, maturity: Numeric
) -> FloatArray: ...


def forward_price(
    spot: Numeric, rate: Numeric, dividend_yield: Numeric, maturity: Numeric
) -> Numeric:
    r"""Compute the forward price of the underlying at maturity.

    Parameters
    ----------
    spot
        Current price of the underlying.
    rate
        Continuously compounded risk-free rate, as a decimal.
    dividend_yield
        Continuous dividend yield, as a decimal.
    maturity
        Time to expiry in years.

    Returns
    -------
    Numeric
        :math:`F = S e^{(r - q)\tau}`.

    Notes
    -----
    Exported because the sign of :math:`F - K` decides moneyness, which the zero
    volatility limit in :func:`d1_d2` and the arbitrage bounds both need.

    Examples
    --------
    >>> round(forward_price(100.0, 0.05, 0.0, 1.0), 6)
    105.12711

    A dividend yield equal to the rate leaves the forward at spot:

    >>> forward_price(100.0, 0.05, 0.05, 1.0)
    100.0
    """
    result = as_array(spot) * np.exp(
        (as_array(rate) - as_array(dividend_yield)) * as_array(maturity)
    )
    return scalarize(result, spot, rate, dividend_yield, maturity)


def _d1_d2_arrays(
    spot: FloatArray,
    strike: FloatArray,
    rate: FloatArray,
    dividend_yield: FloatArray,
    maturity: FloatArray,
    volatility: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    r"""Compute ``d1`` and ``d2`` on arrays, including the degenerate limits.

    Parameters
    ----------
    spot, strike, rate, dividend_yield, maturity, volatility
        Broadcastable float arrays.

    Returns
    -------
    tuple of FloatArray
        ``(d1, d2)``, with :math:`\pm\infty` wherever total volatility is zero.
    """
    total_volatility = volatility * np.sqrt(maturity)
    log_moneyness = np.log(spot / strike)
    carry = (rate - dividend_yield) * maturity

    # Substitute a harmless denominator where the real one vanishes, compute
    # unconditionally, then overwrite those positions with the true limits. Guarding
    # this way keeps the expression vectorised and avoids a divide-by-zero warning.
    #
    # The threshold is _MIN_TOTAL_VOLATILITY rather than zero. A merely nonzero
    # denominator is not enough: a subnormal volatility such as 1e-309 passes a `> 0`
    # test and then overflows the division to infinity, with a warning. This was found
    # by the property suite, not by inspection.
    degenerate = total_volatility <= _MIN_TOTAL_VOLATILITY
    denominator = np.where(degenerate, 1.0, total_volatility)

    d1 = (
        log_moneyness + carry + 0.5 * volatility * volatility * maturity
    ) / denominator
    d2 = d1 - total_volatility

    # sign(F - K) has the same sign as log(S/K) + (r - q)*tau.
    limit = np.where(log_moneyness + carry > 0.0, np.inf, -np.inf)
    return np.where(degenerate, limit, d1), np.where(degenerate, limit, d2)


@overload
def d1_d2(  # type: ignore[overload-overlap]
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    maturity: float,
    volatility: float,
) -> tuple[float, float]: ...


@overload
def d1_d2(
    spot: Numeric,
    strike: Numeric,
    rate: Numeric,
    dividend_yield: Numeric,
    maturity: Numeric,
    volatility: Numeric,
) -> tuple[FloatArray, FloatArray]: ...


def d1_d2(
    spot: Numeric,
    strike: Numeric,
    rate: Numeric,
    dividend_yield: Numeric,
    maturity: Numeric,
    volatility: Numeric,
) -> tuple[Numeric, Numeric]:
    r"""Compute the two Black-Scholes arguments to :math:`\Phi`.

    Parameters
    ----------
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
    volatility
        Annualised volatility, as a decimal, non-negative.

    Returns
    -------
    tuple of Numeric
        ``(d1, d2)``. Where :math:`\sigma\sqrt{\tau} = 0` these are
        :math:`+\infty` if the forward exceeds the strike and :math:`-\infty`
        otherwise, which are the correct limits.

    Examples
    --------
    At the money with a year to run:

    >>> d1, d2 = d1_d2(100.0, 100.0, 0.05, 0.0, 1.0, 0.2)
    >>> round(d1, 6), round(d2, 6)
    (0.35, 0.15)

    The two differ by exactly the total volatility:

    >>> round(d1 - d2 - 0.2 * 1.0, 15)
    0.0

    An expired in-the-money call takes the positive limit:

    >>> d1_d2(120.0, 100.0, 0.05, 0.0, 0.0, 0.2)
    (inf, inf)

    An expired out-of-the-money call takes the negative one:

    >>> d1_d2(80.0, 100.0, 0.05, 0.0, 0.0, 0.2)
    (-inf, -inf)
    """
    arguments = (spot, strike, rate, dividend_yield, maturity, volatility)
    d1, d2 = _d1_d2_arrays(*(as_array(argument) for argument in arguments))
    return scalarize(d1, *arguments), scalarize(d2, *arguments)


@overload
def call_price(  # type: ignore[overload-overlap]
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    maturity: float,
    volatility: float,
) -> float: ...


@overload
def call_price(
    spot: Numeric,
    strike: Numeric,
    rate: Numeric,
    dividend_yield: Numeric,
    maturity: Numeric,
    volatility: Numeric,
) -> FloatArray: ...


def call_price(
    spot: Numeric,
    strike: Numeric,
    rate: Numeric,
    dividend_yield: Numeric,
    maturity: Numeric,
    volatility: Numeric,
) -> Numeric:
    r"""Compute the price of a European call.

    Parameters
    ----------
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
    volatility
        Annualised volatility, as a decimal, non-negative.

    Returns
    -------
    Numeric
        :math:`S e^{-q\tau}\Phi(d_1) - K e^{-r\tau}\Phi(d_2)`.

    Examples
    --------
    The textbook at-the-money case:

    >>> round(call_price(100.0, 100.0, 0.05, 0.0, 1.0, 0.2), 6)
    10.450584

    At expiry the price is the payoff:

    >>> call_price(120.0, 100.0, 0.05, 0.0, 0.0, 0.2)
    20.0
    >>> call_price(80.0, 100.0, 0.05, 0.0, 0.0, 0.2)
    0.0

    With no volatility it is the discounted intrinsic value on the forward:

    >>> round(call_price(100.0, 100.0, 0.05, 0.0, 1.0, 0.0), 6)
    4.877058

    Any argument may be an array, and they broadcast against one another. This is how
    :mod:`bsm.grids` fills a surface — one call, no Python loop:

    >>> spots = np.array([90.0, 100.0, 110.0])
    >>> call_price(spots, 100.0, 0.05, 0.0, 1.0, 0.2).round(4)
    array([ 5.0912, 10.4506, 17.663 ])
    """
    arguments = (spot, strike, rate, dividend_yield, maturity, volatility)
    s, k, r, q, t, v = (as_array(argument) for argument in arguments)
    d1, d2 = _d1_d2_arrays(s, k, r, q, t, v)
    discounted_spot = s * np.exp(-q * t)
    discounted_strike = k * np.exp(-r * t)
    result = discounted_spot * cdf(d1) - discounted_strike * cdf(d2)
    result = np.clip(
        result, np.maximum(discounted_spot - discounted_strike, 0.0), discounted_spot
    )
    return scalarize(result, *arguments)


@overload
def put_price(  # type: ignore[overload-overlap]
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    maturity: float,
    volatility: float,
) -> float: ...


@overload
def put_price(
    spot: Numeric,
    strike: Numeric,
    rate: Numeric,
    dividend_yield: Numeric,
    maturity: Numeric,
    volatility: Numeric,
) -> FloatArray: ...


def put_price(
    spot: Numeric,
    strike: Numeric,
    rate: Numeric,
    dividend_yield: Numeric,
    maturity: Numeric,
    volatility: Numeric,
) -> Numeric:
    r"""Compute the price of a European put.

    Parameters
    ----------
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
    volatility
        Annualised volatility, as a decimal, non-negative.

    Returns
    -------
    Numeric
        :math:`K e^{-r\tau}\Phi(-d_2) - S e^{-q\tau}\Phi(-d_1)`.

    Notes
    -----
    Computed directly rather than through put-call parity, so that parity becomes an
    independent invariant the test suite can check rather than an identity the code
    imposes by construction.

    Examples
    --------
    >>> round(put_price(100.0, 100.0, 0.05, 0.0, 1.0, 0.2), 6)
    5.573526

    Put-call parity, :math:`C - P = S e^{-q\tau} - K e^{-r\tau}`, holds to machine
    precision:

    >>> import math
    >>> call = call_price(100.0, 100.0, 0.05, 0.0, 1.0, 0.2)
    >>> put = put_price(100.0, 100.0, 0.05, 0.0, 1.0, 0.2)
    >>> abs(round(call - put - (100.0 - 100.0 * math.exp(-0.05)), 12))
    0.0

    At expiry the price is the payoff:

    >>> put_price(80.0, 100.0, 0.05, 0.0, 0.0, 0.2)
    20.0
    """
    arguments = (spot, strike, rate, dividend_yield, maturity, volatility)
    s, k, r, q, t, v = (as_array(argument) for argument in arguments)
    d1, d2 = _d1_d2_arrays(s, k, r, q, t, v)
    discounted_spot = s * np.exp(-q * t)
    discounted_strike = k * np.exp(-r * t)
    result = discounted_strike * cdf(-d2) - discounted_spot * cdf(-d1)
    result = np.clip(
        result, np.maximum(discounted_strike - discounted_spot, 0.0), discounted_strike
    )
    return scalarize(result, *arguments)


def price(contract: Contract, market: Market) -> float:
    """Price a contract against a market state.

    The validated entry point, and the one to reach for unless filling a grid: both
    arguments have already checked their own preconditions at construction.

    Parameters
    ----------
    contract
        The option's terms.
    market
        Spot, rate, volatility, and dividend yield.

    Returns
    -------
    float
        The option's value.

    Examples
    --------
    >>> from bsm.types import Contract, Market, OptionKind
    >>> contract = Contract(strike=100.0, maturity=1.0, kind=OptionKind.CALL)
    >>> market = Market(spot=100.0, rate=0.05, volatility=0.2)
    >>> round(price(contract, market), 6)
    10.450584

    >>> put = Contract(strike=100.0, maturity=1.0, kind=OptionKind.PUT)
    >>> round(price(put, market), 6)
    5.573526
    """
    formula = call_price if contract.kind is OptionKind.CALL else put_price
    return float(
        formula(
            market.spot,
            contract.strike,
            market.rate,
            market.dividend_yield,
            contract.maturity,
            market.volatility,
        )
    )
