r"""Analytic sensitivities of the Black-Scholes-Merton price.

Each Greek below is a closed-form partial derivative of the price, not a finite
difference. The closed forms are exact and cost about as much as one price evaluation;
finite differences appear only in the test suite, where they check these formulas
against the pricer they are supposed to differentiate.

Conventions
-----------
Every quantity is a raw derivative with respect to its variable in the units the rest
of the package uses: volatility and rates in decimals, time in years. Trading desks
usually rescale — vega per one volatility point, theta per calendar day, rho per basis
point — but rescaling is a presentation choice, and doing it here would mean the
identities that hold between the Greeks no longer hold between the returned numbers.
Divide by 100, 365, or 10,000 at the point of display.

Theta follows the market sign convention: it is :math:`\partial V / \partial t`, the
rate of change as the clock runs forward, so it is negative for most long positions.
Note that it is the negative of :math:`\partial V / \partial \tau`, the derivative with
respect to time *remaining*, which is what a naive finite difference in ``maturity``
computes.

Degenerate inputs
-----------------
Where total volatility vanishes, gamma and the time-decay term of theta are
:math:`0/0`. The limits are taken explicitly: gamma is zero, and the decay term is
zero. Delta, vega and rho need no special handling, because
:math:`\varphi(\pm\infty) = 0` and :math:`\Phi(\pm\infty)` is exactly 1 or 0 make the
formulas collapse on their own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from bsm._numeric import BoolArray, FloatArray, Numeric, as_array, scalarize
from bsm.normal import cdf, pdf
from bsm.pricing import _MIN_TOTAL_VOLATILITY, _d1_d2_arrays
from bsm.types import Contract, Market, OptionKind

__all__ = ["Greeks", "delta", "gamma", "greeks", "rho", "theta", "vega"]


class _Common(NamedTuple):
    """Quantities every Greek shares, computed once per call.

    Attributes
    ----------
    spot, strike, rate, dividend_yield, maturity, volatility
        The inputs, coerced to float arrays.
    d1, d2
        The two Black-Scholes arguments to the normal CDF.
    discount_spot, discount_strike
        The factors ``exp(-q * tau)`` and ``exp(-r * tau)``.
    degenerate
        Mask marking cells where total volatility vanishes, and where gamma and the
        decay term of theta must be replaced by their limits.
    """

    spot: FloatArray
    strike: FloatArray
    rate: FloatArray
    dividend_yield: FloatArray
    maturity: FloatArray
    volatility: FloatArray
    d1: FloatArray
    d2: FloatArray
    discount_spot: FloatArray
    discount_strike: FloatArray
    degenerate: BoolArray


def _prepare(
    spot: Numeric,
    strike: Numeric,
    rate: Numeric,
    dividend_yield: Numeric,
    maturity: Numeric,
    volatility: Numeric,
) -> _Common:
    """Compute the quantities every Greek shares.

    Parameters
    ----------
    spot, strike, rate, dividend_yield, maturity, volatility
        Scalars or broadcastable arrays.

    Returns
    -------
    _Common
        The shared intermediates.
    """
    s, k, r, q, t, v = (
        as_array(value)
        for value in (spot, strike, rate, dividend_yield, maturity, volatility)
    )
    d1, d2 = _d1_d2_arrays(s, k, r, q, t, v)
    return _Common(
        spot=s,
        strike=k,
        rate=r,
        dividend_yield=q,
        maturity=t,
        volatility=v,
        d1=d1,
        d2=d2,
        discount_spot=np.exp(-q * t),
        discount_strike=np.exp(-r * t),
        degenerate=v * np.sqrt(t) <= _MIN_TOTAL_VOLATILITY,
    )


def delta(
    kind: OptionKind,
    spot: Numeric,
    strike: Numeric,
    rate: Numeric,
    dividend_yield: Numeric,
    maturity: Numeric,
    volatility: Numeric,
) -> Numeric:
    r"""Sensitivity of the price to the underlying, :math:`\partial V / \partial S`.

    Parameters
    ----------
    kind
        Call or put.
    spot, strike, rate, dividend_yield, maturity, volatility
        Market and contract parameters; see :mod:`bsm.pricing`.

    Returns
    -------
    Numeric
        :math:`e^{-q\tau}\Phi(d_1)` for a call, :math:`-e^{-q\tau}\Phi(-d_1)` for a
        put. A call delta lies in :math:`[0, e^{-q\tau}]`, a put delta in
        :math:`[-e^{-q\tau}, 0]`.

    Examples
    --------
    An at-the-money call is slightly more than half sensitive, because the forward sits
    above the strike:

    >>> round(delta(OptionKind.CALL, 100.0, 100.0, 0.05, 0.0, 1.0, 0.2), 6)
    0.636831

    Call and put delta differ by the dividend discount factor, which is parity
    differentiated:

    >>> args = (100.0, 100.0, 0.05, 0.0, 1.0, 0.2)
    >>> round(delta(OptionKind.CALL, *args) - delta(OptionKind.PUT, *args), 12)
    1.0

    Deep in the money, a call moves one for one with the underlying:

    >>> round(delta(OptionKind.CALL, 1000.0, 100.0, 0.05, 0.0, 1.0, 0.2), 6)
    1.0
    """
    arguments = (spot, strike, rate, dividend_yield, maturity, volatility)
    common = _prepare(*arguments)
    if kind is OptionKind.CALL:
        result = common.discount_spot * cdf(common.d1)
    else:
        result = -common.discount_spot * cdf(-common.d1)
    return scalarize(result, *arguments)


def gamma(
    spot: Numeric,
    strike: Numeric,
    rate: Numeric,
    dividend_yield: Numeric,
    maturity: Numeric,
    volatility: Numeric,
) -> Numeric:
    r"""Curvature with respect to the underlying, :math:`\partial^2 V / \partial S^2`.

    Parameters
    ----------
    spot, strike, rate, dividend_yield, maturity, volatility
        Market and contract parameters; see :mod:`bsm.pricing`.

    Returns
    -------
    Numeric
        :math:`e^{-q\tau}\varphi(d_1) / (S\sigma\sqrt{\tau})`, and zero where total
        volatility vanishes.

    Notes
    -----
    Takes no ``kind`` argument: a call and a put with the same terms have identical
    gamma. Put-call parity is linear in the underlying, so it contributes nothing to
    the second derivative. The same argument applies to vega.

    Examples
    --------
    >>> round(gamma(100.0, 100.0, 0.05, 0.0, 1.0, 0.2), 8)
    0.01876202

    Gamma is largest near the money and decays in both directions:

    >>> atm = gamma(100.0, 100.0, 0.05, 0.0, 1.0, 0.2)
    >>> gamma(60.0, 100.0, 0.05, 0.0, 1.0, 0.2) < atm
    True
    >>> gamma(160.0, 100.0, 0.05, 0.0, 1.0, 0.2) < atm
    True

    An expired option has no curvature left:

    >>> gamma(100.0, 100.0, 0.05, 0.0, 0.0, 0.2)
    0.0
    """
    arguments = (spot, strike, rate, dividend_yield, maturity, volatility)
    common = _prepare(*arguments)
    denominator = common.spot * common.volatility * np.sqrt(common.maturity)
    safe = np.where(common.degenerate, 1.0, denominator)
    result = np.where(
        common.degenerate, 0.0, common.discount_spot * pdf(common.d1) / safe
    )
    return scalarize(result, *arguments)


def vega(
    spot: Numeric,
    strike: Numeric,
    rate: Numeric,
    dividend_yield: Numeric,
    maturity: Numeric,
    volatility: Numeric,
) -> Numeric:
    r"""Sensitivity to volatility, :math:`\partial V / \partial \sigma`.

    Parameters
    ----------
    spot, strike, rate, dividend_yield, maturity, volatility
        Market and contract parameters; see :mod:`bsm.pricing`.

    Returns
    -------
    Numeric
        :math:`S e^{-q\tau}\varphi(d_1)\sqrt{\tau}`, per unit of volatility. Divide by
        100 for the conventional "per volatility point" quote.

    Notes
    -----
    Strictly positive for any option with time and volatility remaining, which is the
    fact that makes implied volatility well posed: the price is strictly increasing in
    :math:`\sigma`, so the inverse is unique. It approaches zero for deep in- and
    out-of-the-money options, which is also why a Newton solver on this problem is
    unreliable and :mod:`bsm.implied` brackets instead.

    Examples
    --------
    >>> round(vega(100.0, 100.0, 0.05, 0.0, 1.0, 0.2), 6)
    37.524035

    Vanishes at expiry and far from the money:

    >>> vega(100.0, 100.0, 0.05, 0.0, 0.0, 0.2)
    0.0
    >>> round(vega(100.0, 1000.0, 0.05, 0.0, 0.25, 0.1), 12)
    0.0
    """
    arguments = (spot, strike, rate, dividend_yield, maturity, volatility)
    common = _prepare(*arguments)
    result = (
        common.spot * common.discount_spot * pdf(common.d1) * np.sqrt(common.maturity)
    )
    return scalarize(result, *arguments)


def theta(
    kind: OptionKind,
    spot: Numeric,
    strike: Numeric,
    rate: Numeric,
    dividend_yield: Numeric,
    maturity: Numeric,
    volatility: Numeric,
) -> Numeric:
    r"""Sensitivity to the passage of time, :math:`\partial V / \partial t`.

    Parameters
    ----------
    kind
        Call or put.
    spot, strike, rate, dividend_yield, maturity, volatility
        Market and contract parameters; see :mod:`bsm.pricing`.

    Returns
    -------
    Numeric
        Theta per year, in the market sign convention: negative where holding the
        option costs time value. Divide by 365 for a per-day quote.

    Notes
    -----
    This is the negative of the derivative with respect to time to maturity. A finite
    difference in the ``maturity`` argument therefore recovers ``-theta``, which is the
    single most common sign error in Greek implementations and is asserted explicitly
    in the test suite.

    Examples
    --------
    >>> round(theta(OptionKind.CALL, 100.0, 100.0, 0.05, 0.0, 1.0, 0.2), 6)
    -6.414028

    Per calendar day, which is how a desk would quote it:

    >>> round(theta(OptionKind.CALL, 100.0, 100.0, 0.05, 0.0, 1.0, 0.2) / 365, 4)
    -0.0176

    A deep in-the-money put can have positive theta: the discounted strike it will
    receive draws nearer:

    >>> theta(OptionKind.PUT, 40.0, 100.0, 0.05, 0.0, 1.0, 0.2) > 0
    True
    """
    arguments = (spot, strike, rate, dividend_yield, maturity, volatility)
    s, k, r, q, t, v, d1, d2, discount_spot, discount_strike, degenerate = _prepare(
        *arguments
    )

    # The decay term is 0/0 at zero total volatility; its limit there is zero.
    root_time = np.sqrt(t)
    safe_root_time = np.where(degenerate, 1.0, root_time)
    decay = np.where(
        degenerate, 0.0, -s * discount_spot * pdf(d1) * v / (2.0 * safe_root_time)
    )

    if kind is OptionKind.CALL:
        result = (
            decay - r * k * discount_strike * cdf(d2) + q * s * discount_spot * cdf(d1)
        )
    else:
        result = (
            decay
            + r * k * discount_strike * cdf(-d2)
            - q * s * discount_spot * cdf(-d1)
        )
    return scalarize(result, *arguments)


def rho(
    kind: OptionKind,
    spot: Numeric,
    strike: Numeric,
    rate: Numeric,
    dividend_yield: Numeric,
    maturity: Numeric,
    volatility: Numeric,
) -> Numeric:
    r"""Sensitivity to the risk-free rate, :math:`\partial V / \partial r`.

    Parameters
    ----------
    kind
        Call or put.
    spot, strike, rate, dividend_yield, maturity, volatility
        Market and contract parameters; see :mod:`bsm.pricing`.

    Returns
    -------
    Numeric
        :math:`K\tau e^{-r\tau}\Phi(d_2)` for a call and
        :math:`-K\tau e^{-r\tau}\Phi(-d_2)` for a put, per unit of rate. Divide by
        10,000 for a per-basis-point quote.

    Examples
    --------
    >>> round(rho(OptionKind.CALL, 100.0, 100.0, 0.05, 0.0, 1.0, 0.2), 6)
    53.232482

    A call gains and a put loses when rates rise, since a call defers payment of the
    strike and a put defers its receipt:

    >>> rho(OptionKind.PUT, 100.0, 100.0, 0.05, 0.0, 1.0, 0.2) < 0
    True

    An expired option is insensitive to rates:

    >>> rho(OptionKind.CALL, 100.0, 100.0, 0.05, 0.0, 0.0, 0.2)
    0.0
    """
    arguments = (spot, strike, rate, dividend_yield, maturity, volatility)
    common = _prepare(*arguments)
    scale = common.strike * common.maturity * common.discount_strike
    if kind is OptionKind.CALL:
        result = scale * cdf(common.d2)
    else:
        result = -scale * cdf(-common.d2)
    return scalarize(result, *arguments)


@dataclass(frozen=True, slots=True)
class Greeks:
    """The five first- and second-order sensitivities of an option price.

    Attributes
    ----------
    delta
        Sensitivity to the underlying, per unit of price.
    gamma
        Curvature with respect to the underlying.
    vega
        Sensitivity to volatility, per unit (not per point).
    theta
        Sensitivity to the passage of time, per year (not per day).
    rho
        Sensitivity to the risk-free rate, per unit (not per basis point).
    """

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def greeks(contract: Contract, market: Market) -> Greeks:
    """Compute all five Greeks for a contract against a market state.

    The validated entry point, mirroring :func:`bsm.pricing.price`. Both arguments have
    already checked their own preconditions at construction.

    Parameters
    ----------
    contract
        The option's terms.
    market
        Spot, rate, volatility, and dividend yield.

    Returns
    -------
    Greeks
        All five sensitivities.

    Examples
    --------
    >>> from bsm.types import Contract, Market, OptionKind
    >>> contract = Contract(strike=100.0, maturity=1.0, kind=OptionKind.CALL)
    >>> market = Market(spot=100.0, rate=0.05, volatility=0.2)
    >>> result = greeks(contract, market)
    >>> round(result.delta, 6)
    0.636831
    >>> round(result.vega, 6)
    37.524035
    """
    arguments = (
        market.spot,
        contract.strike,
        market.rate,
        market.dividend_yield,
        contract.maturity,
        market.volatility,
    )
    return Greeks(
        delta=float(delta(contract.kind, *arguments)),
        gamma=float(gamma(*arguments)),
        vega=float(vega(*arguments)),
        theta=float(theta(contract.kind, *arguments)),
        rho=float(rho(contract.kind, *arguments)),
    )
