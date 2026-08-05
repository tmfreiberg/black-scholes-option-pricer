r"""Implied volatility: the inverse of the pricing map.

Given an observed price, find the volatility that reproduces it. This is well posed
because vega is strictly positive: the Black-Scholes price is a strictly increasing,
continuous function of :math:`\sigma` on :math:`(0, \infty)`, running from the lower
no-arbitrage bound at :math:`\sigma = 0` up to the upper bound as
:math:`\sigma \to \infty`. So a price strictly inside the bounds has exactly one
implied volatility, a price on the lower bound implies zero, and a price outside has
none.

That structure decides the algorithm. The bounds in :mod:`bsm.bounds` supply a bracket
for nothing, and :func:`bsm.roots.brentq` cannot escape a bracket, so the solve either
converges or reports why it could not. A Newton iteration would need no bracket but
divides by vega, which vanishes for deep in- and out-of-the-money options — exactly the
quotes for which implied volatility is most often requested and least well determined.

Where the inverse does not exist numerically
-------------------------------------------
Uniqueness is a statement about real numbers, not about doubles. A deep out-of-the-money
option at low volatility can be worth less than the smallest positive double, in which
case its computed price is exactly zero — indistinguishable from its price at zero
volatility. Zero is then returned, because it is the smallest volatility consistent with
the observed price and it reprices correctly. No solver can do better: the information
was destroyed by the forward computation, not by the inversion.

Vega is the diagnostic. Where it is negligible at the recovered volatility, the implied
volatility is poorly determined however it was obtained.

Failure is typed
----------------
Every way this can fail raises a subclass of :class:`ImpliedVolatilityError`, which is
itself a :class:`ValueError`. Nothing returns a sentinel value, a string, or ``None``.
A caller who wants a fallback writes ``except ImpliedVolatilityError``; a caller who
does not gets a traceback pointing at the actual problem.
"""

from __future__ import annotations

from bsm.bounds import price_bounds
from bsm.pricing import call_price, put_price
from bsm.roots import RootFindingError, brentq
from bsm.types import Contract, Market, OptionKind

__all__ = [
    "ImpliedVolatilityError",
    "NoMaturityRemaining",
    "PriceOutsideBounds",
    "SolverFailed",
    "implied_volatility",
    "implied_volatility_for",
]

# Volatility beyond this is not a meaningful answer, only evidence that the price does
# not correspond to any plausible one. 100.0 is 10,000% annualised.
_MAX_VOLATILITY: float = 100.0

# Starting width for the upper end of the bracket, doubled until it contains the root.
_INITIAL_UPPER: float = 1.0


class ImpliedVolatilityError(ValueError):
    """Base class for failures to invert the pricing map."""


class PriceOutsideBounds(ImpliedVolatilityError):
    """The observed price admits no implied volatility.

    Raised when the price is below the lower no-arbitrage bound or at or above the
    upper one. Below the lower bound the quote is an arbitrage rather than an option
    price; at or above the upper bound no finite volatility is large enough.
    """


class NoMaturityRemaining(ImpliedVolatilityError):
    """The contract has expired, so its price carries no volatility information.

    At zero maturity the price is the payoff, which does not depend on volatility at
    all. There is nothing to invert.
    """


class SolverFailed(ImpliedVolatilityError):
    """The price lies within bounds but the root-finder did not converge.

    This should not occur for well-formed inputs and indicates a numerical problem
    worth investigating rather than a bad quote.
    """


def implied_volatility(
    price: float,
    kind: OptionKind,
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    maturity: float,
    *,
    tolerance: float = 1e-12,
) -> float:
    r"""Recover the volatility implied by an observed option price.

    Parameters
    ----------
    price
        Observed price of the option.
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
        Time to expiry in years, strictly positive.
    tolerance
        Absolute tolerance on the returned volatility.

    Returns
    -------
    float
        The unique :math:`\sigma \ge 0` for which the Black-Scholes price equals
        ``price``.

    Raises
    ------
    NoMaturityRemaining
        If ``maturity`` is zero.
    PriceOutsideBounds
        If ``price`` is below the lower no-arbitrage bound, or at or above the upper
        one.
    SolverFailed
        If the bracketed solve does not converge.

    Notes
    -----
    Scalar only. Root-finding is iterative and each quote needs its own bracket, so
    there is nothing to vectorise; map over a chain of quotes in the caller.

    The returned volatility always reprices to ``price``, but is only *unique* where
    vega is appreciable. See the module docstring on prices that round to their lower
    bound.

    Examples
    --------
    Round-tripping a Black-Scholes price recovers the volatility that produced it:

    >>> from bsm.pricing import call_price
    >>> terms = (100.0, 100.0, 0.05, 0.0, 1.0)
    >>> observed = call_price(*terms, 0.2)
    >>> round(implied_volatility(observed, OptionKind.CALL, *terms), 10)
    0.2

    Puts invert just as readily:

    >>> from bsm.pricing import put_price
    >>> terms = (90.0, 100.0, 0.03, 0.01, 0.5)
    >>> observed = put_price(*terms, 0.35)
    >>> round(implied_volatility(observed, OptionKind.PUT, *terms), 10)
    0.35

    A price exactly on the lower bound implies zero volatility:

    >>> lower, _ = price_bounds(OptionKind.CALL, 100.0, 100.0, 0.05, 0.0, 1.0)
    >>> implied_volatility(lower, OptionKind.CALL, 100.0, 100.0, 0.05, 0.0, 1.0)
    0.0

    A price above the upper bound is rejected, with the bound reported:

    >>> implied_volatility(101.0, OptionKind.CALL, 100.0, 100.0, 0.05, 0.0, 1.0)
    Traceback (most recent call last):
        ...
    bsm.implied.PriceOutsideBounds: price 101 is not below the upper bound 100

    An expired contract says nothing about volatility:

    >>> implied_volatility(20.0, OptionKind.CALL, 120.0, 100.0, 0.05, 0.0, 0.0)
    Traceback (most recent call last):
        ...
    bsm.implied.NoMaturityRemaining: expired contracts carry no volatility information
    """
    if maturity <= 0.0:
        raise NoMaturityRemaining("expired contracts carry no volatility information")

    lower_bound, upper_bound = price_bounds(
        kind, spot, strike, rate, dividend_yield, maturity
    )

    if price < lower_bound:
        raise PriceOutsideBounds(
            f"price {price:g} is below the lower no-arbitrage bound {lower_bound:g}"
        )
    if price >= upper_bound:
        raise PriceOutsideBounds(
            f"price {price:g} is not below the upper bound {upper_bound:g}"
        )
    if price == lower_bound:
        # The bound is attained exactly at zero volatility, and the solver would be
        # asked to find a root sitting on the endpoint of its own bracket.
        return 0.0

    formula = call_price if kind is OptionKind.CALL else put_price

    def residual(volatility: float) -> float:
        return (
            float(formula(spot, strike, rate, dividend_yield, maturity, volatility))
            - price
        )

    # Grow the upper end until it overshoots. The price is increasing in volatility and
    # bounded above, so this terminates for any price strictly inside the bounds; the
    # cap turns an otherwise infinite loop into a reported failure.
    upper = _INITIAL_UPPER
    while residual(upper) < 0.0:
        upper *= 2.0
        if upper > _MAX_VOLATILITY:
            raise SolverFailed(
                f"no volatility below {_MAX_VOLATILITY:g} reproduces price {price:g}, "
                f"though it lies within the no-arbitrage bounds "
                f"[{lower_bound:g}, {upper_bound:g})"
            )

    try:
        return brentq(residual, 0.0, upper, xtol=tolerance)
    except RootFindingError as error:  # pragma: no cover - defensive
        raise SolverFailed(f"bracketed solve failed for price {price:g}") from error


def implied_volatility_for(
    price: float, contract: Contract, market: Market, *, tolerance: float = 1e-12
) -> float:
    """Recover implied volatility for a contract and market state.

    The validated entry point. ``market.volatility`` is ignored — it is the quantity
    being solved for — and everything else is taken from the two objects.

    Parameters
    ----------
    price
        Observed price of the option.
    contract
        The option's terms.
    market
        Spot, rate, and dividend yield. Its volatility is not used.
    tolerance
        Absolute tolerance on the returned volatility.

    Returns
    -------
    float
        The implied volatility.

    Raises
    ------
    ImpliedVolatilityError
        As for :func:`implied_volatility`.

    Examples
    --------
    >>> from bsm.pricing import price as bs_price
    >>> from bsm.types import Contract, Market, OptionKind
    >>> contract = Contract(strike=100.0, maturity=1.0, kind=OptionKind.CALL)
    >>> market = Market(spot=100.0, rate=0.05, volatility=0.28)
    >>> observed = bs_price(contract, market)
    >>> round(implied_volatility_for(observed, contract, market), 10)
    0.28
    """
    return implied_volatility(
        price,
        contract.kind,
        market.spot,
        contract.strike,
        market.rate,
        market.dividend_yield,
        contract.maturity,
        tolerance=tolerance,
    )
