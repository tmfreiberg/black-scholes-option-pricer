"""Value objects describing an option contract and the market it is priced in.

The package draws a deliberate line between two layers:

* This module is the **validated boundary**. Constructing a :class:`Contract` or a
  :class:`Market` checks every precondition the pricing formulas rely on, and raises
  immediately if one is violated.
* The functions in :mod:`bsm.pricing` are the **unvalidated core**. They accept scalars
  or arrays, assume their preconditions hold, and do no checking at all, because they
  are called once per cell of a sensitivity grid.

Callers who build objects from this module never need to think about that distinction.
Callers who reach for the raw formulas take responsibility for their own inputs, and
each such function documents what it assumes.

Both classes are frozen and slotted, so an instance can be used as a dictionary key,
shared freely, and cannot acquire a typo'd attribute.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

__all__ = ["Contract", "Market", "OptionKind"]


class OptionKind(StrEnum):
    """Whether an option confers the right to buy or to sell.

    Inheriting from :class:`enum.StrEnum` means members compare equal to their string
    value, which keeps command-line arguments, database columns, and JSON payloads free
    of conversion boilerplate.

    Examples
    --------
    >>> OptionKind("call") is OptionKind.CALL
    True
    >>> OptionKind.PUT == "put"
    True
    """

    CALL = "call"
    PUT = "put"


def _require_finite(value: float, name: str) -> None:
    """Raise :class:`ValueError` unless ``value`` is a finite real number.

    Parameters
    ----------
    value
        The quantity to check.
    name
        Field name, used to build the error message.

    Raises
    ------
    ValueError
        If ``value`` is NaN or infinite.
    """
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


@dataclass(frozen=True, slots=True)
class Contract:
    """The terms of a European option.

    Parameters
    ----------
    strike
        Exercise price, strictly positive.
    maturity
        Time to expiry in years, non-negative. Obtain it from
        :func:`bsm.daycount.year_fraction` or :func:`bsm.daycount.tenor_to_years`
        rather than by arithmetic on calendar units, which mixes incompatible year
        lengths unless it is done against real dates.
    kind
        Call or put.

    Raises
    ------
    ValueError
        If ``strike`` is not strictly positive, or ``maturity`` is negative, or either
        is not finite.

    Examples
    --------
    >>> Contract(strike=100.0, maturity=0.5, kind=OptionKind.CALL)
    Contract(strike=100.0, maturity=0.5, kind=<OptionKind.CALL: 'call'>)

    An expired contract is legal and prices to its intrinsic value:

    >>> Contract(strike=100.0, maturity=0.0, kind=OptionKind.PUT).maturity
    0.0

    A negative strike is not:

    >>> Contract(strike=-1.0, maturity=1.0, kind=OptionKind.CALL)
    Traceback (most recent call last):
        ...
    ValueError: strike must be strictly positive, got -1.0
    """

    strike: float
    maturity: float
    kind: OptionKind

    def __post_init__(self) -> None:
        """Validate the contract terms."""
        _require_finite(self.strike, "strike")
        _require_finite(self.maturity, "maturity")
        if self.strike <= 0.0:
            raise ValueError(f"strike must be strictly positive, got {self.strike!r}")
        if self.maturity < 0.0:
            raise ValueError(f"maturity must be non-negative, got {self.maturity!r}")

    @property
    def is_expired(self) -> bool:
        """Whether the contract has reached expiry.

        Examples
        --------
        >>> Contract(100.0, 0.0, OptionKind.CALL).is_expired
        True
        """
        return self.maturity == 0.0


@dataclass(frozen=True, slots=True)
class Market:
    """The market state an option is priced against.

    Rates and volatility are held as decimals, never percentages. Carrying both
    conventions in one codebase is a reliable source of silent unit errors, so the
    convention is fixed once here and enforced by the type system thereafter: if it is
    a :class:`Market`, it is decimal. Percentages belong at the presentation layer.

    Parameters
    ----------
    spot
        Current price of the underlying, strictly positive.
    rate
        Continuously compounded risk-free rate, as a decimal. May be negative; several
        sovereign curves have been.
    volatility
        Annualised volatility of the underlying, as a decimal, non-negative. Zero is
        admissible and yields the discounted intrinsic value.
    dividend_yield
        Continuous dividend yield, as a decimal. Defaults to zero, which recovers the
        original 1973 Black-Scholes formula from Merton's generalisation.

    Raises
    ------
    ValueError
        If ``spot`` is not strictly positive, ``volatility`` is negative, or any field
        is not finite.

    Examples
    --------
    >>> Market(spot=100.0, rate=0.05, volatility=0.2)
    Market(spot=100.0, rate=0.05, volatility=0.2, dividend_yield=0.0)

    Percentages are a common mistake and cannot be detected automatically — 20.0 is a
    legal volatility of 2000% — so the constructor accepts it and the documentation
    carries the warning:

    >>> Market(spot=100.0, rate=0.05, volatility=20.0).volatility
    20.0
    """

    spot: float
    rate: float
    volatility: float
    dividend_yield: float = 0.0

    def __post_init__(self) -> None:
        """Validate the market state."""
        _require_finite(self.spot, "spot")
        _require_finite(self.rate, "rate")
        _require_finite(self.volatility, "volatility")
        _require_finite(self.dividend_yield, "dividend_yield")
        if self.spot <= 0.0:
            raise ValueError(f"spot must be strictly positive, got {self.spot!r}")
        if self.volatility < 0.0:
            raise ValueError(
                f"volatility must be non-negative, got {self.volatility!r}"
            )
