r"""Day count conventions and calendar-correct tenor arithmetic.

Time to maturity enters the Black-Scholes formula through :math:`\sqrt{\tau}` and
:math:`e^{-r\tau}`, so an error in it biases every price.

The tempting shortcut is to sum calendar units directly::

    tau = years + months / 12 + weeks / 52 + days / 365

This is wrong, and wrong in a way that does not announce itself. Each term implies a
different year length: 12 months of 30.42 days, 52 weeks of 7 days (a 364-day year),
and 365 days. A tenor of "1 year 2 weeks" and the same tenor written as "54 weeks"
give different answers, and neither matches the number of days actually between the
two dates.

This module does calendar arithmetic on real dates and divides by an explicit basis.
:func:`tenor_to_years` advances an anchor date by the requested tenor and then measures
the result, so "1 month" means the actual number of days in that month.
"""

from __future__ import annotations

import calendar
from datetime import date
from enum import StrEnum

__all__ = ["DayCount", "add_months", "tenor_to_years", "year_fraction"]


class DayCount(StrEnum):
    """Supported day count conventions.

    Attributes
    ----------
    ACT_365F
        Actual days divided by a fixed 365. The market standard for equity and index
        options and the default throughout this package.
    ACT_360
        Actual days divided by a fixed 360. Common in money markets, offered here
        because the discount factor in the pricing formula is sometimes quoted on it.
    """

    ACT_365F = "ACT/365F"
    ACT_360 = "ACT/360"


_BASIS_DENOMINATOR: dict[DayCount, float] = {
    DayCount.ACT_365F: 365.0,
    DayCount.ACT_360: 360.0,
}


def year_fraction(start: date, end: date, basis: DayCount = DayCount.ACT_365F) -> float:
    """Measure the time between two dates in years.

    Parameters
    ----------
    start
        First date, typically the valuation date.
    end
        Second date, typically expiry. Must not precede ``start``.
    basis
        Day count convention. Defaults to ACT/365F.

    Returns
    -------
    float
        Non-negative time in years.

    Raises
    ------
    ValueError
        If ``end`` precedes ``start``. Negative time to maturity is meaningless, and
        silently returning a negative number would produce ``nan`` deep inside the
        pricing formula where the cause is much harder to see.

    Examples
    --------
    A full non-leap year is exactly one under ACT/365F:

    >>> year_fraction(date(2026, 1, 1), date(2027, 1, 1))
    1.0

    A leap year is slightly more, which is correct — there really was an extra day:

    >>> round(year_fraction(date(2028, 1, 1), date(2029, 1, 1)), 6)
    1.00274

    The same interval is longer when divided by 360:

    >>> round(year_fraction(date(2026, 1, 1), date(2027, 1, 1), DayCount.ACT_360), 6)
    1.013889
    """
    if end < start:
        raise ValueError(f"end ({end}) must not precede start ({start})")
    return (end - start).days / _BASIS_DENOMINATOR[basis]


def add_months(anchor: date, months: int) -> date:
    """Advance a date by a whole number of months, clamping to month end.

    Month arithmetic is not associative with day arithmetic and has no single correct
    answer at month ends. This function follows the market convention: the day of month
    is preserved where it exists in the target month, and clamped to the last day of
    that month where it does not.

    Parameters
    ----------
    anchor
        Starting date.
    months
        Number of months to advance. May be negative.

    Returns
    -------
    datetime.date
        The advanced date.

    Examples
    --------
    >>> add_months(date(2026, 1, 15), 3)
    datetime.date(2026, 4, 15)

    The 31st of January plus one month has no natural answer; it clamps to the 28th:

    >>> add_months(date(2026, 1, 31), 1)
    datetime.date(2026, 2, 28)

    In a leap year the same call clamps to the 29th:

    >>> add_months(date(2028, 1, 31), 1)
    datetime.date(2028, 2, 29)

    Clamping is lossy, so the operation does not invert:

    >>> add_months(add_months(date(2026, 1, 31), 1), -1)
    datetime.date(2026, 1, 28)

    Negative and multi-year offsets are supported:

    >>> add_months(date(2026, 3, 10), -15)
    datetime.date(2024, 12, 10)
    """
    zero_based = anchor.month - 1 + months
    year = anchor.year + zero_based // 12
    month = zero_based % 12 + 1
    last_day_of_month = calendar.monthrange(year, month)[1]
    return date(year, month, min(anchor.day, last_day_of_month))


def tenor_to_years(
    anchor: date,
    *,
    years: int = 0,
    months: int = 0,
    weeks: int = 0,
    days: int = 0,
    basis: DayCount = DayCount.ACT_365F,
) -> float:
    """Convert a tenor expressed in calendar units into a year fraction.

    The tenor is applied to ``anchor`` in a fixed order — years, then months, then
    weeks and days — and the resulting date is measured back against the anchor. This
    means the answer depends on the anchor date, which is not a defect but the point:
    three months from January is a different number of days than three months from
    June, and the option knows the difference.

    Parameters
    ----------
    anchor
        The valuation date the tenor is measured from.
    years, months, weeks, days
        Components of the tenor, applied in that order. Keyword-only, because
        ``tenor_to_years(today, 0, 3)`` is ambiguous to a reader in a way that
        ``tenor_to_years(today, months=3)`` is not.
    basis
        Day count convention. Defaults to ACT/365F.

    Returns
    -------
    float
        Time to maturity in years.

    Raises
    ------
    ValueError
        If the tenor is negative overall, i.e. resolves to a date before ``anchor``.

    Examples
    --------
    One year from a non-leap-year start is exactly one:

    >>> tenor_to_years(date(2026, 1, 1), years=1)
    1.0

    Three months from January is 90 days; from June it is 92, so the same nominal
    tenor is not the same amount of time:

    >>> round(tenor_to_years(date(2026, 1, 1), months=3), 6)
    0.246575
    >>> round(tenor_to_years(date(2026, 6, 1), months=3), 6)
    0.252055

    Components compose in the documented order:

    >>> round(tenor_to_years(date(2026, 1, 1), years=1, months=6, days=10), 6)
    1.523288

    A zero tenor is legal and marks an expired contract:

    >>> tenor_to_years(date(2026, 1, 1))
    0.0
    """
    expiry = add_months(anchor, 12 * years + months)
    expiry = date.fromordinal(expiry.toordinal() + 7 * weeks + days)
    return year_fraction(anchor, expiry, basis)
