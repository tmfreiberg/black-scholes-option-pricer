"""Tests for day count conventions and tenor arithmetic.

The behaviour under test is that a tenor resolves against a real calendar: spans that
are genuinely equal must agree, and the same nominal tenor measured from different
anchors must not.
"""

from __future__ import annotations

from datetime import date

import pytest
from hypothesis import given
from hypothesis import strategies as st

from bsm.daycount import DayCount, add_months, tenor_to_years, year_fraction


class TestYearFraction:
    def test_full_common_year(self) -> None:
        assert year_fraction(date(2026, 1, 1), date(2027, 1, 1)) == 1.0

    def test_leap_year_is_longer(self) -> None:
        assert year_fraction(date(2028, 1, 1), date(2029, 1, 1)) == 366.0 / 365.0

    def test_same_day_is_zero(self) -> None:
        assert year_fraction(date(2026, 5, 5), date(2026, 5, 5)) == 0.0

    def test_act_360_basis(self) -> None:
        assert year_fraction(date(2026, 1, 1), date(2027, 1, 1), DayCount.ACT_360) == (
            365.0 / 360.0
        )

    def test_reversed_dates_raise(self) -> None:
        with pytest.raises(ValueError, match="must not precede"):
            year_fraction(date(2027, 1, 1), date(2026, 1, 1))

    @given(
        st.dates(min_value=date(1990, 1, 1), max_value=date(2050, 1, 1)),
        st.integers(min_value=0, max_value=10_000),
    )
    def test_additivity(self, start: date, offset: int) -> None:
        """Time is additive: measuring in two hops equals measuring in one."""
        middle = date.fromordinal(start.toordinal() + offset // 2)
        end = date.fromordinal(start.toordinal() + offset)
        total = year_fraction(start, end)
        parts = year_fraction(start, middle) + year_fraction(middle, end)
        assert total == pytest.approx(parts, abs=1e-12)


class TestAddMonths:
    def test_simple(self) -> None:
        assert add_months(date(2026, 1, 15), 3) == date(2026, 4, 15)

    def test_year_rollover(self) -> None:
        assert add_months(date(2026, 11, 10), 3) == date(2027, 2, 10)

    def test_negative(self) -> None:
        assert add_months(date(2026, 3, 10), -15) == date(2024, 12, 10)

    def test_zero_is_identity(self) -> None:
        assert add_months(date(2026, 7, 31), 0) == date(2026, 7, 31)

    @pytest.mark.parametrize(
        ("start", "months", "expected"),
        [
            (date(2026, 1, 31), 1, date(2026, 2, 28)),
            (date(2028, 1, 31), 1, date(2028, 2, 29)),
            (date(2026, 3, 31), 1, date(2026, 4, 30)),
            (date(2027, 8, 31), 6, date(2028, 2, 29)),
        ],
    )
    def test_month_end_clamping(self, start: date, months: int, expected: date) -> None:
        assert add_months(start, months) == expected

    def test_clamping_is_lossy(self) -> None:
        """Documented asymmetry: clamping is lossy, so this is not a group action."""
        assert add_months(add_months(date(2026, 1, 31), 1), -1) == date(2026, 1, 28)

    @given(
        st.dates(min_value=date(1990, 1, 1), max_value=date(2050, 1, 1)),
        st.integers(min_value=-240, max_value=240),
    )
    def test_lands_in_expected_month(self, start: date, months: int) -> None:
        result = add_months(start, months)
        zero_based = start.month - 1 + months
        assert result.year == start.year + zero_based // 12
        assert result.month == zero_based % 12 + 1
        assert result.day <= start.day


class TestTenorToYears:
    def test_one_year(self) -> None:
        assert tenor_to_years(date(2026, 1, 1), years=1) == 1.0

    def test_zero_tenor(self) -> None:
        assert tenor_to_years(date(2026, 1, 1)) == 0.0

    def test_three_months_depends_on_anchor(self) -> None:
        """The point of doing calendar arithmetic: Q1 and Q3 are different lengths."""
        winter = tenor_to_years(date(2026, 1, 1), months=3)
        summer = tenor_to_years(date(2026, 6, 1), months=3)
        assert winter != summer
        assert winter == 90.0 / 365.0
        assert summer == 92.0 / 365.0

    def test_components_compose(self) -> None:
        anchor = date(2026, 1, 1)
        combined = tenor_to_years(anchor, years=1, months=6, days=10)
        assert combined == year_fraction(anchor, date(2027, 7, 11))

    def test_weeks_are_seven_days(self) -> None:
        anchor = date(2026, 1, 1)
        assert tenor_to_years(anchor, weeks=4) == tenor_to_years(anchor, days=28)

    def test_negative_tenor_raises(self) -> None:
        with pytest.raises(ValueError, match="must not precede"):
            tenor_to_years(date(2026, 1, 1), months=-1)
