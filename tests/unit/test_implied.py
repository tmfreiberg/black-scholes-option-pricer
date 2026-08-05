"""Tests for implied volatility."""

from __future__ import annotations

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from bsm import implied as implied_module
from bsm.bounds import price_bounds
from bsm.greeks import vega
from bsm.implied import (
    ImpliedVolatilityError,
    NoMaturityRemaining,
    PriceOutsideBounds,
    SolverFailed,
    implied_volatility,
    implied_volatility_for,
)
from bsm.pricing import call_price, price, put_price
from bsm.types import Contract, Market, OptionKind


class TestRoundTrip:
    @pytest.mark.parametrize("kind", [OptionKind.CALL, OptionKind.PUT])
    @pytest.mark.parametrize("volatility", [0.01, 0.05, 0.2, 0.5, 1.0, 2.5, 5.0])
    def test_recovers_the_volatility_that_produced_the_price(
        self, kind: OptionKind, volatility: float
    ) -> None:
        formula = call_price if kind is OptionKind.CALL else put_price
        observed = float(formula(100.0, 100.0, 0.05, 0.0, 1.0, volatility))
        recovered = implied_volatility(observed, kind, 100.0, 100.0, 0.05, 0.0, 1.0)
        assert recovered == pytest.approx(volatility, abs=1e-10)

    @pytest.mark.parametrize(
        ("spot", "strike", "maturity"),
        [
            (100.0, 100.0, 1.0),
            (100.0, 150.0, 0.25),  # far out of the money: vega is small
            (100.0, 50.0, 0.25),  # far in the money: vega is small
            (100.0, 100.0, 0.01),  # nearly expired
            (100.0, 100.0, 10.0),  # long dated
        ],
    )
    def test_round_trip_where_vega_is_small(
        self, spot: float, strike: float, maturity: float
    ) -> None:
        """The regime where a Newton iteration would be unreliable."""
        observed = float(call_price(spot, strike, 0.05, 0.0, maturity, 0.3))
        recovered = implied_volatility(
            observed, OptionKind.CALL, spot, strike, 0.05, 0.0, maturity
        )
        assert recovered == pytest.approx(0.3, abs=1e-8)

    def test_with_dividend_yield(self) -> None:
        observed = float(put_price(90.0, 100.0, 0.03, 0.02, 0.5, 0.35))
        assert implied_volatility(
            observed, OptionKind.PUT, 90.0, 100.0, 0.03, 0.02, 0.5
        ) == pytest.approx(0.35, abs=1e-10)

    @settings(max_examples=200, deadline=None)
    @given(
        st.floats(min_value=50.0, max_value=200.0),
        st.floats(min_value=50.0, max_value=200.0),
        st.floats(min_value=0.05, max_value=5.0),
        st.floats(min_value=0.02, max_value=2.0),
    )
    def test_recovered_volatility_always_reprices(
        self, spot: float, strike: float, maturity: float, volatility: float
    ) -> None:
        """The property that holds unconditionally: the answer reproduces the price.

        Stronger claims fail where the price underflows to its lower bound, because the
        forward computation has already destroyed the information. This one does not.
        """
        observed = float(call_price(spot, strike, 0.04, 0.01, maturity, volatility))
        recovered = implied_volatility(
            observed, OptionKind.CALL, spot, strike, 0.04, 0.01, maturity
        )
        repriced = float(call_price(spot, strike, 0.04, 0.01, maturity, recovered))
        assert repriced == pytest.approx(observed, abs=1e-10 * spot)

    @settings(max_examples=200, deadline=None)
    @given(
        st.floats(min_value=50.0, max_value=200.0),
        st.floats(min_value=50.0, max_value=200.0),
        st.floats(min_value=0.05, max_value=5.0),
        st.floats(min_value=0.02, max_value=2.0),
    )
    def test_round_trip_where_vega_is_appreciable(
        self, spot: float, strike: float, maturity: float, volatility: float
    ) -> None:
        """Uniqueness needs vega. Below the threshold the inverse is not determined."""
        sensitivity = float(vega(spot, strike, 0.04, 0.01, maturity, volatility))
        assume(sensitivity > 1e-3)
        observed = float(call_price(spot, strike, 0.04, 0.01, maturity, volatility))
        recovered = implied_volatility(
            observed, OptionKind.CALL, spot, strike, 0.04, 0.01, maturity
        )
        assert recovered == pytest.approx(volatility, abs=1e-6)

    def test_price_that_underflows_to_zero_cannot_be_inverted(self) -> None:
        """The genuine limit: when the price rounds to zero, nothing can recover it.

        A call struck at 111.125 with spot 50, a year to run and 2% volatility is worth
        about 1e-322 in exact arithmetic, which is below the smallest double. Its
        computed price is exactly zero, indistinguishable from the price at zero
        volatility, so zero is returned. The information was destroyed by the forward
        computation, not by the solve, and vega at 5e-320 says as much.
        """
        arguments = (50.0, 111.125, 0.04, 0.01, 1.0)
        observed = float(call_price(*arguments, 0.02))
        lower, _ = price_bounds(OptionKind.CALL, *arguments)

        assert observed == lower == 0.0
        assert float(vega(*arguments, 0.02)) < 1e-300
        assert implied_volatility(observed, OptionKind.CALL, *arguments) == 0.0


class TestBoundaries:
    def test_price_on_the_lower_bound_implies_zero(self) -> None:
        lower, _ = price_bounds(OptionKind.CALL, 100.0, 100.0, 0.05, 0.0, 1.0)
        assert (
            implied_volatility(
                float(lower), OptionKind.CALL, 100.0, 100.0, 0.05, 0.0, 1.0
            )
            == 0.0
        )

    def test_price_below_the_lower_bound_raises(self) -> None:
        lower, _ = price_bounds(OptionKind.CALL, 100.0, 100.0, 0.05, 0.0, 1.0)
        with pytest.raises(PriceOutsideBounds, match="below the lower"):
            implied_volatility(
                float(lower) - 0.5, OptionKind.CALL, 100.0, 100.0, 0.05, 0.0, 1.0
            )

    def test_price_at_the_upper_bound_raises(self) -> None:
        _, upper = price_bounds(OptionKind.CALL, 100.0, 100.0, 0.05, 0.0, 1.0)
        with pytest.raises(PriceOutsideBounds, match="not below the upper"):
            implied_volatility(
                float(upper), OptionKind.CALL, 100.0, 100.0, 0.05, 0.0, 1.0
            )

    def test_expired_contract_raises(self) -> None:
        with pytest.raises(NoMaturityRemaining):
            implied_volatility(20.0, OptionKind.CALL, 120.0, 100.0, 0.05, 0.0, 0.0)

    def test_all_failures_are_value_errors(self) -> None:
        """Callers can catch the base class, or ValueError, and cover every case."""
        assert issubclass(PriceOutsideBounds, ImpliedVolatilityError)
        assert issubclass(NoMaturityRemaining, ImpliedVolatilityError)
        assert issubclass(SolverFailed, ImpliedVolatilityError)
        assert issubclass(ImpliedVolatilityError, ValueError)

    def test_failure_message_names_the_bound(self) -> None:
        with pytest.raises(PriceOutsideBounds) as info:
            implied_volatility(101.0, OptionKind.CALL, 100.0, 100.0, 0.05, 0.0, 1.0)
        assert "100" in str(info.value)

    def test_nothing_returns_a_sentinel(self) -> None:
        """Every failure raises; no string, no None, no negative volatility."""
        for bad_price in (-1.0, 1e9):
            with pytest.raises(ImpliedVolatilityError):
                implied_volatility(
                    bad_price, OptionKind.CALL, 100.0, 100.0, 0.05, 0.0, 1.0
                )

    def test_bracket_expansion_gives_up_rather_than_looping(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The cap turns a non-terminating search into a reported failure.

        Unreachable with the real cap of 10,000% volatility, so the cap is lowered to
        make the branch testable. Without it a price inside the bounds that no finite
        volatility reproduces would spin forever.
        """
        monkeypatch.setattr(implied_module, "_MAX_VOLATILITY", 1.5)
        observed = float(call_price(100.0, 100.0, 0.05, 0.0, 1.0, 3.0))
        with pytest.raises(SolverFailed, match="no volatility below"):
            implied_volatility(observed, OptionKind.CALL, 100.0, 100.0, 0.05, 0.0, 1.0)


class TestValidatedEntryPoint:
    def test_matches_the_raw_function(self) -> None:
        contract = Contract(100.0, 1.0, OptionKind.CALL)
        market = Market(100.0, 0.05, 0.28)
        observed = price(contract, market)
        assert implied_volatility_for(observed, contract, market) == pytest.approx(
            0.28, abs=1e-10
        )

    def test_ignores_the_market_volatility(self) -> None:
        """The volatility on the Market object is the unknown, not an input."""
        contract = Contract(100.0, 1.0, OptionKind.CALL)
        observed = price(contract, Market(100.0, 0.05, 0.28))
        from_wrong_market = implied_volatility_for(
            observed, contract, Market(100.0, 0.05, 0.99)
        )
        assert from_wrong_market == pytest.approx(0.28, abs=1e-10)
