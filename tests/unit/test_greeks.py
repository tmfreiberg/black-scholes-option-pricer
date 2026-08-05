"""Tests for the analytic Greeks.

The central check is `TestAgainstFiniteDifferences`: each closed form is compared to a
central difference of the pricer it claims to differentiate. That catches an algebra
error in a way no reference table can, because it tests the two implementations against
each other rather than both against a remembered number.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from bsm.greeks import Greeks, delta, gamma, greeks, rho, theta, vega
from bsm.pricing import call_price, put_price
from bsm.types import Contract, Market, OptionKind

CASES = [
    (100.0, 100.0, 0.05, 0.00, 1.00, 0.20),
    (90.0, 100.0, 0.03, 0.01, 0.50, 0.35),
    (130.0, 100.0, 0.07, 0.02, 2.00, 0.15),
    (100.0, 150.0, 0.01, 0.00, 0.25, 0.45),
    (50.0, 40.0, -0.01, 0.03, 1.50, 0.60),
]
KINDS = [OptionKind.CALL, OptionKind.PUT]


def price_of(kind: OptionKind) -> Callable[..., float]:
    formula = call_price if kind is OptionKind.CALL else put_price

    def evaluate(s: float, k: float, r: float, q: float, t: float, v: float) -> float:
        return float(formula(s, k, r, q, t, v))

    return evaluate


def central_difference(f: Callable[[float], float], x: float, step: float) -> float:
    return (f(x + step) - f(x - step)) / (2.0 * step)


def second_central_difference(
    f: Callable[[float], float], x: float, step: float
) -> float:
    return (f(x + step) - 2.0 * f(x) + f(x - step)) / (step * step)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(("s", "k", "r", "q", "t", "v"), CASES)
class TestAgainstFiniteDifferences:
    def test_delta(
        self,
        kind: OptionKind,
        s: float,
        k: float,
        r: float,
        q: float,
        t: float,
        v: float,
    ) -> None:
        price = price_of(kind)
        numeric = central_difference(lambda x: price(x, k, r, q, t, v), s, s * 1e-5)
        assert delta(kind, s, k, r, q, t, v) == pytest.approx(numeric, rel=1e-6)

    def test_gamma(
        self,
        kind: OptionKind,
        s: float,
        k: float,
        r: float,
        q: float,
        t: float,
        v: float,
    ) -> None:
        price = price_of(kind)
        numeric = second_central_difference(
            lambda x: price(x, k, r, q, t, v), s, s * 1e-3
        )
        assert gamma(s, k, r, q, t, v) == pytest.approx(numeric, rel=1e-4)

    def test_vega(
        self,
        kind: OptionKind,
        s: float,
        k: float,
        r: float,
        q: float,
        t: float,
        v: float,
    ) -> None:
        price = price_of(kind)
        numeric = central_difference(lambda x: price(s, k, r, q, t, x), v, 1e-6)
        assert vega(s, k, r, q, t, v) == pytest.approx(numeric, rel=1e-6)

    def test_rho(
        self,
        kind: OptionKind,
        s: float,
        k: float,
        r: float,
        q: float,
        t: float,
        v: float,
    ) -> None:
        price = price_of(kind)
        numeric = central_difference(lambda x: price(s, k, x, q, t, v), r, 1e-6)
        assert rho(kind, s, k, r, q, t, v) == pytest.approx(numeric, rel=1e-6)

    def test_theta_has_the_market_sign(
        self,
        kind: OptionKind,
        s: float,
        k: float,
        r: float,
        q: float,
        t: float,
        v: float,
    ) -> None:
        """Theta is minus the derivative with respect to time *remaining*.

        Differencing the `maturity` argument gives dV/dtau; theta is dV/dt. Getting
        this backwards is the most common sign error in Greek implementations, so the
        relationship is asserted rather than assumed.
        """
        price = price_of(kind)
        numeric = central_difference(lambda x: price(s, k, r, q, x, v), t, 1e-6)
        assert theta(kind, s, k, r, q, t, v) == pytest.approx(-numeric, rel=1e-5)


class TestKnownValues:
    def test_at_the_money_call_delta(self) -> None:
        assert delta(OptionKind.CALL, 100.0, 100.0, 0.05, 0.0, 1.0, 0.2) == (
            pytest.approx(0.6368306511756191, abs=1e-12)
        )

    def test_gamma_and_vega_do_not_depend_on_kind(self) -> None:
        """Parity is linear in spot, so it drops out of both."""
        args = (100.0, 100.0, 0.05, 0.0, 1.0, 0.2)
        call_curvature = second_central_difference(
            lambda x: float(call_price(x, 100.0, 0.05, 0.0, 1.0, 0.2)), 100.0, 0.1
        )
        put_curvature = second_central_difference(
            lambda x: float(put_price(x, 100.0, 0.05, 0.0, 1.0, 0.2)), 100.0, 0.1
        )
        assert call_curvature == pytest.approx(put_curvature, rel=1e-8)
        assert gamma(*args) == pytest.approx(call_curvature, rel=1e-4)

    def test_delta_difference_is_the_dividend_discount(self) -> None:
        """Differentiating put-call parity: dC/dS - dP/dS = exp(-q*tau)."""
        args = (100.0, 100.0, 0.05, 0.03, 1.0, 0.2)
        difference = delta(OptionKind.CALL, *args) - delta(OptionKind.PUT, *args)
        assert difference == pytest.approx(np.exp(-0.03), abs=1e-12)


class TestBoundaries:
    def test_expired_option_has_no_gamma_or_vega(self) -> None:
        assert gamma(100.0, 100.0, 0.05, 0.0, 0.0, 0.2) == 0.0
        assert vega(100.0, 100.0, 0.05, 0.0, 0.0, 0.2) == 0.0

    def test_zero_volatility_has_no_gamma(self) -> None:
        assert gamma(100.0, 100.0, 0.05, 0.0, 1.0, 0.0) == 0.0

    def test_expired_option_has_no_rho(self) -> None:
        assert rho(OptionKind.CALL, 100.0, 100.0, 0.05, 0.0, 0.0, 0.2) == 0.0

    def test_expired_deltas_are_zero_or_one(self) -> None:
        assert delta(OptionKind.CALL, 120.0, 100.0, 0.05, 0.0, 0.0, 0.2) == 1.0
        assert delta(OptionKind.CALL, 80.0, 100.0, 0.05, 0.0, 0.0, 0.2) == 0.0
        assert delta(OptionKind.PUT, 80.0, 100.0, 0.05, 0.0, 0.0, 0.2) == -1.0

    def test_no_nan_at_the_boundaries(self) -> None:
        for t, v in [(0.0, 0.2), (1.0, 0.0), (0.0, 0.0)]:
            for value in (
                delta(OptionKind.CALL, 100.0, 100.0, 0.05, 0.0, t, v),
                gamma(100.0, 100.0, 0.05, 0.0, t, v),
                vega(100.0, 100.0, 0.05, 0.0, t, v),
                theta(OptionKind.CALL, 100.0, 100.0, 0.05, 0.0, t, v),
                rho(OptionKind.CALL, 100.0, 100.0, 0.05, 0.0, t, v),
            ):
                assert np.isfinite(value)


class TestVectorisation:
    def test_greeks_broadcast(self) -> None:
        spots = np.linspace(80.0, 120.0, 9)
        result = delta(OptionKind.CALL, spots, 100.0, 0.05, 0.0, 1.0, 0.2)
        assert isinstance(result, np.ndarray)
        assert result.shape == (9,)
        assert np.all(np.diff(result) > 0.0)

    def test_array_matches_scalar(self) -> None:
        spots = np.linspace(80.0, 120.0, 9)
        vectorised = gamma(spots, 100.0, 0.05, 0.0, 1.0, 0.2)
        elementwise = np.array(
            [gamma(float(s), 100.0, 0.05, 0.0, 1.0, 0.2) for s in spots]
        )
        assert np.allclose(vectorised, elementwise, rtol=0.0, atol=1e-18)


class TestAggregate:
    def test_greeks_matches_individual_functions(self) -> None:
        contract = Contract(100.0, 1.0, OptionKind.CALL)
        market = Market(100.0, 0.05, 0.2)
        args = (100.0, 100.0, 0.05, 0.0, 1.0, 0.2)
        result = greeks(contract, market)
        assert result.delta == delta(OptionKind.CALL, *args)
        assert result.gamma == gamma(*args)
        assert result.vega == vega(*args)
        assert result.theta == theta(OptionKind.CALL, *args)
        assert result.rho == rho(OptionKind.CALL, *args)

    def test_returns_plain_floats(self) -> None:
        result = greeks(Contract(100.0, 1.0, OptionKind.PUT), Market(100.0, 0.05, 0.2))
        assert all(
            type(getattr(result, field)) is float
            for field in ("delta", "gamma", "vega", "theta", "rho")
        )

    def test_is_frozen(self) -> None:
        result = greeks(Contract(100.0, 1.0, OptionKind.CALL), Market(100.0, 0.05, 0.2))
        assert isinstance(result, Greeks)
        assert not hasattr(result, "__dict__")
