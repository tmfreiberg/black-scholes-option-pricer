"""Tests for Merton jump-diffusion and the smile it generates.

Three things carry the module and are tested hardest: that zero intensity reduces
exactly to Black-Scholes, that the model-free relationships survive a model with jumps
in it, and that the smile has the shape the jump parameters imply rather than merely
being non-flat.
"""

from __future__ import annotations

import math

import mpmath as mp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bsm import merton
from bsm.bounds import price_bounds
from bsm.greeks import vega
from bsm.merton import JumpParameters, jump_price, volatility_smile
from bsm.pricing import call_price, put_price
from bsm.types import OptionKind

QUIET = JumpParameters(intensity=0.0, mean=0.0, volatility=0.0)
DOWNWARD = JumpParameters(intensity=1.0, mean=-0.10, volatility=0.15)
SYMMETRIC = JumpParameters(intensity=1.0, mean=0.0, volatility=0.20)

TERMS = (100.0, 100.0, 0.05, 0.0, 1.0, 0.20)


class TestJumpParameters:
    def test_expected_relative_size(self) -> None:
        jumps = JumpParameters(1.0, -0.10, 0.15)
        assert jumps.expected_relative_size == pytest.approx(
            math.exp(-0.10 + 0.5 * 0.15**2) - 1.0
        )

    def test_zero_log_mean_gives_positive_relative_mean(self) -> None:
        """Jensen: E[exp(X)] > exp(E[X]) whenever X has variance."""
        assert JumpParameters(1.0, 0.0, 0.2).expected_relative_size > 0.0

    def test_zero_volatility_and_mean_gives_no_jump(self) -> None:
        assert JumpParameters(5.0, 0.0, 0.0).expected_relative_size == 0.0

    @pytest.mark.parametrize(
        ("intensity", "mean", "volatility"),
        [(-1.0, 0.0, 0.1), (1.0, 0.0, -0.1)],
    )
    def test_negative_parameters_rejected(
        self, intensity: float, mean: float, volatility: float
    ) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            JumpParameters(intensity, mean, volatility)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_non_finite_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            JumpParameters(bad, 0.0, 0.1)

    def test_immutable(self) -> None:
        jumps = JumpParameters(1.0, 0.0, 0.1)
        with pytest.raises(AttributeError):
            jumps.intensity = 2.0  # type: ignore[misc]  # noqa: PGH003

    def test_hashable_and_comparable_by_value(self) -> None:
        assert JumpParameters(1.0, 0.0, 0.1) == JumpParameters(1.0, 0.0, 0.1)
        assert len({JumpParameters(1.0, 0.0, 0.1), JumpParameters(1.0, 0.0, 0.1)}) == 1

    def test_differs_in_each_field(self) -> None:
        base = JumpParameters(1.0, 0.0, 0.1)
        assert base != JumpParameters(2.0, 0.0, 0.1)
        assert base != JumpParameters(1.0, 0.1, 0.1)
        assert base != JumpParameters(1.0, 0.0, 0.2)

    def test_repr_round_trips(self) -> None:
        jumps = JumpParameters(1.0, -0.1, 0.15)
        assert eval(repr(jumps)) == jumps  # noqa: S307


class TestReductionToBlackScholes:
    """Zero intensity must give back the base model, bit for bit."""

    @pytest.mark.parametrize("kind", [OptionKind.CALL, OptionKind.PUT])
    @pytest.mark.parametrize(
        ("s", "k", "r", "q", "t", "v"),
        [
            (100.0, 100.0, 0.05, 0.00, 1.00, 0.20),
            (90.0, 110.0, 0.02, 0.01, 0.25, 0.45),
            (150.0, 100.0, 0.07, 0.03, 3.00, 0.15),
        ],
    )
    def test_zero_intensity_is_exactly_black_scholes(
        self,
        kind: OptionKind,
        s: float,
        k: float,
        r: float,
        q: float,
        t: float,
        v: float,
    ) -> None:
        formula = call_price if kind is OptionKind.CALL else put_price
        assert jump_price(kind, s, k, r, q, t, v, QUIET) == formula(s, k, r, q, t, v)

    def test_zero_jump_size_recovers_black_scholes_to_rounding(self) -> None:
        """Jumps that arrive but do nothing leave the price unchanged, near enough.

        Not bitwise, unlike the zero-intensity case above, and the difference is
        instructive. With no jumps at all a single Poisson term contributes and the sum
        is one Black-Scholes evaluation. With frequent jumps of zero size, thirty-odd
        identical evaluations are summed against weights that total one only to within
        machine epsilon, so the last digit moves. Demanding equality here would be
        demanding that floating-point addition be associative.
        """
        inert = JumpParameters(intensity=5.0, mean=0.0, volatility=0.0)
        assert jump_price(OptionKind.CALL, *TERMS, inert) == pytest.approx(
            call_price(*TERMS), rel=1e-14
        )


class TestModelFreeRelationships:
    """Arbitrage arguments do not know about jumps, so these must survive."""

    @settings(max_examples=50, deadline=None)
    @given(
        st.floats(min_value=50.0, max_value=200.0),
        st.floats(min_value=50.0, max_value=200.0),
        st.floats(min_value=0.05, max_value=3.0),
        st.floats(min_value=0.05, max_value=1.0),
        st.floats(min_value=0.0, max_value=3.0),
        st.floats(min_value=-0.3, max_value=0.3),
    )
    def test_put_call_parity(
        self,
        spot: float,
        strike: float,
        maturity: float,
        volatility: float,
        intensity: float,
        mean: float,
    ) -> None:
        jumps = JumpParameters(intensity, mean, 0.15)
        terms = (spot, strike, 0.04, 0.01, maturity, volatility)
        call = float(jump_price(OptionKind.CALL, *terms, jumps))
        put = float(jump_price(OptionKind.PUT, *terms, jumps))
        expected = spot * math.exp(-0.01 * maturity) - strike * math.exp(
            -0.04 * maturity
        )
        assert call - put == pytest.approx(expected, abs=1e-8, rel=1e-10)

    @settings(max_examples=50, deadline=None)
    @given(
        st.floats(min_value=50.0, max_value=200.0),
        st.floats(min_value=50.0, max_value=200.0),
        st.floats(min_value=0.05, max_value=3.0),
        st.floats(min_value=0.0, max_value=3.0),
    )
    def test_no_arbitrage_bounds(
        self, spot: float, strike: float, maturity: float, intensity: float
    ) -> None:
        jumps = JumpParameters(intensity, -0.1, 0.15)
        terms = (spot, strike, 0.04, 0.01, maturity, 0.2)
        lower, upper = price_bounds(OptionKind.CALL, *terms[:5])
        value = float(jump_price(OptionKind.CALL, *terms, jumps))
        assert float(lower) - 1e-9 <= value <= float(upper) + 1e-9

    def test_expiry_gives_the_payoff(self) -> None:
        """Jumps cannot happen in zero time, so the price is the payoff."""
        assert (
            jump_price(OptionKind.CALL, 120.0, 100.0, 0.05, 0.0, 0.0, 0.2, DOWNWARD)
            == 20.0
        )
        assert (
            jump_price(OptionKind.PUT, 80.0, 100.0, 0.05, 0.0, 0.0, 0.2, DOWNWARD)
            == 20.0
        )


class TestSeriesConvergence:
    def test_matches_a_high_precision_sum(self) -> None:
        """The series against the same series evaluated in 40-digit arithmetic."""
        spot, strike, rate, dividend, maturity, volatility = TERMS
        jumps = DOWNWARD

        with mp.workdps(40):
            k = mp.exp(mp.mpf("-0.10") + mp.mpf("0.15") ** 2 / 2) - 1
            lam = mp.mpf(jumps.intensity)
            lam_prime = lam * (1 + k)
            tau = mp.mpf(maturity)
            total = mp.mpf(0)
            for n in range(120):
                weight = (
                    mp.exp(-lam_prime * tau) * (lam_prime * tau) ** n / mp.factorial(n)
                )
                sigma_n = mp.sqrt(
                    mp.mpf(volatility) ** 2 + n * mp.mpf("0.15") ** 2 / tau
                )
                rate_n = mp.mpf(rate) - lam * k + n * mp.log(1 + k) / tau
                d1 = (
                    mp.log(mp.mpf(spot) / strike) + (rate_n + sigma_n**2 / 2) * tau
                ) / (sigma_n * mp.sqrt(tau))
                d2 = d1 - sigma_n * mp.sqrt(tau)

                def phi(x: mp.mpf) -> mp.mpf:
                    return mp.erfc(-x / mp.sqrt(2)) / 2

                total += weight * (
                    mp.mpf(spot) * phi(d1)
                    - mp.mpf(strike) * mp.exp(-rate_n * tau) * phi(d2)
                )
            exact = float(total)

        computed = float(jump_price(OptionKind.CALL, *TERMS, jumps))
        assert computed == pytest.approx(exact, abs=1e-10)

    def test_high_intensity_still_converges(self) -> None:
        busy = JumpParameters(intensity=50.0, mean=-0.02, volatility=0.05)
        value = float(jump_price(OptionKind.CALL, *TERMS, busy))
        assert math.isfinite(value)
        assert value > 0.0

    def test_gives_up_rather_than_looping(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ceiling turns a non-terminating search into a reported failure.

        Unreachable with the real ceiling of 2,000 terms, since the Poisson
        distribution concentrates and even lambda*tau of 100 needs under 200. The
        ceiling is lowered here so the branch is exercised rather than asserted to be
        unreachable.
        """
        monkeypatch.setattr(merton, "_MAX_TERMS", 5)
        with pytest.raises(ValueError, match="has not converged"):
            merton._term_count(50.0)

    def test_long_maturity_still_converges(self) -> None:
        value = float(
            jump_price(OptionKind.CALL, 100.0, 100.0, 0.05, 0.0, 10.0, 0.2, DOWNWARD)
        )
        assert math.isfinite(value)


class TestVectorisation:
    def test_array_of_strikes_matches_scalar_calls(self) -> None:
        strikes = np.linspace(70.0, 130.0, 13)
        vectorised = jump_price(
            OptionKind.CALL, 100.0, strikes, 0.05, 0.0, 1.0, 0.2, DOWNWARD
        )
        elementwise = np.array(
            [
                float(
                    jump_price(
                        OptionKind.CALL, 100.0, float(k), 0.05, 0.0, 1.0, 0.2, DOWNWARD
                    )
                )
                for k in strikes
            ]
        )
        assert np.allclose(vectorised, elementwise, rtol=0.0, atol=1e-14)

    def test_scalar_returns_float(self) -> None:
        assert isinstance(jump_price(OptionKind.CALL, *TERMS, DOWNWARD), float)


class TestSmile:
    STRIKES = np.linspace(70.0, 130.0, 25)

    def smile(self, jumps: JumpParameters) -> np.ndarray:
        return volatility_smile(
            self.STRIKES,
            spot=100.0,
            rate=0.05,
            dividend_yield=0.0,
            maturity=1.0,
            volatility=0.20,
            jumps=jumps,
        )

    def test_no_jumps_gives_a_flat_smile(self) -> None:
        """The control. Without it, a downward slope proves nothing."""
        assert np.allclose(self.smile(QUIET), 0.20, atol=1e-9)

    def test_downward_jumps_give_a_downward_slope(self) -> None:
        """Low strikes imply the most volatility: the equity index skew."""
        smile = self.smile(DOWNWARD)
        assert smile[0] > smile[-1]
        assert np.all(np.diff(smile) < 0.0)

    def test_symmetric_jumps_give_a_smile_not_a_skew(self) -> None:
        """With no directional bias the curve should turn up at both ends."""
        smile = self.smile(SYMMETRIC)
        middle = len(smile) // 2
        assert smile[0] > smile[middle]
        assert smile[-1] > smile[middle]

    def test_every_implied_volatility_exceeds_the_diffusion(self) -> None:
        """Jumps add variance the inverting model can only attribute to sigma."""
        assert np.all(self.smile(DOWNWARD) > 0.20)

    def test_more_intense_jumps_give_a_steeper_smile(self) -> None:
        mild = self.smile(JumpParameters(0.5, -0.10, 0.15))
        fierce = self.smile(JumpParameters(2.0, -0.10, 0.15))
        assert fierce.max() - fierce.min() > mild.max() - mild.min()

    def test_prices_round_trip_through_the_implied_volatility(self) -> None:
        """The inversion is exact: repricing at the implied volatility recovers it."""
        smile = self.smile(DOWNWARD)
        generated = jump_price(
            OptionKind.CALL, 100.0, self.STRIKES, 0.05, 0.0, 1.0, 0.20, DOWNWARD
        )
        repriced = call_price(100.0, self.STRIKES, 0.05, 0.0, 1.0, smile)
        assert np.allclose(repriced, generated, rtol=0.0, atol=1e-8)

    def test_calls_and_puts_imply_the_same_volatility(self) -> None:
        """Parity forces it, so this checks the pipeline rather than the mathematics."""
        from bsm.implied import implied_volatility

        call_implied = self.smile(DOWNWARD)
        put_prices = jump_price(
            OptionKind.PUT, 100.0, self.STRIKES, 0.05, 0.0, 1.0, 0.20, DOWNWARD
        )
        put_implied = np.array(
            [
                implied_volatility(
                    float(price), OptionKind.PUT, 100.0, float(strike), 0.05, 0.0, 1.0
                )
                for price, strike in zip(
                    np.asarray(put_prices), self.STRIKES, strict=True
                )
            ]
        )
        assert np.allclose(call_implied, put_implied, atol=1e-8)

    def test_vega_is_appreciable_across_the_range(self) -> None:
        """Guards the demonstration: a smile read off ill-conditioned inversions."""
        smile = self.smile(DOWNWARD)
        sensitivity = vega(100.0, self.STRIKES, 0.05, 0.0, 1.0, smile)
        assert np.all(np.asarray(sensitivity) > 1.0)

    def test_two_dimensional_strikes_rejected(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional"):
            volatility_smile(
                self.STRIKES.reshape(-1, 1),
                spot=100.0,
                rate=0.05,
                dividend_yield=0.0,
                maturity=1.0,
                volatility=0.2,
                jumps=DOWNWARD,
            )
