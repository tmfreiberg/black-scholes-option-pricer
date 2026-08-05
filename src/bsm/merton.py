r"""Merton's jump-diffusion model, and the smile it produces.

Black-Scholes assumes the asset price moves continuously. Merton's 1976 extension adds
discontinuities: alongside the diffusion, jumps arrive as a Poisson process with
intensity :math:`\lambda`, and each multiplies the price by a lognormal factor
:math:`Y` with :math:`\ln Y \sim N(\mu_J, \delta^2)`.

The model has a closed form, and its shape is the reason this module exists.
Conditional on exactly :math:`n` jumps occurring, the terminal distribution is again
lognormal — with a wider variance and a shifted drift — so the price conditional on
:math:`n` is a Black-Scholes price. The unconditional price is therefore the
Poisson-weighted average of Black-Scholes prices:

.. math::

    C = \sum_{n=0}^{\infty}
        \frac{e^{-\lambda'\tau}(\lambda'\tau)^n}{n!}\,
        C_{\mathrm{BS}}(S, K, r_n, q, \tau, \sigma_n),

.. math::

    \sigma_n^2 = \sigma^2 + \frac{n\delta^2}{\tau}, \qquad
    r_n = r - \lambda k + \frac{n\ln(1+k)}{\tau}, \qquad
    k = e^{\mu_J + \delta^2/2} - 1,

with :math:`\lambda' = \lambda(1+k)`. The compensator :math:`-\lambda k` in each
:math:`r_n` offsets the average jump so that the discounted price stays a martingale;
without it the model would price in a drift and admit arbitrage.

Why this is here
----------------
Not for its own sake. Implied volatility is defined by inverting Black-Scholes at an
observed price, so if Black-Scholes were the true model, every option on one underlying
at one expiry would imply the same :math:`\sigma`. Real markets do not behave that way,
and the failure is systematic rather than noisy.

This module lets that be demonstrated rather than asserted. Generate prices under a
model whose returns are *not* lognormal, invert them through Black-Scholes, and watch
implied volatility vary with strike. The ground truth is known because we produced it,
so the exercise isolates the mechanism: a smile is what appears when the inverting model
is wrong about the shape of the distribution.

That is a claim about mechanism, not about markets. Showing that jumps generate a smile
does not show that jumps are why equity index options exhibit one — stochastic
volatility produces a similar shape, and the empirical question is not settled here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from bsm._numeric import FloatArray, Numeric, as_array, scalarize
from bsm.implied import implied_volatility
from bsm.pricing import call_price, put_price
from bsm.types import OptionKind

__all__ = ["JumpParameters", "jump_price", "volatility_smile"]

# The Poisson tail is truncated once the omitted mass falls below this. At double
# precision nothing beyond it can affect the sum.
_TAIL_TOLERANCE = 1e-16

# A hard ceiling, so a pathological intensity fails loudly rather than looping.
_MAX_TERMS = 2_000


@dataclass(frozen=True, slots=True)
class JumpParameters:
    r"""The three parameters governing the jump component.

    Held together because they are meaningless apart: an intensity without a size
    distribution says nothing, and a size distribution that never fires has no effect.

    Parameters
    ----------
    intensity
        Expected number of jumps per year, :math:`\lambda`. Non-negative. Zero
        recovers Black-Scholes exactly.
    mean
        Mean of the log jump size, :math:`\mu_J`. Negative values make downward jumps
        the norm, which is the empirically relevant case for equity indices.
    volatility
        Standard deviation of the log jump size, :math:`\delta`. Non-negative.

    Raises
    ------
    ValueError
        If ``intensity`` or ``volatility`` is negative, or any field is not finite.

    Examples
    --------
    >>> jumps = JumpParameters(intensity=1.0, mean=-0.10, volatility=0.15)
    >>> round(jumps.expected_relative_size, 6)
    -0.084926

    A negative mean gives an expected jump downward, which is what produces a downward
    sloping smile:

    >>> jumps.expected_relative_size < 0
    True

    No jumps at all is admissible and is the Black-Scholes case:

    >>> JumpParameters(intensity=0.0, mean=0.0, volatility=0.0).intensity
    0.0
    """

    intensity: float
    mean: float
    volatility: float

    def __post_init__(self) -> None:
        """Validate the jump parameters."""
        for name, value in (
            ("intensity", self.intensity),
            ("mean", self.mean),
            ("volatility", self.volatility),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if self.intensity < 0.0:
            raise ValueError(f"intensity must be non-negative, got {self.intensity!r}")
        if self.volatility < 0.0:
            raise ValueError(
                f"volatility must be non-negative, got {self.volatility!r}"
            )

    @property
    def expected_relative_size(self) -> float:
        r"""Return the mean proportional jump, :math:`k = e^{\mu_J + \delta^2/2} - 1`.

        This is what the compensator in each conditional drift cancels.

        Examples
        --------
        A zero-mean *log* jump still has a positive expected *relative* jump, by
        Jensen's inequality:

        >>> round(JumpParameters(1.0, 0.0, 0.2).expected_relative_size, 6)
        0.020201
        """
        return math.exp(self.mean + 0.5 * self.volatility * self.volatility) - 1.0


def _term_count(intensity_time: float) -> int:
    r"""Count the Poisson terms needed before the tail is negligible.

    Parameters
    ----------
    intensity_time
        The Poisson parameter :math:`\lambda'\tau`.

    Returns
    -------
    int
        A term count whose omitted tail mass is below ``_TAIL_TOLERANCE``.

    Raises
    ------
    ValueError
        If no count below ``_MAX_TERMS`` suffices.

    Examples
    --------
    A quiet process needs very few terms:

    >>> _term_count(0.0)
    1
    >>> _term_count(1.0) < 20
    True

    A busy one needs more, but the growth is slow — the Poisson distribution
    concentrates:

    >>> _term_count(100.0) < 200
    True
    """
    if intensity_time <= 0.0:
        return 1

    # Accumulate mass until the terms stop mattering. Done in log space, since
    # (lambda*tau)**n overflows long before the weight itself becomes small.
    #
    # The exit condition is the individual weight, not the accumulated mass. Summing
    # doubles that approach 1 leaves a residue of order machine epsilon, so waiting for
    # `1 - cumulative` to fall below 1e-16 waits forever. Past the mode the weights
    # decrease monotonically, so a negligible weight there bounds the whole tail.
    cumulative = 0.0
    for count in range(1, _MAX_TERMS + 1):
        index = count - 1
        log_weight = (
            -intensity_time + index * math.log(intensity_time) - math.lgamma(index + 1)
        )
        weight = math.exp(log_weight)
        cumulative += weight
        if index > intensity_time and weight < _TAIL_TOLERANCE:
            return count
    raise ValueError(
        f"Poisson series has not converged after {_MAX_TERMS} terms at "
        f"lambda*tau = {intensity_time:g}"
    )


def jump_price(
    kind: OptionKind,
    spot: Numeric,
    strike: Numeric,
    rate: Numeric,
    dividend_yield: Numeric,
    maturity: Numeric,
    volatility: Numeric,
    jumps: JumpParameters,
) -> Numeric:
    r"""Price a European option under Merton jump-diffusion.

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
    volatility
        Volatility of the *diffusion* component only, as a decimal. Total volatility is
        larger, because the jumps contribute variance of their own.
    jumps
        The jump intensity and size distribution.

    Returns
    -------
    Numeric
        The option value, summed over enough Poisson terms that the omitted tail is
        below double precision.

    Notes
    -----
    Each term calls the Black-Scholes pricer, so the whole sum is vectorised: an array
    of strikes costs one pass per term rather than one pass per strike. Fifty terms
    covers every case reached here.

    Examples
    --------
    With no jumps the model is Black-Scholes, to the last bit:

    >>> quiet = JumpParameters(intensity=0.0, mean=0.0, volatility=0.0)
    >>> jumped = jump_price(
    ...     OptionKind.CALL, 100.0, 100.0, 0.05, 0.0, 1.0, 0.2, quiet
    ... )
    >>> plain = call_price(100.0, 100.0, 0.05, 0.0, 1.0, 0.2)
    >>> jumped == plain
    True

    Adding jumps adds variance, so an at-the-money option is worth more:

    >>> jumps = JumpParameters(intensity=1.0, mean=-0.10, volatility=0.15)
    >>> with_jumps = jump_price(
    ...     OptionKind.CALL, 100.0, 100.0, 0.05, 0.0, 1.0, 0.2, jumps
    ... )
    >>> with_jumps > plain
    True

    Put-call parity is model-free and survives:

    >>> call = jump_price(OptionKind.CALL, 100.0, 100.0, 0.05, 0.0, 1.0, 0.2, jumps)
    >>> put = jump_price(OptionKind.PUT, 100.0, 100.0, 0.05, 0.0, 1.0, 0.2, jumps)
    >>> import math
    >>> abs((call - put) - (100.0 - 100.0 * math.exp(-0.05))) < 1e-10
    True
    """
    arguments = (spot, strike, rate, dividend_yield, maturity, volatility)
    s, k, r, q, t, v = (as_array(argument) for argument in arguments)

    relative_size = jumps.expected_relative_size
    compensated_intensity = jumps.intensity * (1.0 + relative_size)
    intensity_time = compensated_intensity * t

    # sigma_n and r_n both divide by tau. At expiry the n >= 1 weights are zero, so the
    # terms vanish regardless of what is substituted here — but 0 * inf is NaN, so a
    # finite stand-in is required rather than merely tidy.
    safe_maturity = np.where(t > 0.0, t, 1.0)
    safe_intensity_time = np.where(intensity_time > 0.0, intensity_time, 1.0)

    jump_variance = jumps.volatility * jumps.volatility
    log_growth = math.log1p(relative_size) if relative_size > -1.0 else -math.inf
    drift_offset = r - jumps.intensity * relative_size

    formula = call_price if kind is OptionKind.CALL else put_price
    total = np.zeros_like(s * k * r * q * t * v, dtype=np.float64)

    for index in range(_term_count(float(np.max(intensity_time)))):
        log_weight = (
            -intensity_time
            + index * np.log(safe_intensity_time)
            - math.lgamma(index + 1)
        )
        weight = np.where(
            intensity_time > 0.0, np.exp(log_weight), 1.0 if index == 0 else 0.0
        )
        conditional_volatility = np.sqrt(v * v + index * jump_variance / safe_maturity)
        conditional_rate = drift_offset + index * log_growth / safe_maturity
        total = total + weight * formula(
            s, k, conditional_rate, q, t, conditional_volatility
        )

    return scalarize(total, *arguments)


def volatility_smile(
    strikes: FloatArray,
    *,
    spot: float,
    rate: float,
    dividend_yield: float,
    maturity: float,
    volatility: float,
    jumps: JumpParameters,
) -> FloatArray:
    r"""Price under jump-diffusion, then invert through Black-Scholes.

    This is the demonstration the module exists for. Prices are generated by a model
    whose returns are not lognormal, and then read back through one that assumes they
    are. The implied volatilities that come out are not constant, and their variation
    with strike is the smile.

    Parameters
    ----------
    strikes
        One-dimensional array of exercise prices, strictly positive.
    spot
        Current price of the underlying, strictly positive.
    rate
        Continuously compounded risk-free rate, as a decimal.
    dividend_yield
        Continuous dividend yield, as a decimal.
    maturity
        Time to expiry in years, strictly positive.
    volatility
        Diffusion volatility used to generate the prices, as a decimal.
    jumps
        The jump intensity and size distribution.

    Returns
    -------
    FloatArray
        Black-Scholes implied volatility at each strike, same shape as ``strikes``.

    Raises
    ------
    ValueError
        If ``strikes`` is not one-dimensional.
    bsm.implied.ImpliedVolatilityError
        If any generated price cannot be inverted. This should not happen for sensible
        parameters, and indicates a numerical problem rather than a bad quote.

    Notes
    -----
    Calls are used throughout. Put-call parity means the choice cannot affect the
    result: two prices satisfying parity imply the same volatility by construction, and
    the test suite checks that rather than assuming it.

    The inversion is where the two models meet, and it is the whole argument. If
    Black-Scholes described the price-generating process, the returned array would be
    constant. Every departure from flatness measures a disagreement about the shape of
    the terminal distribution.

    Examples
    --------
    With no jumps, the generating model *is* Black-Scholes and the smile is flat:

    >>> import numpy as np
    >>> strikes = np.array([80.0, 100.0, 120.0])
    >>> quiet = JumpParameters(0.0, 0.0, 0.0)
    >>> flat = volatility_smile(
    ...     strikes, spot=100.0, rate=0.05, dividend_yield=0.0,
    ...     maturity=1.0, volatility=0.2, jumps=quiet,
    ... )
    >>> bool(np.allclose(flat, 0.2, atol=1e-9))
    True

    With downward jumps it is not, and low strikes imply the most:

    >>> jumps = JumpParameters(intensity=1.0, mean=-0.10, volatility=0.15)
    >>> smile = volatility_smile(
    ...     strikes, spot=100.0, rate=0.05, dividend_yield=0.0,
    ...     maturity=1.0, volatility=0.2, jumps=jumps,
    ... )
    >>> bool(smile[0] > smile[1] > smile[2])
    True

    Every implied volatility exceeds the diffusion volatility that generated it, because
    the jumps contribute variance the inverting model can only attribute to sigma:

    >>> bool(np.all(smile > 0.2))
    True
    """
    if strikes.ndim != 1:
        raise ValueError(f"strikes must be one-dimensional, got {strikes.ndim}")

    prices = jump_price(
        OptionKind.CALL,
        spot,
        strikes,
        rate,
        dividend_yield,
        maturity,
        volatility,
        jumps,
    )
    return np.array(
        [
            implied_volatility(
                float(price),
                OptionKind.CALL,
                spot,
                float(strike),
                rate,
                dividend_yield,
                maturity,
            )
            for price, strike in zip(np.asarray(prices), strikes, strict=True)
        ],
        dtype=np.float64,
    )
