"""Black-Scholes-Merton option pricing, Greeks, and implied volatility.

A small, typed, dependency-light library for European options. NumPy is the only
runtime dependency; the normal CDF and the root-finder are implemented here rather
than pulled from SciPy, so the package loads quickly under Pyodide and can run inside
the reader's browser in the accompanying document.

Examples
--------
The validated entry point takes two value objects:

>>> from bsm import Contract, Market, OptionKind, price
>>> contract = Contract(strike=100.0, maturity=1.0, kind=OptionKind.CALL)
>>> market = Market(spot=100.0, rate=0.05, volatility=0.2)
>>> round(price(contract, market), 4)
10.4506

The formulas underneath accept arrays, which is how sensitivity surfaces are built:

>>> import numpy as np
>>> from bsm import call_price
>>> call_price(np.array([90.0, 110.0]), 100.0, 0.05, 0.0, 1.0, 0.2).round(4)
array([ 5.0912, 17.663 ])
"""

from bsm.bounds import parity_residual, price_bounds
from bsm.daycount import DayCount, add_months, tenor_to_years, year_fraction
from bsm.greeks import Greeks, delta, gamma, greeks, rho, theta, vega
from bsm.grids import Grid, price_surface
from bsm.implied import (
    ImpliedVolatilityError,
    implied_volatility,
    implied_volatility_for,
)
from bsm.merton import JumpParameters, jump_price, volatility_smile
from bsm.normal import cdf, pdf
from bsm.pricing import call_price, d1_d2, forward_price, price, put_price
from bsm.roots import brentq
from bsm.store import Scenario, ScenarioStore, fingerprint
from bsm.types import Contract, Market, OptionKind

__version__ = "0.1.0"

__all__ = [
    "Contract",
    "DayCount",
    "Greeks",
    "Grid",
    "ImpliedVolatilityError",
    "JumpParameters",
    "Market",
    "OptionKind",
    "Scenario",
    "ScenarioStore",
    "__version__",
    "add_months",
    "brentq",
    "call_price",
    "cdf",
    "d1_d2",
    "delta",
    "fingerprint",
    "forward_price",
    "gamma",
    "greeks",
    "implied_volatility",
    "implied_volatility_for",
    "jump_price",
    "parity_residual",
    "pdf",
    "price",
    "price_bounds",
    "price_surface",
    "put_price",
    "rho",
    "tenor_to_years",
    "theta",
    "vega",
    "volatility_smile",
    "year_fraction",
]
