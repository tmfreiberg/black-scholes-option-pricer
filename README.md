# bsm-pricer

European option pricing under Black-Scholes-Merton: prices, Greeks, implied
volatility, and sensitivity surfaces, as a typed and tested Python package.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

NumPy is the only runtime dependency. The standard normal CDF and the root-finder are
implemented here rather than taken from SciPy, so the package is small enough to load
under Pyodide and run in a browser. SciPy appears only as a *test* oracle.

## Use

```python
from bsm import Contract, Market, OptionKind, greeks, price

contract = Contract(strike=100.0, maturity=1.0, kind=OptionKind.CALL)
market = Market(spot=100.0, rate=0.05, volatility=0.2)

price(contract, market)  # 10.450583572185543
greeks(contract, market)  # Greeks(delta=0.6368..., gamma=0.0187..., ...)
```

Implied volatility runs the map backwards:

```python
from bsm import implied_volatility_for

implied_volatility_for(10.4505835, contract, market)  # 0.19999999...
```

Rates and volatility are decimals, never percentages. Time to maturity comes from the
calendar rather than from arithmetic on mixed units:

```python
from datetime import date
from bsm import tenor_to_years

tenor_to_years(date(2026, 8, 5), months=3)  # 0.25205479452054796
```

The formulas underneath take arrays as readily as scalars, and broadcast:

```python
import numpy as np
from bsm import call_price

spots = np.linspace(80, 120, 41).reshape(-1, 1)
vols = np.linspace(0.1, 0.5, 21).reshape(1, -1)
surface = call_price(spots, 100.0, 0.05, 0.0, 1.0, vols)  # (41, 21)
```

That dual behaviour is deliberate, and it is why there is no second implementation of
the mathematics for grids. Public functions convert inputs to arrays, compute
unconditionally, and return a plain float if and only if every argument was scalar.
Overloads make that precise to a type checker, so an array result is indexable without
a cast.

## Design notes

**A validated boundary and an unvalidated core.** `Contract` and `Market` check every
precondition at construction and are frozen, so they can key a cache. The raw formulas
check nothing, because they are called once per grid cell; each documents what it
assumes. Callers choose which layer they are working at.

**Degenerate inputs have no special case.** When total volatility is zero — an expired
contract, or a riskless underlying — `d1` and `d2` are returned as ±∞, chosen by the
sign of forward minus strike. The CDF then evaluates to exactly 1 or 0 and the price
formula collapses to the discounted intrinsic value by itself. This is why the CDF
clips its argument: unclipped, `exp(-inf) * polynomial(inf)` is `0 * inf`, which is
NaN. Branching on maturity instead would not vectorise, and would have to be written
once for the call and again for the put.

The threshold for "zero" is `1e-100`, not `0.0`. A subnormal volatility passes a `> 0`
test and then overflows the division. The property suite found that; inspection did
not.

**The stdlib proves the fast path right.** The shipped CDF is Hart's algorithm, which
vectorises. `math.erfc` is exact but scalar-only, so it serves as the test oracle: the
suite asserts agreement to 1e-15 across the full range, and further asserts that the
tolerance is *tight* — that the true error is within an order of magnitude of the
stated bound, so the number is a measurement rather than a comfortable margin. The same
discipline applies one level up: prices are checked against a 50-digit mpmath
evaluation, and come within 1e-15 of exact per unit of spot.

Brent's method is pinned against `scipy.optimize.brentq` the same way, with SciPy a
test-only dependency that the package never imports. Both solvers are driven to the
tightest tolerances double precision admits, so the comparison measures the algorithms
rather than their default stopping rules.

**Greeks are checked against the pricer they differentiate.** Every closed form is
compared to a central difference of the price function, including the sign of theta,
which is `dV/dt` and therefore the negative of what differencing `maturity` gives.

**The put is not derived from the call.** Computing it directly means put-call parity
is an independent check on two expressions rather than an identity imposed by
construction.

**Prices are clipped into their no-arbitrage bounds.** Far out of the money the two
terms of the formula agree to within rounding and their difference can come out a
denormal below zero; far in the money it can land one ulp below the lower bound. Both
are financially meaningless and both break downstream code entitled to assume the
bounds hold — the implied-volatility solver rejects a price below its own bracket,
which is right for an impossible quote and wrong for a rounding error of 1e-322.

**Sign tests compare signs, they do not multiply.** The textbook bracketing condition
`f(a) * f(b) < 0` is wrong in floating point: two residuals of order 1e-210 have a
product that underflows to zero, so a genuine sign change reads as none. The same
applies inside the inverse-quadratic step, whose denominators are *products* of
residual differences. Both arise routinely when inverting the price of a deep
out-of-the-money option.

**Implied volatility brackets rather than iterating from a guess.** Newton needs no
bracket but divides by vega, which vanishes exactly where implied volatility is most
often requested. The no-arbitrage bounds supply a bracket for free, and Brent cannot
escape one.

**The scenario log is a log, not a cache.** A surface takes under a millisecond to
compute, so storing one to avoid recomputing it would trade a free operation for a disk
round trip and a consistency problem. What is worth recording is which regimes were
looked at. Rows are keyed on a twelve-character SHA-256 fingerprint of the canonicalised
parameters, which makes `0.1 + 0.2` and `0.3` the same scenario — a composite key over
six float columns would make them different ones. `PRAGMA foreign_keys = ON` is issued on
every connection, without which SQLite parses the foreign key and then ignores it.

**Nothing asserts a full-precision literal.** NumPy dispatches `exp` and `log` to
different kernels on different CPUs, so the same inputs can give results differing in
the last two digits between Windows and Linux. Tests therefore assert tolerances, exact
round-trips, or invariants — never a seventeen-digit number, which would be a test of
the machine rather than of the code.

**Plots assert structure, not pixels.** A golden-image suite compares reliably only
against itself in one environment; across matplotlib versions in CI it drifts, gets
disabled, and stops catching anything. The tests check artist counts, axis labels taken
from the grid, and orientation — `origin="lower"`, because the matplotlib default would
draw a surface rising in volatility upside down and it would still look plausible.

## Development

```bash
ruff check . && ruff format --check .
mypy src tests
pytest --cov
```

All three gate every push in CI, with the test suite running on Python 3.11, 3.12 and
3.13. Doctests are collected and executed, so the examples in every docstring are
tests and cannot drift from the code.

`mypy` runs in strict mode with `possibly-undefined` enabled — an error code that is
not part of `--strict`, and worth having, since a name read on a branch where it was
never bound is a crash no test is guaranteed to reach.

`mypy` targets Python 3.12 even though the package supports 3.11 at runtime: NumPy's
stubs use PEP 695 `type` statements, which mypy cannot parse under a 3.11 target. The
3.11 test job covers runtime support.

Four `# type: ignore[overload-overlap]` comments appear in `pricing.py`, one per public
formula. The all-float overload is a subtype of the general one, so mypy flags the pair
as an unsafe overlap; it is not unsafe, because `scalarize` returns a float exactly when
every input was scalar, which is the relationship the overloads encode and which mypy
cannot verify. Each ignore silences that one code and no other.

## Command line

Everything the library does is reachable from a shell, so any claim in the documentation
can be checked in one line.

```
bsm price --spot 100 --strike 100 --rate 5 --maturity 1 --volatility 20
bsm greeks --kind put
bsm implied 10.45 --strike 100
bsm surface --csv surface.csv --image surface.png --database scenarios.db
bsm store list
```

Rates and volatilities are entered as percentages here and only here. The library works
in decimals throughout; conversion happens once, at the boundary, so nothing downstream
ever sees a percentage.

## The document

`docs/index.qmd` is a Quarto document whose code cells run this package in the reader's
browser through Pyodide. There is no server and nothing precomputed: the wheel is built
from the commit being published, `micropip` installs it client-side, and the code that
runs is the code the tests run.

This is what the no-SciPy decision bought. The package depends only on NumPy, so the
wheel is `py3-none-any` and the runtime is small enough to load in a few seconds. The
narrative figures are rendered ahead of time by `docs/make_figures.py`, so the page is
readable before the runtime finishes starting.

The publish workflow refuses to deploy a page that would not work: it checks the wheel is
pure Python, checks the document names the wheel that was actually built, and then drives
a real Chromium through the rendered site to confirm the runtime boots and a live cell
produces the textbook price. A rendered page can look perfect and be entirely inert.

## Roadmap

Implemented: value objects, day count, normal distribution, forward, `d1`/`d2`, call
and put prices, the five Greeks, no-arbitrage bounds and put-call parity, Brent's
method, implied volatility, vectorised surfaces, a SQLite scenario log, a command-line
interface, and the published document.

Planned: pricing under Merton jump-diffusion, and inverting those prices through
Black-Scholes to exhibit a volatility smile.

## Assistance

This package was written with LLM assistance. The direction, the review, and the
responsibility for what is published are mine.

## Licence

MIT. See [LICENSE](LICENSE).
