# Plan

A typed, tested Python package for European option pricing under Black-Scholes-Merton,
with an interactive published document.

## Decisions

**Interactivity via Pyodide.** Live cells in the published document run the package
itself, so there is one implementation of the mathematics and nothing to host. This is
only viable because the package does not depend on SciPy.

**No SciPy.** The normal CDF is Hart's algorithm, implemented here; the root-finder is
Brent's method, implemented here. SciPy remains a *test-only* dependency: our `brentq`
is pinned against `scipy.optimize.brentq`, and the CDF against `math.erfc`.

**NumPy is required.** A stdlib-only scalar path alongside a NumPy array path would
mean two implementations of every formula, which is the duplication the Pyodide
decision exists to avoid. NumPy is pre-built in Pyodide, adds roughly 7 MB, and loads
quickly.

**SQLite as a scenario log, not a cache.** Caching a microsecond computation is
indefensible. Recording which parameter regimes were explored is not. Rows are keyed on
a 12-character SHA-256 fingerprint of the canonical parameter tuple, which avoids
float-equality lookup and makes the foreign key meaningful.

**One document, not a book.** Roughly 2,000–2,500 words: the model and its assumptions,
the pricer, the Greeks, the inverse problem, the sensitivity surface, and a closing
section on what the model does not capture.

## Package layout

```
src/bsm/
  types.py       frozen dataclasses, validated at construction
  daycount.py    ACT/365F and ACT/360, calendar-correct tenor arithmetic
  normal.py      standard normal PDF and CDF (Hart)
  pricing.py     forward, d1/d2, call and put prices
  greeks.py      delta, gamma, vega, theta, rho          [Phase 2]
  bounds.py      no-arbitrage bounds, put-call parity     [Phase 2]
  roots.py       Brent's method                           [Phase 2]
  implied.py     implied volatility, typed failure        [Phase 2]
  grids.py       vectorised sensitivity surfaces          [Phase 2]
  store.py       SQLite scenario log                      [Phase 3]
  plot.py        heatmap and line plots                   [Phase 3]
  cli.py         typer entry point                        [Phase 3]
  merton.py      jump-diffusion, for the smile section    [Phase 4]
```

## Phases

1. **Foundation.** Tooling, CI with lint and type-check gating every push, `types`,
   `daycount`, `normal`, `pricing`. *Complete.*
2. **Substance.** Greeks, bounds, Brent, implied volatility, grids. Property suite:
   put-call parity, monotonicity, arbitrage bounds, Greeks against central finite
   differences, implied-volatility round trip. *Complete.*
3. **Surfaces.** CLI, store, plots, wheel build, Quarto document with Pyodide cells,
   Pages deploy, Playwright smoke test. *Complete.*
4. **The smile.** Price under Merton jump-diffusion — a Poisson-weighted sum of
   Black-Scholes prices — invert through the Black-Scholes solver, and observe the
   smile. This demonstrates the mechanism by which a smile arises, with ground truth
   known because it was generated here. It does not claim that real markets exhibit
   one; the text cites the empirical literature for that. *Complete.*

## Testing strategy

- **Unit** — reference values computed independently in the test, boundary behaviour at
  zero maturity and zero volatility, calendar arithmetic across month ends and leap
  years.
- **Property** (Hypothesis) — invariants that must hold for all admissible inputs.
- **Numerical** — implementations pinned against stdlib and SciPy oracles.
- **Integration** — CLI round trip, store write/read identity, fingerprint stability.
- **End-to-end** — the document builds in CI and its cells execute; Playwright smoke
  test.

Golden files record grid *data* as CSV, never rendered images: pixel comparison is
stable within one environment but drifts across matplotlib versions in CI.
