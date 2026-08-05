"""Generate the static figures the document embeds.

The narrative figures are rendered ahead of time rather than in live cells, so the page
is readable the moment it loads. The Pyodide runtime is several megabytes and takes a
few seconds to start; gating the argument behind it would mean a reader sees an empty
page first and the point of the article second.

Run from the repository root::

    python docs/make_figures.py

Regenerating is idempotent: same inputs, same output, no timestamps embedded.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bsm.grids import price_surface  # noqa: E402
from bsm.merton import JumpParameters, volatility_smile  # noqa: E402
from bsm.plot import heatmap, price_curves  # noqa: E402
from bsm.types import OptionKind  # noqa: E402

FIGURES = Path(__file__).parent / "figures"
DPI = 150


def build_curves() -> None:
    """Price against spot at several times to maturity, including the payoff."""
    figure, ax = plt.subplots(figsize=(8.0, 5.0))
    price_curves(
        OptionKind.CALL,
        spots=np.linspace(50.0, 150.0, 201),
        maturities=np.array([0.0, 0.08, 0.25, 1.0, 3.0]),
        strike=100.0,
        rate=0.05,
        dividend_yield=0.0,
        volatility=0.20,
        ax=ax,
        title="Call price approaching its payoff",
    )
    figure.savefig(FIGURES / "curves.png", dpi=DPI, bbox_inches="tight")
    plt.close(figure)


def build_heatmap() -> None:
    """Price over spot and volatility."""
    grid = price_surface(
        OptionKind.CALL,
        volatilities=np.linspace(0.10, 0.60, 11),
        spots=np.linspace(70.0, 130.0, 13),
        strike=100.0,
        rate=0.05,
        dividend_yield=0.0,
        maturity=1.0,
    )
    figure, ax = plt.subplots(figsize=(11.0, 6.0))
    heatmap(grid, ax=ax, title="Call price over spot and volatility")
    figure.savefig(FIGURES / "heatmap.png", dpi=DPI, bbox_inches="tight")
    plt.close(figure)


def build_smile() -> None:
    """Implied volatility against strike, for prices generated with jumps."""
    strikes = np.linspace(70.0, 130.0, 61)
    shared = dict(
        spot=100.0, rate=0.05, dividend_yield=0.0, maturity=1.0, volatility=0.20
    )

    figure, ax = plt.subplots(figsize=(8.0, 5.0))
    for label, jumps in (
        ("no jumps", JumpParameters(0.0, 0.0, 0.0)),
        ("symmetric jumps", JumpParameters(1.0, 0.0, 0.20)),
        ("downward jumps", JumpParameters(1.0, -0.10, 0.15)),
    ):
        smile = volatility_smile(strikes, jumps=jumps, **shared)  # type: ignore[arg-type]
        ax.plot(strikes, 100.0 * smile, label=label)

    ax.axvline(100.0, color="grey", linewidth=0.8, linestyle="--", zorder=0)
    ax.axhline(20.0, color="grey", linewidth=0.8, linestyle=":", zorder=0)
    ax.set_xlabel("strike")
    ax.set_ylabel("Black-Scholes implied volatility (%)")
    ax.set_title("A smile is what a wrong distribution looks like from inside")
    ax.grid(True, alpha=0.3)
    ax.legend()
    figure.savefig(FIGURES / "smile.png", dpi=DPI, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Write every figure the document embeds."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    build_curves()
    build_heatmap()
    build_smile()
    print(f"wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
