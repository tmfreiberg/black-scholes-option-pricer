"""Rendering for surfaces and curves.

Matplotlib is an optional dependency, declared under the ``plot`` extra. Everything
else in the package works without it; importing this module without it installed raises
an :class:`ImportError` that says which extra to install rather than a bare
``ModuleNotFoundError`` from three frames down.

Every function takes an optional ``ax`` and returns the axes it drew on, so callers can
compose figures, and none of them calls ``show`` or ``savefig``. Deciding where output
goes is the caller's business; a plotting helper that displays a window is unusable
from a script, a notebook, and a test alike.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bsm._numeric import FloatArray
from bsm.grids import Grid
from bsm.pricing import call_price, put_price
from bsm.types import OptionKind

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as error:  # pragma: no cover - exercised by CI, not tests
    raise ImportError(
        "bsm.plot needs matplotlib; install it with: pip install 'bsm-pricer[plot]'"
    ) from error

__all__ = ["heatmap", "price_curves"]

# Diverging maps are for data with a meaningful centre. Option prices are non-negative
# and have none, so a perceptually uniform sequential map is the honest choice: equal
# steps in price become equal steps in perceived lightness, and it survives greyscale
# printing and the common forms of colour blindness.
_DEFAULT_COLORMAP = "viridis"

# Above roughly this many cells, per-cell text is unreadable and the annotation loop
# dominates the render time.
_MAX_ANNOTATED_CELLS = 400


def heatmap(
    grid: Grid,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    colormap: str = _DEFAULT_COLORMAP,
    annotate: bool | None = None,
    value_label: str = "price",
) -> Axes:
    """Draw a grid as a colour-mapped heatmap.

    Parameters
    ----------
    grid
        The surface to draw. ``grid.values[i, j]`` is placed at row ``i``, column ``j``,
        matching NumPy indexing and :func:`matplotlib.pyplot.imshow`.
    ax
        Axes to draw on. A new figure and axes are created if omitted.
    title
        Figure title. Omitted if not given.
    colormap
        Any matplotlib colormap name.
    annotate
        Whether to write each cell's value into the cell. Defaults to doing so when the
        grid has at most 400 cells and not otherwise, since beyond that the text is
        unreadable anyway.
    value_label
        Label for the colour bar.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.

    Notes
    -----
    Uses ``origin="lower"`` so the first row appears at the bottom. The default in
    matplotlib is ``"upper"``, which is right for images and wrong for data whose row
    axis is a quantity that increases: with the default, a surface rising in volatility
    would be drawn upside down and still look entirely plausible.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import numpy as np
    >>> from bsm.grids import price_surface
    >>> grid = price_surface(
    ...     OptionKind.CALL,
    ...     volatilities=np.linspace(0.1, 0.5, 5),
    ...     spots=np.linspace(80.0, 120.0, 9),
    ...     strike=100.0,
    ...     rate=0.05,
    ...     dividend_yield=0.0,
    ...     maturity=1.0,
    ... )
    >>> ax = heatmap(grid, title="Call price")
    >>> ax.get_xlabel(), ax.get_ylabel()
    ('spot', 'volatility')
    >>> len(ax.images)
    1
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10.0, 6.0))

    image = ax.imshow(
        grid.values,
        cmap=colormap,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(len(grid.columns)))
    ax.set_xticklabels([f"{value:g}" for value in grid.columns], rotation=90)
    ax.set_yticks(np.arange(len(grid.rows)))
    ax.set_yticklabels([f"{value:g}" for value in grid.rows])
    ax.set_xlabel(grid.column_label)
    ax.set_ylabel(grid.row_label)

    if title is not None:
        ax.set_title(title)

    bar = ax.figure.colorbar(image, ax=ax)
    bar.set_label(value_label)

    if annotate is None:
        annotate = grid.values.size <= _MAX_ANNOTATED_CELLS
    if annotate:
        _annotate_cells(ax, grid)

    return ax


def _annotate_cells(ax: Axes, grid: Grid) -> None:
    """Write each cell's value into the cell, in a contrasting colour.

    Parameters
    ----------
    ax
        Axes already carrying the heatmap image.
    grid
        The surface being drawn.

    Notes
    -----
    Text colour is chosen per cell from the value's position in the range, rather than
    fixed. A single colour is illegible at one end of any sequential colormap.
    """
    midpoint = 0.5 * (float(grid.values.min()) + float(grid.values.max()))
    for i in range(len(grid.rows)):
        for j in range(len(grid.columns)):
            value = float(grid.values[i, j])
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black" if value > midpoint else "white",
            )


def price_curves(
    kind: OptionKind,
    *,
    spots: FloatArray,
    maturities: FloatArray,
    strike: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Plot price against spot at several times to maturity.

    Parameters
    ----------
    kind
        Call or put.
    spots
        One-dimensional array of spot prices, the horizontal axis.
    maturities
        One curve is drawn per entry, in years.
    strike, rate, dividend_yield, volatility
        Fixed terms shared by every curve.
    ax
        Axes to draw on. A new figure and axes are created if omitted.
    title
        Figure title. Omitted if not given.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.

    Notes
    -----
    Including ``0.0`` among the maturities draws the payoff itself, since an expired
    option is worth its intrinsic value. That curve is the kink the other curves
    approach, and drawing it makes the convergence visible rather than asserted.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import numpy as np
    >>> ax = price_curves(
    ...     OptionKind.CALL,
    ...     spots=np.linspace(50.0, 150.0, 101),
    ...     maturities=np.array([0.0, 0.25, 1.0]),
    ...     strike=100.0,
    ...     rate=0.05,
    ...     dividend_yield=0.0,
    ...     volatility=0.2,
    ... )
    >>> len(ax.lines)  # three curves, plus the vertical guide at the strike
    4
    >>> ax.get_xlabel()
    'spot'
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8.0, 5.0))

    formula = call_price if kind is OptionKind.CALL else put_price
    for maturity in maturities:
        values = formula(
            spots, strike, rate, dividend_yield, float(maturity), volatility
        )
        label = "payoff" if maturity == 0.0 else f"$\\tau$ = {float(maturity):g}y"
        ax.plot(spots, values, label=label)

    ax.axvline(strike, color="grey", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xlabel("spot")
    ax.set_ylabel(f"{kind.value} price")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if title is not None:
        ax.set_title(title)

    return ax
