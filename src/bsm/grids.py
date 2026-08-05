r"""Two-dimensional sensitivity surfaces.

A surface is what the pricing formulas were made array-capable for: vary two parameters
over a grid, hold the rest fixed, and evaluate every cell at once. The 41-by-21 grid in
the README is a single call into :func:`bsm.pricing.call_price` with two broadcast
arrays, not 861 scalar calls in a nested loop.

The :class:`Grid` returned here carries its axes and their labels alongside the values,
so a plot or a CSV can be produced without the caller having to remember which axis was
which — an ordering that is easy to get wrong and, once wrong, produces a picture that
looks entirely plausible.

Orientation
-----------
``values[i, j]`` corresponds to ``rows[i]`` and ``columns[j]``, the convention followed
by NumPy indexing, :func:`matplotlib.pyplot.imshow`, and CSV files alike. Rows vary
down the first axis and columns across the second.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bsm._numeric import FloatArray
from bsm.pricing import call_price, put_price
from bsm.types import OptionKind

__all__ = ["Grid", "price_surface"]


@dataclass(frozen=True, slots=True)
class Grid:
    """A labelled two-dimensional array of computed values.

    Attributes
    ----------
    rows
        Values of the parameter varying down the first axis.
    columns
        Values of the parameter varying across the second axis.
    values
        Computed values, of shape ``(len(rows), len(columns))``.
    row_label
        Human-readable name of the row parameter, for axis labels and CSV headers.
    column_label
        Human-readable name of the column parameter.

    Raises
    ------
    ValueError
        If ``values`` does not have shape ``(len(rows), len(columns))``. Checked at
        construction because a transposed grid is otherwise indistinguishable from a
        correct one whenever the two axes happen to have the same length.

    Examples
    --------
    >>> grid = Grid(
    ...     rows=np.array([0.1, 0.2]),
    ...     columns=np.array([90.0, 100.0, 110.0]),
    ...     values=np.zeros((2, 3)),
    ...     row_label="volatility",
    ...     column_label="spot",
    ... )
    >>> grid.shape
    (2, 3)
    """

    rows: FloatArray
    columns: FloatArray
    values: FloatArray
    row_label: str
    column_label: str

    def __post_init__(self) -> None:
        """Validate that the values match the declared axes."""
        expected = (len(self.rows), len(self.columns))
        if self.values.shape != expected:
            raise ValueError(
                f"values has shape {self.values.shape}, expected {expected} from "
                f"{len(self.rows)} rows and {len(self.columns)} columns"
            )

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the value array.

        Examples
        --------
        >>> price_surface(
        ...     OptionKind.CALL,
        ...     volatilities=np.linspace(0.1, 0.5, 5),
        ...     spots=np.linspace(80.0, 120.0, 9),
        ...     strike=100.0,
        ...     rate=0.05,
        ...     dividend_yield=0.0,
        ...     maturity=1.0,
        ... ).shape
        (5, 9)
        """
        return len(self.rows), len(self.columns)

    def to_csv(self, path: str | Path) -> None:
        r"""Write the grid to a CSV file with labelled axes.

        The first cell of the header holds ``"{row_label}\{column_label}"``, the rest
        of the header row holds the column values, and each subsequent row begins with
        its row value.

        Parameters
        ----------
        path
            Destination file. Any existing content is overwritten.

        Notes
        -----
        This is the format the test suite uses for golden files. Grid *data* is
        recorded rather than a rendered image, because pixel comparison of a plot is
        stable within one environment and drifts across matplotlib versions in CI.

        Examples
        --------
        >>> import tempfile, pathlib
        >>> grid = price_surface(
        ...     OptionKind.CALL,
        ...     volatilities=np.array([0.2]),
        ...     spots=np.array([100.0]),
        ...     strike=100.0,
        ...     rate=0.05,
        ...     dividend_yield=0.0,
        ...     maturity=1.0,
        ... )
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     destination = pathlib.Path(directory) / "surface.csv"
        ...     grid.to_csv(destination)
        ...     header, row = destination.read_text().strip().splitlines()
        >>> header
        'volatility\\spot,100.0'

        The value is written at full precision, so the file round-trips exactly and can
        serve as a golden. Asserted rather than printed: the last two digits of a price
        differ between platforms, because NumPy dispatches `exp` and `log` to different
        kernels, and a literal here would be a test of the CPU rather than of the code.

        >>> bool(float(row.split(",")[1]) == grid.values[0, 0])
        True
        """
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [f"{self.row_label}\\{self.column_label}", *self.columns.tolist()]
            )
            for row_value, row_values in zip(
                self.rows.tolist(), self.values.tolist(), strict=True
            ):
                writer.writerow([row_value, *row_values])


def price_surface(
    kind: OptionKind,
    *,
    volatilities: FloatArray,
    spots: FloatArray,
    strike: float,
    rate: float,
    dividend_yield: float,
    maturity: float,
) -> Grid:
    """Evaluate option prices over a grid of volatilities and spot prices.

    Parameters
    ----------
    kind
        Call or put.
    volatilities
        One-dimensional array of volatilities, varying down the rows.
    spots
        One-dimensional array of spot prices, varying across the columns.
    strike
        Exercise price, strictly positive.
    rate
        Continuously compounded risk-free rate, as a decimal.
    dividend_yield
        Continuous dividend yield, as a decimal.
    maturity
        Time to expiry in years, non-negative.

    Returns
    -------
    Grid
        Prices, with ``values[i, j]`` the price at ``volatilities[i]`` and
        ``spots[j]``.

    Raises
    ------
    ValueError
        If either axis is not one-dimensional.

    Notes
    -----
    Keyword-only after ``kind``. Six positional floats in a row is a call site nobody
    can read, and transposing two of them produces a plausible-looking surface rather
    than an error.

    The whole grid is one broadcast evaluation: ``volatilities`` is reshaped to a
    column and ``spots`` left as a row, so NumPy computes every cell in a single pass.

    Examples
    --------
    >>> grid = price_surface(
    ...     OptionKind.CALL,
    ...     volatilities=np.array([0.1, 0.2, 0.3]),
    ...     spots=np.array([90.0, 100.0, 110.0]),
    ...     strike=100.0,
    ...     rate=0.05,
    ...     dividend_yield=0.0,
    ...     maturity=1.0,
    ... )
    >>> grid.shape
    (3, 3)
    >>> grid.row_label, grid.column_label
    ('volatility', 'spot')

    Prices increase along both axes, in the spot direction because a call gains with
    the underlying, and in the volatility direction because vega is positive:

    >>> bool(np.all(np.diff(grid.values, axis=0) > 0))
    True
    >>> bool(np.all(np.diff(grid.values, axis=1) > 0))
    True

    The at-the-money cell agrees with a direct scalar call:

    >>> from bsm.pricing import call_price
    >>> bool(grid.values[1, 1] == call_price(100.0, 100.0, 0.05, 0.0, 1.0, 0.2))
    True
    """
    if volatilities.ndim != 1:
        raise ValueError(
            f"volatilities must be one-dimensional, got {volatilities.ndim}"
        )
    if spots.ndim != 1:
        raise ValueError(f"spots must be one-dimensional, got {spots.ndim}")

    formula = call_price if kind is OptionKind.CALL else put_price
    values = formula(
        spots.reshape(1, -1),
        strike,
        rate,
        dividend_yield,
        maturity,
        volatilities.reshape(-1, 1),
    )

    return Grid(
        rows=volatilities,
        columns=spots,
        values=np.asarray(values, dtype=np.float64),
        row_label="volatility",
        column_label="spot",
    )
