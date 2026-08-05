"""Tests for the plotting helpers.

Structure is asserted, not pixels. A rendered image compares reliably only against
itself in the same environment; across matplotlib versions in CI it drifts, and a
golden-image suite that flakes gets disabled and then stops catching anything. What can
be checked robustly is that the right number of artists exist, that the axes carry the
labels the grid declared, and that the orientation is the one the data implies.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from bsm.grids import price_surface  # noqa: E402
from bsm.plot import heatmap, price_curves  # noqa: E402
from bsm.types import OptionKind  # noqa: E402

VOLS = np.linspace(0.1, 0.5, 5)
SPOTS = np.linspace(80.0, 120.0, 9)


@pytest.fixture(autouse=True)
def close_figures() -> object:
    yield
    plt.close("all")


def make_grid() -> object:
    return price_surface(
        OptionKind.CALL,
        volatilities=VOLS,
        spots=SPOTS,
        strike=100.0,
        rate=0.05,
        dividend_yield=0.0,
        maturity=1.0,
    )


class TestHeatmap:
    def test_draws_one_image(self) -> None:
        ax = heatmap(make_grid())  # type: ignore[arg-type]
        assert len(ax.images) == 1

    def test_axis_labels_come_from_the_grid(self) -> None:
        ax = heatmap(make_grid())  # type: ignore[arg-type]
        assert ax.get_xlabel() == "spot"
        assert ax.get_ylabel() == "volatility"

    def test_tick_counts_match_the_axes(self) -> None:
        ax = heatmap(make_grid())  # type: ignore[arg-type]
        assert len(ax.get_xticks()) == len(SPOTS)
        assert len(ax.get_yticks()) == len(VOLS)

    def test_origin_is_lower(self) -> None:
        """Rows increase upward. The matplotlib default would draw this inverted."""
        ax = heatmap(make_grid())  # type: ignore[arg-type]
        bottom, top = ax.get_ylim()
        assert bottom < top

    def test_data_is_not_transposed(self) -> None:
        grid = make_grid()
        ax = heatmap(grid)  # type: ignore[arg-type]
        drawn = ax.images[0].get_array()
        assert drawn is not None
        assert drawn.shape == grid.shape  # type: ignore[attr-defined]

    def test_title_is_optional(self) -> None:
        assert heatmap(make_grid()).get_title() == ""  # type: ignore[arg-type]
        assert (
            heatmap(make_grid(), title="Call price").get_title()  # type: ignore[arg-type]
            == "Call price"
        )

    def test_annotates_small_grids_by_default(self) -> None:
        ax = heatmap(make_grid())  # type: ignore[arg-type]
        assert len(ax.texts) == len(VOLS) * len(SPOTS)

    def test_does_not_annotate_large_grids_by_default(self) -> None:
        grid = price_surface(
            OptionKind.CALL,
            volatilities=np.linspace(0.1, 0.5, 25),
            spots=np.linspace(80.0, 120.0, 25),
            strike=100.0,
            rate=0.05,
            dividend_yield=0.0,
            maturity=1.0,
        )
        assert len(heatmap(grid).texts) == 0

    def test_annotation_can_be_forced_off(self) -> None:
        assert len(heatmap(make_grid(), annotate=False).texts) == 0  # type: ignore[arg-type]

    def test_draws_on_a_supplied_axes(self) -> None:
        figure, ax = plt.subplots()
        assert heatmap(make_grid(), ax=ax) is ax  # type: ignore[arg-type]

    def test_colorbar_is_labelled(self) -> None:
        ax = heatmap(make_grid(), value_label="premium")  # type: ignore[arg-type]
        labels = [a.get_ylabel() for a in ax.figure.axes if a is not ax]
        assert "premium" in labels

    def test_does_not_display_or_save(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A helper that opens a window is unusable from a script or a test."""
        called = []
        monkeypatch.setattr(plt, "show", lambda *a, **k: called.append("show"))
        heatmap(make_grid())  # type: ignore[arg-type]
        assert called == []


def curves_only(ax: object) -> list[object]:
    """Drop the strike guide, which matplotlib auto-labels with an underscore."""
    return [
        line
        for line in ax.lines  # type: ignore[attr-defined]
        if not str(line.get_label()).startswith("_")
    ]


class TestPriceCurves:
    def test_one_line_per_maturity(self) -> None:
        ax = price_curves(
            OptionKind.CALL,
            spots=SPOTS,
            maturities=np.array([0.0, 0.25, 1.0]),
            strike=100.0,
            rate=0.05,
            dividend_yield=0.0,
            volatility=0.2,
        )
        assert len(curves_only(ax)) == 3

    def test_zero_maturity_is_labelled_as_the_payoff(self) -> None:
        ax = price_curves(
            OptionKind.CALL,
            spots=SPOTS,
            maturities=np.array([0.0, 1.0]),
            strike=100.0,
            rate=0.05,
            dividend_yield=0.0,
            volatility=0.2,
        )
        assert curves_only(ax)[0].get_label() == "payoff"  # type: ignore[attr-defined]

    def test_strike_is_marked(self) -> None:
        """A vertical guide at the strike, excluded from the legend."""
        ax = price_curves(
            OptionKind.CALL,
            spots=SPOTS,
            maturities=np.array([1.0]),
            strike=100.0,
            rate=0.05,
            dividend_yield=0.0,
            volatility=0.2,
        )
        assert len(ax.lines) == len(curves_only(ax)) + 1

    def test_labels(self) -> None:
        ax = price_curves(
            OptionKind.PUT,
            spots=SPOTS,
            maturities=np.array([1.0]),
            strike=100.0,
            rate=0.05,
            dividend_yield=0.0,
            volatility=0.2,
        )
        assert ax.get_xlabel() == "spot"
        assert ax.get_ylabel() == "put price"

    def test_longer_maturity_is_worth_more(self) -> None:
        """A structural check on the drawn data, not just on the artists."""
        ax = price_curves(
            OptionKind.CALL,
            spots=SPOTS,
            maturities=np.array([0.0, 2.0]),
            strike=100.0,
            rate=0.05,
            dividend_yield=0.0,
            volatility=0.2,
        )
        drawn = curves_only(ax)
        payoff = drawn[0].get_ydata()  # type: ignore[attr-defined]
        long_dated = drawn[1].get_ydata()  # type: ignore[attr-defined]
        assert np.all(np.asarray(long_dated) >= np.asarray(payoff) - 1e-12)

    def test_title_is_optional(self) -> None:
        ax = price_curves(
            OptionKind.CALL,
            spots=SPOTS,
            maturities=np.array([1.0]),
            strike=100.0,
            rate=0.05,
            dividend_yield=0.0,
            volatility=0.2,
            title="Contours",
        )
        assert ax.get_title() == "Contours"

    def test_draws_on_a_supplied_axes(self) -> None:
        figure, target = plt.subplots()
        returned = price_curves(
            OptionKind.CALL,
            spots=SPOTS,
            maturities=np.array([1.0]),
            strike=100.0,
            rate=0.05,
            dividend_yield=0.0,
            volatility=0.2,
            ax=target,
        )
        assert returned is target
