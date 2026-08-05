"""Command-line interface.

Every capability the library offers is reachable from a shell, so any claim made in the
documentation can be checked with a one-line command rather than a Python session. The
entry point is ``bsm``.

Percentages at the boundary
---------------------------
The library works in decimals throughout, and says so in one place. The command line is
the exception: rates and volatilities are entered and displayed as percentages, because
that is how they are quoted everywhere a number would be read off and typed in.
Conversion happens once, at the boundary, in :func:`_as_decimal` — nothing downstream
ever sees a percentage. Mixing the two conventions inside a codebase is a reliable
source of silent hundred-fold errors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from bsm import __version__
from bsm.greeks import greeks as compute_greeks
from bsm.grids import price_surface
from bsm.implied import ImpliedVolatilityError, implied_volatility
from bsm.pricing import price as compute_price
from bsm.store import ScenarioStore
from bsm.types import Contract, Market, OptionKind

__all__ = ["app", "main"]

app = typer.Typer(
    name="bsm",
    help="Black-Scholes-Merton option pricing, Greeks, and implied volatility.",
    add_completion=False,
    no_args_is_help=True,
)
store_app = typer.Typer(help="Inspect the scenario log.", no_args_is_help=True)
app.add_typer(store_app, name="store")

console = Console()

DEFAULT_DATABASE = Path("scenarios.db")

Kind = Annotated[OptionKind, typer.Option("--kind", "-k", help="call or put")]
Spot = Annotated[float, typer.Option("--spot", "-s", help="Spot price.")]
Strike = Annotated[float, typer.Option("--strike", "-K", help="Strike price.")]
Rate = Annotated[float, typer.Option("--rate", "-r", help="Risk-free rate, percent.")]
Yield = Annotated[float, typer.Option("--yield", "-q", help="Dividend yield, percent.")]
Maturity = Annotated[float, typer.Option("--maturity", "-t", help="Years to expiry.")]
Volatility = Annotated[
    float, typer.Option("--volatility", "-v", help="Volatility, percent.")
]


def _as_decimal(percent: float) -> float:
    """Convert a percentage entered on the command line to a decimal.

    Parameters
    ----------
    percent
        A rate or volatility as a percentage, so 20.0 means 20%.

    Returns
    -------
    float
        The same quantity as a decimal.

    Examples
    --------
    >>> _as_decimal(20.0)
    0.2
    """
    return percent / 100.0


def _version_callback(value: bool) -> None:
    """Print the version and exit, if requested."""
    if value:
        console.print(f"bsm-pricer {__version__}")
        raise typer.Exit


@app.callback()
def _main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version_callback, is_eager=True, help="Show version."
        ),
    ] = False,
) -> None:
    """Black-Scholes-Merton option pricing, Greeks, and implied volatility."""


@app.command()
def price(
    kind: Kind = OptionKind.CALL,
    spot: Spot = 100.0,
    strike: Strike = 100.0,
    rate: Rate = 5.0,
    dividend_yield: Yield = 0.0,
    maturity: Maturity = 1.0,
    volatility: Volatility = 20.0,
) -> None:
    """Price a European option."""
    contract = Contract(strike=strike, maturity=maturity, kind=kind)
    market = Market(
        spot=spot,
        rate=_as_decimal(rate),
        volatility=_as_decimal(volatility),
        dividend_yield=_as_decimal(dividend_yield),
    )
    console.print(f"{compute_price(contract, market):.6f}")


@app.command()
def greeks(
    kind: Kind = OptionKind.CALL,
    spot: Spot = 100.0,
    strike: Strike = 100.0,
    rate: Rate = 5.0,
    dividend_yield: Yield = 0.0,
    maturity: Maturity = 1.0,
    volatility: Volatility = 20.0,
) -> None:
    """Show all five Greeks, in the conventional quoting units."""
    contract = Contract(strike=strike, maturity=maturity, kind=kind)
    market = Market(
        spot=spot,
        rate=_as_decimal(rate),
        volatility=_as_decimal(volatility),
        dividend_yield=_as_decimal(dividend_yield),
    )
    result = compute_greeks(contract, market)

    table = Table(title=f"{kind.value} greeks")
    table.add_column("greek")
    table.add_column("value", justify="right")
    table.add_column("per", justify="left", style="dim")
    table.add_row("delta", f"{result.delta:.6f}", "unit of spot")
    table.add_row("gamma", f"{result.gamma:.6f}", "unit of spot squared")
    table.add_row("vega", f"{result.vega / 100.0:.6f}", "volatility point")
    table.add_row("theta", f"{result.theta / 365.0:.6f}", "calendar day")
    table.add_row("rho", f"{result.rho / 10_000.0:.6f}", "basis point")
    console.print(table)


@app.command()
def implied(
    observed: Annotated[float, typer.Argument(help="Observed option price.")],
    kind: Kind = OptionKind.CALL,
    spot: Spot = 100.0,
    strike: Strike = 100.0,
    rate: Rate = 5.0,
    dividend_yield: Yield = 0.0,
    maturity: Maturity = 1.0,
) -> None:
    """Recover the volatility implied by an observed price."""
    try:
        result = implied_volatility(
            observed,
            kind,
            spot,
            strike,
            _as_decimal(rate),
            _as_decimal(dividend_yield),
            maturity,
        )
    except ImpliedVolatilityError as error:
        # The library raises rather than returning a sentinel, so the shell gets a
        # nonzero status and the message goes to stderr where a pipeline expects it.
        console.print(f"[red]{error}[/red]", style="bold")
        raise typer.Exit(code=1) from error
    console.print(f"{100.0 * result:.6f}%")


@app.command()
def surface(
    kind: Kind = OptionKind.CALL,
    strike: Strike = 100.0,
    rate: Rate = 5.0,
    dividend_yield: Yield = 0.0,
    maturity: Maturity = 1.0,
    spot_low: Annotated[float, typer.Option(help="Lowest spot.")] = 80.0,
    spot_high: Annotated[float, typer.Option(help="Highest spot.")] = 120.0,
    spot_steps: Annotated[int, typer.Option(help="Number of spots.")] = 9,
    volatility_low: Annotated[float, typer.Option(help="Lowest vol, %.")] = 10.0,
    volatility_high: Annotated[float, typer.Option(help="Highest vol, %.")] = 50.0,
    volatility_steps: Annotated[int, typer.Option(help="Number of vols.")] = 5,
    csv: Annotated[Path | None, typer.Option(help="Write the grid here.")] = None,
    image: Annotated[Path | None, typer.Option(help="Write a heatmap here.")] = None,
    database: Annotated[
        Path | None, typer.Option(help="Record the scenario here.")
    ] = None,
) -> None:
    """Compute a price surface over spot and volatility."""
    spots = np.linspace(spot_low, spot_high, spot_steps)
    volatilities = np.linspace(
        _as_decimal(volatility_low), _as_decimal(volatility_high), volatility_steps
    )
    grid = price_surface(
        kind,
        volatilities=volatilities,
        spots=spots,
        strike=strike,
        rate=_as_decimal(rate),
        dividend_yield=_as_decimal(dividend_yield),
        maturity=maturity,
    )

    if csv is not None:
        grid.to_csv(csv)
        console.print(f"wrote {csv}")

    if image is not None:
        # Imported here, not at module scope: matplotlib is an optional dependency and
        # every other command must work without it.
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from bsm.plot import heatmap

        # Create the figure here rather than reading it back off the axes: the
        # `Axes.figure` attribute is typed as `Figure | SubFigure`, and only the
        # former can be saved.
        figure, axes = plt.subplots(figsize=(10.0, 6.0))
        heatmap(grid, ax=axes, title=f"{kind.value} price")
        figure.savefig(image, dpi=150, bbox_inches="tight")
        plt.close(figure)
        console.print(f"wrote {image}")

    if database is not None:
        with ScenarioStore(database) as store:
            scenario = store.record(
                grid,
                kind=kind,
                strike=strike,
                rate=_as_decimal(rate),
                dividend_yield=_as_decimal(dividend_yield),
                maturity=maturity,
            )
        console.print(f"recorded {scenario.fingerprint}")

    if csv is None and image is None and database is None:
        _print_grid(grid)


def _print_grid(grid: object) -> None:
    """Render a grid to the console as a table.

    Parameters
    ----------
    grid
        The surface to print.
    """
    from bsm.grids import Grid

    assert isinstance(grid, Grid)  # noqa: S101 - narrowing for the type checker
    table = Table(title=f"{grid.row_label} down, {grid.column_label} across")
    table.add_column(grid.row_label, justify="right", style="bold")
    for column in grid.columns:
        table.add_column(f"{column:g}", justify="right")
    for i, row in enumerate(grid.rows):
        table.add_row(
            f"{row:g}", *[f"{value:.4f}" for value in grid.values[i].tolist()]
        )
    console.print(table)


@store_app.command("list")
def store_list(
    database: Annotated[Path, typer.Option(help="Database file.")] = DEFAULT_DATABASE,
) -> None:
    """List recorded scenarios."""
    with ScenarioStore(database) as store:
        scenarios = store.scenarios()

    if not scenarios:
        console.print("no scenarios recorded")
        return

    table = Table(title=f"{len(scenarios)} scenario(s) in {database}")
    for heading in (
        "fingerprint",
        "recorded",
        "kind",
        "strike",
        "rate %",
        "yield %",
        "years",
    ):
        table.add_column(heading)
    for scenario in scenarios:
        table.add_row(
            scenario.fingerprint,
            scenario.recorded_at,
            scenario.kind.value,
            f"{scenario.strike:g}",
            f"{100.0 * scenario.rate:g}",
            f"{100.0 * scenario.dividend_yield:g}",
            f"{scenario.maturity:g}",
        )
    console.print(table)


@store_app.command("show")
def store_show(
    digest: Annotated[str, typer.Argument(help="Scenario fingerprint.")],
    database: Annotated[Path, typer.Option(help="Database file.")] = DEFAULT_DATABASE,
) -> None:
    """Reprint a recorded surface."""
    with ScenarioStore(database) as store:
        try:
            grid = store.grid(digest)
        except KeyError as error:
            console.print(f"[red]{error}[/red]")
            raise typer.Exit(code=1) from error
    _print_grid(grid)


@store_app.command("clear")
def store_clear(
    database: Annotated[Path, typer.Option(help="Database file.")] = DEFAULT_DATABASE,
    force: Annotated[bool, typer.Option("--force", help="Skip confirmation.")] = False,
) -> None:
    """Delete every recorded scenario."""
    if not force:
        typer.confirm(f"Delete every scenario in {database}?", abort=True)
    with ScenarioStore(database) as store:
        store.clear()
    console.print("cleared")


def main() -> None:
    """Console entry point."""
    app()
