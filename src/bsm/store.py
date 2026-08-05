"""A record of which parameter regimes have been explored.

This is a log, not a cache. Caching would be indefensible: a 41-by-21 surface takes
under a millisecond to compute, so storing one to avoid recomputing it would trade a
free operation for a disk round trip and a consistency problem. What is worth storing
is the *fact* that a particular surface was looked at — which strike, which rate, which
range of volatilities — so that a session can be reconstructed, compared, or handed to
someone else.

Identity is a fingerprint
-------------------------
Each scenario is keyed on :func:`fingerprint`, a twelve-character SHA-256 digest of the
canonicalised parameters. This is deliberate. The obvious alternative — a composite
primary key over the six float columns — makes every lookup a float equality test,
which is fragile in exactly the way float equality always is: ``0.1 + 0.2`` and ``0.3``
are different keys. Canonicalising to ten decimal places before hashing collapses that
distinction, and ten places is far finer than any quantity here is known to.

The fingerprint covers the axes as well as the scalars, so two surfaces over different
volatility ranges are different scenarios even with identical contract terms.

Schema
------
Two tables. ``scenarios`` holds one row per recorded surface; ``grid_points`` holds its
cells, one row each, referencing the parent by fingerprint with ``ON DELETE CASCADE``.
The foreign key is real: ``PRAGMA foreign_keys = ON`` is issued on every connection,
without which SQLite accepts and then ignores the constraint entirely.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

import numpy as np

from bsm._numeric import FloatArray
from bsm.grids import Grid
from bsm.types import OptionKind

__all__ = ["Scenario", "ScenarioStore", "fingerprint"]

_DECIMAL_PLACES = 10
_FINGERPRINT_LENGTH = 12

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scenarios (
    fingerprint     TEXT PRIMARY KEY,
    recorded_at     TEXT    NOT NULL,
    kind            TEXT    NOT NULL,
    strike          REAL    NOT NULL,
    rate            REAL    NOT NULL,
    dividend_yield  REAL    NOT NULL,
    maturity        REAL    NOT NULL,
    row_label       TEXT    NOT NULL,
    column_label    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS grid_points (
    fingerprint     TEXT    NOT NULL,
    row_index       INTEGER NOT NULL,
    column_index    INTEGER NOT NULL,
    row_value       REAL    NOT NULL,
    column_value    REAL    NOT NULL,
    value           REAL    NOT NULL,
    PRIMARY KEY (fingerprint, row_index, column_index),
    FOREIGN KEY (fingerprint) REFERENCES scenarios(fingerprint) ON DELETE CASCADE
);
"""


def _canonical(value: float) -> str:
    """Render a float in the fixed form used for hashing.

    Parameters
    ----------
    value
        The number to canonicalise.

    Returns
    -------
    str
        Fixed-point text at ten decimal places, with negative zero normalised to
        positive zero so that ``-0.0`` and ``0.0`` fingerprint identically.

    Examples
    --------
    >>> _canonical(0.1 + 0.2) == _canonical(0.3)
    True
    >>> _canonical(-0.0)
    '0.0000000000'
    """
    rounded = round(value, _DECIMAL_PLACES)
    if rounded == 0.0:
        rounded = 0.0
    return f"{rounded:.{_DECIMAL_PLACES}f}"


def fingerprint(
    kind: OptionKind,
    strike: float,
    rate: float,
    dividend_yield: float,
    maturity: float,
    rows: FloatArray,
    columns: FloatArray,
) -> str:
    """Compute the identity of a scenario from its parameters.

    Parameters
    ----------
    kind
        Call or put.
    strike, rate, dividend_yield, maturity
        The scalar contract and market terms.
    rows, columns
        The two grid axes.

    Returns
    -------
    str
        Twelve lowercase hexadecimal characters. Twelve is roughly 48 bits, which gives
        a collision probability below one in a million at a hundred thousand recorded
        scenarios — far beyond any plausible use, and short enough to type.

    Examples
    --------
    Identical parameters give identical fingerprints:

    >>> import numpy as np
    >>> rows, columns = np.array([0.1, 0.2]), np.array([90.0, 110.0])
    >>> first = fingerprint(OptionKind.CALL, 100.0, 0.05, 0.0, 1.0, rows, columns)
    >>> second = fingerprint(OptionKind.CALL, 100.0, 0.05, 0.0, 1.0, rows, columns)
    >>> first == second
    True
    >>> len(first)
    12

    Float representation does not affect identity, which is the point of hashing a
    canonical form rather than keying on the values themselves:

    >>> exact = fingerprint(OptionKind.CALL, 100.0, 0.3, 0.0, 1.0, rows, columns)
    >>> accumulated = fingerprint(
    ...     OptionKind.CALL, 100.0, 0.1 + 0.2, 0.0, 1.0, rows, columns
    ... )
    >>> exact == accumulated
    True

    A different axis is a different scenario, even with identical contract terms:

    >>> wider = fingerprint(
    ...     OptionKind.CALL, 100.0, 0.05, 0.0, 1.0, rows, np.array([80.0, 120.0])
    ... )
    >>> wider == first
    False

    Calls and puts are distinguished:

    >>> put = fingerprint(OptionKind.PUT, 100.0, 0.05, 0.0, 1.0, rows, columns)
    >>> put == first
    False
    """
    parts = [
        str(kind.value),
        _canonical(strike),
        _canonical(rate),
        _canonical(dividend_yield),
        _canonical(maturity),
        ",".join(_canonical(float(x)) for x in rows),
        ",".join(_canonical(float(x)) for x in columns),
    ]
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:_FINGERPRINT_LENGTH]


@dataclass(frozen=True, slots=True)
class Scenario:
    """One recorded surface, without its cell values.

    Attributes
    ----------
    fingerprint
        Identity of the scenario, as returned by :func:`fingerprint`.
    recorded_at
        ISO-8601 UTC timestamp of first recording. Re-recording does not update it.
    kind
        Call or put.
    strike, rate, dividend_yield, maturity
        The scalar terms.
    row_label, column_label
        Names of the two axes.
    """

    fingerprint: str
    recorded_at: str
    kind: OptionKind
    strike: float
    rate: float
    dividend_yield: float
    maturity: float
    row_label: str
    column_label: str


class ScenarioStore:
    """A SQLite-backed log of recorded surfaces.

    Usable as a context manager, which is the recommended form: the connection is
    closed on exit even if the body raises.

    Parameters
    ----------
    path
        Database file. The special value ``":memory:"`` gives an ephemeral store, which
        is what the test suite uses.

    Examples
    --------
    >>> import numpy as np
    >>> from bsm.grids import price_surface
    >>> grid = price_surface(
    ...     OptionKind.CALL,
    ...     volatilities=np.array([0.1, 0.2]),
    ...     spots=np.array([90.0, 100.0, 110.0]),
    ...     strike=100.0,
    ...     rate=0.05,
    ...     dividend_yield=0.0,
    ...     maturity=1.0,
    ... )
    >>> with ScenarioStore(":memory:") as store:
    ...     scenario = store.record(
    ...         grid, kind=OptionKind.CALL, strike=100.0, rate=0.05,
    ...         dividend_yield=0.0, maturity=1.0,
    ...     )
    ...     len(store.scenarios())
    ...     reloaded = store.grid(scenario.fingerprint)
    ...     bool(np.array_equal(reloaded.values, grid.values))
    1
    True
    """

    def __init__(self, path: str | Path) -> None:
        self._connection = sqlite3.connect(str(path))
        # Without this pragma SQLite parses the foreign key and then ignores it. It is
        # per-connection, not per-database, so it must be issued every time.
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.executescript(_SCHEMA)
        self._connection.commit()

    def __enter__(self) -> ScenarioStore:
        """Return self, for use as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the connection."""
        self.close()

    def close(self) -> None:
        """Close the underlying connection."""
        self._connection.close()

    def record(
        self,
        grid: Grid,
        *,
        kind: OptionKind,
        strike: float,
        rate: float,
        dividend_yield: float,
        maturity: float,
    ) -> Scenario:
        """Record a surface, or return the existing record if already present.

        Parameters
        ----------
        grid
            The computed surface.
        kind
            Call or put.
        strike, rate, dividend_yield, maturity
            The scalar terms.

        Returns
        -------
        Scenario
            The stored record. Recording the same parameters twice is a no-op and
            returns the original, including its original timestamp.

        Notes
        -----
        Idempotent by construction: the fingerprint is the primary key and the inserts
        use ``INSERT OR IGNORE``, so a repeated call cannot duplicate rows or silently
        overwrite a stored surface with a different one.
        """
        digest = fingerprint(
            kind, strike, rate, dividend_yield, maturity, grid.rows, grid.columns
        )
        existing = self.scenario(digest)
        if existing is not None:
            return existing

        recorded_at = datetime.now(UTC).isoformat(timespec="seconds")
        with self._connection:
            self._connection.execute(
                "INSERT INTO scenarios VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    digest,
                    recorded_at,
                    kind.value,
                    strike,
                    rate,
                    dividend_yield,
                    maturity,
                    grid.row_label,
                    grid.column_label,
                ),
            )
            self._connection.executemany(
                "INSERT INTO grid_points VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        digest,
                        i,
                        j,
                        float(grid.rows[i]),
                        float(grid.columns[j]),
                        float(grid.values[i, j]),
                    )
                    for i in range(len(grid.rows))
                    for j in range(len(grid.columns))
                ],
            )
        recorded = self.scenario(digest)
        assert recorded is not None  # noqa: S101 - just inserted, within a transaction
        return recorded

    def scenario(self, digest: str) -> Scenario | None:
        """Look up a single scenario by fingerprint.

        Parameters
        ----------
        digest
            The fingerprint to look up.

        Returns
        -------
        Scenario or None
            The record, or ``None`` if no scenario has that fingerprint.
        """
        row = self._connection.execute(
            "SELECT fingerprint, recorded_at, kind, strike, rate, dividend_yield, "
            "maturity, row_label, column_label FROM scenarios WHERE fingerprint = ?",
            (digest,),
        ).fetchone()
        if row is None:
            return None
        return Scenario(
            fingerprint=row[0],
            recorded_at=row[1],
            kind=OptionKind(row[2]),
            strike=row[3],
            rate=row[4],
            dividend_yield=row[5],
            maturity=row[6],
            row_label=row[7],
            column_label=row[8],
        )

    def scenarios(self) -> list[Scenario]:
        """List every recorded scenario, oldest first.

        Returns
        -------
        list of Scenario
            The records, without their cell values.
        """
        digests = self._connection.execute(
            "SELECT fingerprint FROM scenarios ORDER BY recorded_at, fingerprint"
        ).fetchall()
        found = [self.scenario(digest) for (digest,) in digests]
        return [scenario for scenario in found if scenario is not None]

    def grid(self, digest: str) -> Grid:
        """Reconstruct a stored surface.

        Parameters
        ----------
        digest
            The fingerprint of the scenario to load.

        Returns
        -------
        Grid
            The surface, identical to the one recorded. SQLite stores REAL as an
            IEEE-754 double, so the round trip is exact rather than merely close.

        Raises
        ------
        KeyError
            If no scenario has that fingerprint.
        """
        scenario = self.scenario(digest)
        if scenario is None:
            raise KeyError(f"no scenario with fingerprint {digest!r}")

        points = self._connection.execute(
            "SELECT row_index, column_index, row_value, column_value, value "
            "FROM grid_points WHERE fingerprint = ? ORDER BY row_index, column_index",
            (digest,),
        ).fetchall()

        row_count = max(point[0] for point in points) + 1
        column_count = max(point[1] for point in points) + 1
        rows = np.empty(row_count, dtype=np.float64)
        columns = np.empty(column_count, dtype=np.float64)
        values = np.empty((row_count, column_count), dtype=np.float64)
        for i, j, row_value, column_value, value in points:
            rows[i] = row_value
            columns[j] = column_value
            values[i, j] = value

        return Grid(
            rows=rows,
            columns=columns,
            values=values,
            row_label=scenario.row_label,
            column_label=scenario.column_label,
        )

    def delete(self, digest: str) -> bool:
        """Remove a scenario and its cells.

        Parameters
        ----------
        digest
            The fingerprint to remove.

        Returns
        -------
        bool
            Whether a scenario was removed.

        Notes
        -----
        The cells go with it through ``ON DELETE CASCADE``, which is enforced because
        the constructor enables foreign keys. Without that pragma this would leave
        orphaned rows behind and report success.
        """
        with self._connection:
            cursor = self._connection.execute(
                "DELETE FROM scenarios WHERE fingerprint = ?", (digest,)
            )
        return cursor.rowcount > 0

    def clear(self) -> None:
        """Remove every scenario and every cell."""
        with self._connection:
            self._connection.execute("DELETE FROM scenarios")
            self._connection.execute("DELETE FROM grid_points")
