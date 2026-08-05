"""Type aliases and helpers for the scalar/array boundary.

Every mathematical function in this package accepts either Python floats or NumPy
arrays, and returns whichever the caller supplied. That dual behaviour is what lets the
same code price one option interactively and fill a 200x200 sensitivity grid, with no
second implementation to keep in step.

The mechanism is uniform and lives here: convert inputs with :func:`as_array`, compute
unconditionally on arrays, then hand the result to :func:`scalarize`, which returns a
plain float if and only if every input was scalar.

This module is private. Its names are not part of the public API.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]
"""A NumPy array of double-precision floats."""

BoolArray: TypeAlias = npt.NDArray[np.bool_]
"""A NumPy array of booleans, used for masks over degenerate grid cells."""

Numeric: TypeAlias = float | FloatArray
"""Either a Python float or a float array, the argument type of every public formula."""

__all__ = ["BoolArray", "FloatArray", "Numeric", "as_array", "scalarize"]


def as_array(value: Numeric) -> FloatArray:
    """Coerce a scalar or array to a float64 array.

    Parameters
    ----------
    value
        Scalar or array-like input.

    Returns
    -------
    FloatArray
        A float64 array. Scalars become zero-dimensional arrays, which broadcast
        against arrays of any shape.

    Examples
    --------
    >>> as_array(2.0).shape
    ()
    >>> as_array(np.array([1.0, 2.0])).dtype
    dtype('float64')
    """
    return np.asarray(value, dtype=np.float64)


def scalarize(result: FloatArray, *inputs: Numeric) -> Numeric:
    """Return ``result`` as a float when every input was scalar, else as an array.

    Parameters
    ----------
    result
        The computed array.
    *inputs
        The original arguments, inspected only for their types.

    Returns
    -------
    Numeric
        ``float(result)`` if all of ``inputs`` were Python or NumPy scalars, otherwise
        ``result`` unchanged.

    Notes
    -----
    ``np.float64`` subclasses ``float``, so a NumPy scalar counts as scalar and a
    caller who passes one gets a float back rather than a zero-dimensional array.

    Examples
    --------
    >>> scalarize(as_array(3.0), 1.0, 2.0)
    3.0
    >>> scalarize(as_array([3.0]), 1.0, np.array([2.0]))
    array([3.])
    """
    if all(isinstance(value, float | int) for value in inputs):
        return float(result)
    return result
