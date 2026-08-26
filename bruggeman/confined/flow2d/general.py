"""BIII: General two-dimensional flow (solutions 300)

This module contains Bruggeman's analytical solutions for general two-dimensional
flow in confined systems.

From Bruggeman (1999), Section BIII, solutions 300.
"""

import numpy as np
from mpmath import ellipf, ellipk

from bruggeman.latexify import latexify_function


@latexify_function(
    identifiers={"bruggeman_355_19": "omega", "ellipf": "F", "ellipk": "K"},
    reduce_assignments=False,
)
def bruggeman_355_19(
    x: float | np.ndarray,
    z: float | np.ndarray,
    L: float,
    B: float,
    h: float,
    k: float,
    D: float,
) -> np.ndarray:
    """355.19 Drainage canal on a confined aquifer of finite thickness near open boundary.

    Constant drawdown of the water level in the canal.

    Parameters
    ----------
    x : float or np.ndarray
        distance from open boundary [L]
    z : float or np.ndarray
        depth below the top of the aquifer [L]
    L : float
        distance from open boundary to the middle of the canal [L]
    B : float
        half-width of the canal [L]
    h : float
        drawdown in the canal [L]
    k : float
        hydraulic conductivity of the aquifer [L/T]
    D : float
        thickness of the aquifer [L]

    Returns
    -------
    omega :
        complex potential at (x, z)
    """
    zeta = x + z * 1j
    w = np.tanh(np.pi * zeta / (2 * D)) / np.tanh(np.pi * (L - B) / (2 * D))
    m = (np.tanh(np.pi * (L - B) / (2 * D)) / np.tanh(np.pi * (L + B) / (2 * D))) ** 2
    return k * h / ellipk(m) * ellipf(np.arcsin(w), m)


@latexify_function(
    identifiers={"bruggeman_355_19_total_discharge": "q", "ellipf": "F", "ellipk": "K"},
    reduce_assignments=True,
)
def bruggeman_355_19_total_discharge(
    L: float, B: float, h: float, k: float, D: float
) -> float:
    """355.19 Total discharge to the canal in Bruggeman 355-19.

    Parameters
    ----------
    L : float
        distance from open boundary to the middle of the canal [L]
    B : float
        half-width of the canal [L]
    h : float
        drawdown in the canal [L]
    k : float
        hydraulic conductivity of the aquifer [L/T]
    D : float
        thickness of the aquifer [L]

    Returns
    -------
    q :
        total discharge to the canal [L^2/T]
    """
    m = (np.tanh(np.pi * (L - B) / (2 * D)) / np.tanh(np.pi * (L + B) / (2 * D))) ** 2
    return k * h * ellipk(1 - m) / ellipk(m)
