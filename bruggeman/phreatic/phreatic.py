"""A. Phreatic flow (solutions 10-100)

From Bruggeman (1999), Section A, solutions 10-100.
"""

import numpy as np
from numpy.typing import NDArray

from bruggeman.latexify import latexify_function


@latexify_function(identifiers={"bruggeman_21_11": "h"}, reduce_assignments=True)
def bruggeman_21_11(
    x: float | NDArray[np.float64],
    b: float,
    k: float,
    H: float,
    p: float = 1.0,
) -> float | NDArray[np.float64]:
    """21.11 Confined phreatic aquifer with horizontal 1D-flow.

    Flow caused by precipitation through an infinite strip of
    width 2b, bounded at both sides by open water with equal level

    From Bruggeman 21.11

    Parameters
    ----------
    x : float or ndarray
        Distance from the center of the strip [m]
    b : float
        Half-width of the strip [m]
    k : float
        Hydraulic conductivity [m/d]
    H : float
        Head in the open water [m]
    p : float
        Arbitrary constant precipitation [m/d]

    Returns
    -------
    head: float
        Hydraulic head at distance x [m]
    """

    return np.sqrt(H**2 + p / k * (b**2 - x**2))
