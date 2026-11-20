"""Mazure: seepage into a polder next to open water with a constant level.

The steady state of Bruggeman 123.33: a leaky aquifer below a polder, bounded at x = 0 by
open water whose level differs h from the polder level. The head in the aquifer is
``h * exp(-x / lambda)``, with ``lambda = sqrt(k * D * c)``, and the seepage through the leaky
layer is that head divided by the resistance c (Mazure).

From Mazure, J.P. (1936). Kwel en chloorbezwaar in de Wieringermeer. Bijlage 10 in:
Geo-hydrologische gesteldheid van de Wieringermeer. Rapporten en Mededeelingen betreffende
de Zuiderzeewerken, No. 5.
"""

import numpy as np
from numpy.typing import NDArray

from bruggeman.latexify import latexify_function


@latexify_function(
    identifiers={"q_mazure": "q", "lambda_": "lambda"},
    reduce_assignments=False,
)
def q_mazure(
    x: float | NDArray[np.float64],
    h: float,
    k: float,
    D: float,
    c: float,
) -> float | NDArray[np.float64]:
    """Mazure: seepage through the leaky layer at distance x from the open water.

    ``q(x) = (H(x) - H_polder) / c = (H_0 - H_polder) / c * exp(-x / lambda)``: the head of
    Bruggeman 123.33 divided by the resistance of the leaky layer.

    From Mazure (1936), see the module docstring.

    Parameters
    ----------
    x : float or ndarray
        Distance from the open water [m]
    h : float
        Level of the open water relative to the polder level, H_0 - H_polder [m]
    k : float
        Hydraulic conductivity of the aquifer [m/d]
    D : float
        Aquifer thickness [m]
    c : float
        Resistance of the leaky layer [d]

    Returns
    -------
    q : float or ndarray
        Seepage per unit area at distance x [m/d] (positive: upward, into the polder)
    """
    lambda_ = np.sqrt(k * D * c)
    return h / c * np.exp(-x / lambda_)


@latexify_function(identifiers={"Q_mazure": "Q"}, reduce_assignments=True)
def Q_mazure(
    h: float,
    k: float,
    D: float,
    c: float,
) -> float:
    """Mazure: total seepage into the polder per unit length of the open water.

    The seepage `q_mazure` integrated over x from 0 to infinity:
    ``Q = (H_0 - H_polder) / c * lambda = (H_0 - H_polder) * sqrt(k * D / c)``.

    From Mazure (1936), see the module docstring.

    Parameters
    ----------
    h : float
        Level of the open water relative to the polder level, H_0 - H_polder [m]
    k : float
        Hydraulic conductivity of the aquifer [m/d]
    D : float
        Aquifer thickness [m]
    c : float
        Resistance of the leaky layer [d]

    Returns
    -------
    Q : float
        Total seepage per unit length of the open water [m^2/d]
    """
    return h * np.sqrt(k * D / c)
