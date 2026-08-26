"""Edelman solution for sudden rise in water level in a semi-infinite aquifer."""

import numpy as np
from numpy.typing import NDArray
from scipy.special import erfc as _erfc

from bruggeman.latexify import latexify_function


@latexify_function(
    identifiers={"h_edelman": "varphi"},
    reduce_assignments=True,
    escape_underscores=False,
)
def h_edelman(
    x: float | NDArray[np.float64],
    t: float | NDArray[np.float64],
    T: float,
    S: float,
    h: float,
    t_0: float = 0.0,
) -> float | NDArray[np.float64]:
    """Edelman solution: Head in a semi-infinite aquifer after sudden rise.

    From Analytical Groundwater Modeling, ch. 5.

    Parameters
    ----------
    x : float or ndarray
        Distance from the boundary [L]
    t : float or ndarray
        Time since the rise [T]
    T : float
        Transmissivity [L^2/T]
    S : float
        Storage coefficient [-]
    h : float
        Rise of the water level [L]
    t_0 : float
        Time offset [T], by default 0.0

    Returns
    -------
    head : float or ndarray
        Hydraulic head at distance x and time t [L]
    """
    # from Analyical Groundwater Modeling, ch. 5
    u = np.sqrt(S * x**2 / (4 * T * (t - t_0)))
    return h * _erfc(u)


@latexify_function(
    identifiers={"Qx_edelman": "Q_x"},
    reduce_assignments=True,
    escape_underscores=False,
)
def Qx_edelman(
    x: float | NDArray[np.float64],
    t: float | NDArray[np.float64],
    T: float,
    S: float,
    h: float,
    t_0: float = 0.0,
) -> float | NDArray[np.float64]:
    """Edelman solution: Discharge in a semi-infinite aquifer after sudden rise.

    From Analytical Groundwater Modeling, ch. 5.

    Parameters
    ----------
    x : float or ndarray
        Distance from the boundary [L]
    t : float or ndarray
        Time since the rise [T]
    T : float
        Transmissivity [L^2/T]
    S : float
        Storage coefficient [-]
    h : float
        Rise of the water level [L]
    t_0 : float
        Time offset [T], by default 0.0

    Returns
    -------
    Qx : float or ndarray
        Discharge at distance x and time t [L^2/T]
    """
    # from Analyical Groundwater Modeling, ch. 5
    u = np.sqrt(S * x**2 / (4 * T * (t - t_0)))
    return T * h * 2 * u / (x * np.sqrt(np.pi)) * np.exp(-(u**2))
