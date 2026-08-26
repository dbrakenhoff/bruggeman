"""BII: Two-dimensional radial-symmetric flow (solutions 200).

This module contains Bruggeman's analytical solutions for radial-symmetric
two-dimensional flow in confined systems.

From Bruggeman (1999), Section BII, solutions 200.
"""

import numpy as np
from scipy.special import exp1 as _exp1
from scipy.special import i0 as _i0
from scipy.special import i1 as _i1
from scipy.special import k0 as _k0
from scipy.special import k1 as _k1

from bruggeman.funcs import W
from bruggeman.latexify import latexify_function


@latexify_function(
    identifiers={
        "bruggeman_244_02": "varphi",
        "_i0": "I_0",
        "_i1": "I_1",
        "_k0": "K_0",
        "_k1": "K_1",
        "head": "varphi",
        "denominator": "Delta",
    },
    reduce_assignments=False,
)
def bruggeman_244_02(
    r: float | np.ndarray,
    R: float,
    h: float,
    k: float,
    D: float,
    c1: float,
    c2: float,
) -> float | np.ndarray:
    """244.02 Circular basin with different permeable layers for basin and polder.

    Radially symmetric steady-state flow with two regions:
    1) inside the basin (0 <= r <= R) with leakage resistance c1
    2) outside the basin (r >= R) with leakage resistance c2

    From Bruggeman 244.02

    Parameters
    ----------
    r : float or np.ndarray
        radial distance from center of the basin [L]
    R : float
        radius of the basin [L]
    h : float
        head in the basin relative to the surrounding polder [L]
    k : float
        hydraulic conductivity of the aquifer [L/T]
    D : float
        aquifer thickness [L]
    c1 : float
        leakage resistance inside basin [T]
    c2 : float
        leakage resistance outside basin [T]

    Returns
    -------
    head : float or np.ndarray
        hydraulic head at radial distance r [L]
    """
    r_arr = np.asarray(r)

    lambda_1 = np.sqrt(k * D * c1)
    lambda_2 = np.sqrt(k * D * c2)

    denominator = lambda_1 * _i0(R / lambda_1) * _k1(R / lambda_2) + lambda_2 * _i1(
        R / lambda_1
    ) * _k0(R / lambda_2)
    phi_1 = h - h * lambda_1 * _k1(R / lambda_2) * _i0(r_arr / lambda_1) / denominator
    phi_2 = h * lambda_2 * _i1(R / lambda_1) * _k0(r_arr / lambda_2) / denominator

    head = np.where(r_arr <= R, phi_1, phi_2)
    if np.ndim(r_arr) == 0:
        return float(head)
    return head


@latexify_function(
    identifiers={
        "bruggeman_215_03": "varphi",
        "_exp1": "E_1",
    },
    reduce_assignments=False,
)
def bruggeman_215_03(
    r: float | np.ndarray,
    t: float | np.ndarray,
    Q0: float,
    k: float,
    D: float,
    S: float,
) -> float | np.ndarray:
    """215.03 Continuous abstraction from a line source with constant discharge Q(t) = Q0.

    Parameters
    ----------
    r : float or np.ndarray
        radial distance from the well [L]
    t : float or np.ndarray
        time since pumping started [T]
    Q0 : float
        constant discharge [L^3/T]
    k : float
        hydraulic conductivity of the aquifer [L/T]
    D : float
        aquifer thickness [L]
    S : float
        storage coefficient [-]

    Returns
    -------
    varphi : float or np.ndarray
        drawdown at radius r and time t [L]
    """
    beta = np.sqrt(S / (k * D))
    return Q0 / (4 * np.pi * k * D) * _exp1((beta * r) ** 2 / (4 * t))


@latexify_function(
    identifiers={
        "bruggeman_215_13": "varphi",
        "W": "W",
        "lambda_": "lambda",
    },
    reduce_assignments=False,
)
def bruggeman_215_13(
    r: float | np.ndarray,
    t: float | np.ndarray,
    Q0: float,
    k: float,
    D: float,
    c: float,
    eta: float,
) -> float | np.ndarray:
    """215.13 Hantush solution for constant discharge to a leaky confined aquifer.

    From Bruggeman 215.13

    Parameters
    ----------
    r : float or np.ndarray
        radial distance from the well [L]
    t : float or np.ndarray
        time since pumping started [T]
    Q0 : float
        constant discharge [L^3/T]
    k : float
        hydraulic conductivity of the aquifer [L/T]
    D : float
        aquifer thickness [L]
    c : float
        leakage resistance [T]
    eta : float
        non-steady leakage parameter [T]

    Returns
    -------
    varphi : float or np.ndarray
        drawdown at radius r and time t [L]
    """
    lambda_ = np.sqrt(k * D * c)
    return Q0 / (4 * np.pi * k * D) * W(eta * t, r / lambda_)


@latexify_function(
    identifiers={
        "bruggeman_215_14": "varphi",
        "lambda_": "lambda",
        "_k0": "K_0",
    },
    reduce_assignments=False,
)
def bruggeman_215_14(
    r: float | np.ndarray,
    Q0: float,
    k: float,
    D: float,
    c: float,
) -> float | np.ndarray:
    """215.14 Steady-state Hantush solution for a leaky confined aquifer.

    From Bruggeman 215.14

    Parameters
    ----------
    r : float or np.ndarray
        radial distance from the well [L]
    Q0 : float
        constant discharge [L^3/T]
    k : float
        hydraulic conductivity of the aquifer [L/T]
    D : float
        aquifer thickness [L]
    c : float
        leakage resistance [T]

    Returns
    -------
    varphi : float or np.ndarray
        steady-state drawdown at radius r [L]
    """
    lambda_ = np.sqrt(k * D * c)
    return Q0 / (2 * np.pi * k * D) * _k0(r / lambda_)


@latexify_function(
    identifiers={
        "bruggeman_241_25": "varphi",
        "lambda_": "lambda",
        "_k0": "K_0",
        "_i0": "I_0",
    },
    reduce_assignments=False,
)
def bruggeman_241_25(
    r: float | np.ndarray,
    R: float,
    Q0: float,
    k: float,
    D: float,
    c: float,
) -> float | np.ndarray:
    """241.25 Steady-state solution for a pumping well at the centre of a circular island with open vertical boundary at r = R. Constant discharge.

    From Bruggeman 241.25

    Parameters
    ----------
    r : float or np.ndarray
        radial distance from the well [L]
    R : float
        open vertical boundary at r = R [L]
    Q0 : float
        constant discharge [L^3/T]
    k : float
        hydraulic conductivity of the aquifer [L/T]
    D : float
        aquifer thickness [L]
    c : float
        leakage resistance [T]

    Returns
    -------
    varphi : float or np.ndarray
        steady-state drawdown at radius r [L]
    """
    lambda_ = np.sqrt(k * D * c)
    return (
        Q0
        / (2 * np.pi * k * D)
        * (_k0(r / lambda_) - _k0(R / lambda_) / _i0(R / lambda_) * _i0(r / lambda_))
    )


@latexify_function(
    identifiers={
        "bruggeman_241_27": "varphi",
        "lambda_": "lambda",
        "_k0": "K_0",
        "_k1": "K_1",
        "_i0": "I_0",
        "_i1": "I_1",
    },
    reduce_assignments=False,
)
def bruggeman_241_27(
    r: float | np.ndarray,
    R: float,
    Q0: float,
    k: float,
    D: float,
    c: float,
) -> float | np.ndarray:
    """241.27 Steady-state solution for a pumping well at the centre of a circular island with closed vertical boundary at r = R. Constant discharge.

    From Bruggeman 241.27

    Parameters
    ----------
    r : float or np.ndarray
        radial distance from the well [L]
    R : float
        outer radius of the aquifer boundary [L]
    Q0 : float
        constant discharge [L^3/T]
    k : float
        hydraulic conductivity of the aquifer [L/T]
    D : float
        aquifer thickness [L]
    c : float
        leakage resistance [T]

    Returns
    -------
    varphi : float or np.ndarray
        steady-state drawdown at radius r [L]
    """
    lambda_ = np.sqrt(k * D * c)
    return (
        Q0
        / (2 * np.pi * k * D)
        * (_k0(r / lambda_) + _k1(R / lambda_) / _i1(R / lambda_) * _i0(r / lambda_))
    )
