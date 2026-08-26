"""BI: Confined groundwater - one-dimensional flow (solutions 100).

This module contains Bruggeman's analytical solutions for one-dimensional
flow in confined systems.

From Bruggeman (1999), Section BI, solutions 100.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.special import erfc

from bruggeman.funcs import P, ierfc
from bruggeman.latexify import latexify_function


@latexify_function(identifiers={"bruggeman_123_02": "varphi"}, reduce_assignments=True)
def bruggeman_123_02(
    x: float | NDArray[np.float64],
    t: float | NDArray[np.float64],
    h: float,
    k: float,
    D: float,
    S: float,
) -> float | NDArray[np.float64]:
    """123.02 Solution for sudden rise of the water table in a confined aquifer.

    From Bruggeman 123.02

    Parameters
    ----------
    x : float or ndarray
        Distance from the boundary [m]
    t : float or ndarray
        Time since the start of the rise [d]
    h : float
        Rise of the water table [m]
    k : float
        Hydraulic conductivity [m/d]
    D : float
        Aquifer thickness [m]
    S : float
        Storage coefficient [-]

    Returns
    -------
    head : float
        head in the aquifer at distance x and time t [m]
    """
    beta = np.sqrt(S / (k * D))
    u = beta * x / (2 * np.sqrt(t))
    return h * erfc(u)


@latexify_function(identifiers={"bruggeman_123_03": "varphi"}, reduce_assignments=True)
def bruggeman_123_03(
    x: float | NDArray[np.float64],
    t: float | NDArray[np.float64],
    a: float,
    k: float,
    D: float,
    S: float,
) -> float | NDArray[np.float64]:
    """123.03 Solution for linear rise of the water table in a confined aquifer.

    From Bruggeman 123.03

    Parameters
    ----------
    x : float or ndarray
        Distance from the boundary [m]
    t : float or ndarray
        Time since the start of the rise [d]
    a : float
        Slope of linear rise of the water table [m/d]
    k : float
        Hydraulic conductivity [m/d]
    D : float
        Aquifer thickness [m]
    S : float
        Storage coefficient [-]

    Returns
    -------
    head : float
        head in the aquifer at distance x and time t [m]
    """
    beta = np.sqrt(S / (k * D))
    u = beta * x / (2 * np.sqrt(t))
    return a * t * ierfc(u, 2) / ierfc(0, 2)


@latexify_function(
    identifiers={"bruggeman_123_05_q": "varphi"}, reduce_assignments=False
)
def bruggeman_123_05_q(
    x: float | NDArray[np.float64],
    t: float | NDArray[np.float64],
    Q: float,
    k: float,
    D: float,
    S: float,
) -> float | NDArray[np.float64]:
    """123.05 Solution for constant infiltration/pumping in a confined aquifer.

    From Olsthoorn, Th. 2006. Van Edelman naar Bruggeman. Stromingen 12 (2006) p5-11.

    Parameters
    ----------
    x : float or ndarray
        Distance from the boundary [m]
    t : float or ndarray
        Time since the start of the rise [d]
    Q : float
        Infiltration (positive) or pumping (negative) rate [m^3/d]
    k : float
        Hydraulic conductivity [m/d]
    D : float
        Aquifer thickness [m]
    S : float
        Storage coefficient [-]

    Returns
    -------
    head : float
        head in the aquifer at distance x and time t [m]
    """
    beta = np.sqrt(S / (k * D))
    u = beta * x / (2 * np.sqrt(t))
    return 2 * Q * np.sqrt(t) / np.sqrt(k * D * S) * ierfc(u, 1) / (ierfc(0, 0))


@latexify_function(
    identifiers={
        "bruggeman_123_32": "varphi",
        "lambda_": "lambda",
        "P": "P",
    },
    reduce_assignments=False,
)
def bruggeman_123_32(
    x: float | NDArray[np.float64],
    t: float | NDArray[np.float64],
    h: float,
    k: float,
    D: float,
    c: float,
    eta: float,
) -> float | NDArray[np.float64]:
    """123.32 Sudden drawdown of the surface water level, which is kept constant thereafter.

    From Bruggeman 123.32

    Parameters
    ----------
    x : float or ndarray
        Distance from the boundary [m]
    t : float or ndarray
        Time since the drawdown [d]
    h : float
        Drawdown height [m]
    k : float
        Hydraulic conductivity [m/d]
    D : float
        Aquifer thickness [m]
    c : float
        Leakance [d]
    eta : float
        non-steady leakage parameter [d]
    Returns
    -------
    head : float
        head in the aquifer at distance x and time t [m]
    """
    lambda_ = np.sqrt(k * D * c)
    return h * P(x / (2 * lambda_), np.sqrt(eta * t))


@latexify_function(
    identifiers={"bruggeman_123_33": "varphi", "lambda_": "lambda"},
    reduce_assignments=False,
)
def bruggeman_123_33(
    x: float | NDArray[np.float64],
    h: float,
    k: float,
    D: float,
    c: float,
) -> float | NDArray[np.float64]:
    """123.33 Steady state of Bruggeman 123.32.

    From Bruggeman 123.33

    Parameters
    ----------
    x : float or ndarray
        Distance from the boundary [m]
    h : float
        Drawdown height [m]
    k : float
        Hydraulic conductivity [m/d]
    D : float
        Aquifer thickness [m]
    c : float
        Leakance [d]

    Returns
    -------
    head : float
        steady state head in the aquifer at distance x [m]
    """
    lambda_ = np.sqrt(k * D * c)
    return h * np.exp(-x / lambda_)


@latexify_function(
    identifiers={
        "bruggeman_126_33": "varphi",
        "lambda_": "lambda",
    },
    reduce_assignments=False,
)
def bruggeman_126_33(
    x: float | NDArray[np.float64],
    h: float,
    k: float,
    D: float,
    c: float,
    w: float,
) -> float | NDArray[np.float64]:
    """126.33 Leaky aquifer with entrance resistance. Steady state after head change.

    From Bruggeman 126.33

    Parameters
    ----------
    x : float or ndarray
        Distance from the boundary [m]
    h : float or ndarray
        Rise of the water table [m]
    k : float
        Hydraulic conductivity [m/d]
    D : float
        Aquifer thickness [m]
    c : float
        Leakance [d]
    w : float
        Entry resistance at x=0 [d]

    Returns
    -------
    head : float
        steady state head in the aquifer at distance x [m]
    """
    lambda_ = np.sqrt(k * D * c)
    return h * lambda_ / (k * w + lambda_) * np.exp(-x / lambda_)


@latexify_function(
    identifiers={"bruggeman_128_01": "varphi"},
    reduce_assignments=False,
    escape_underscores=False,
)
def bruggeman_128_01(
    x: float | NDArray[np.float64],
    t: float | NDArray[np.float64],
    h: float,
    S: float,
    k: float,
    D: float,
    tau: float,
) -> float | NDArray[np.float64]:
    """128.01 Tidal fluctuation open water, confined aquifer with open boundary (x = 0).

    From Bruggeman 128.01

    Parameters
    ----------
    x : float or ndarray
        Distance from the boundary [m]
    t : float or ndarray
        time [d]
    h : float
        amplitude of tidal fluctuation [m]
    S : float
        storage coefficient [-]
    k : float
        hydraulic conductivity [m/d]
    D : float
        aquifer thickness [m]
    tau : float
        tidal period [d]

    Returns
    -------
    head : float
        head in the aquifer at distance x and time t [m]
    """
    beta = np.sqrt(S / (k * D))
    omega = 2 * np.pi / tau
    omega_p = beta * np.sqrt(omega / 2)
    return h * np.exp(-omega_p * x) * np.sin(omega * t - omega_p * x)


@latexify_function(
    identifiers={"bruggeman_128_03": "varphi", "j": "i", "real": "Re", "imag": "Im"},
    reduce_assignments=False,
)
def bruggeman_128_03(
    x: float | NDArray[np.float64],
    t: float | NDArray[np.float64],
    h: float,
    S: float,
    k: float,
    D: float,
    tau: float,
    c: float,
) -> float | NDArray[np.float64]:
    """128.03 Tidal fluctuation open water, leaky aquifer with open boundary (x = 0).

    From Bruggeman 128.03

    Parameters
    ----------
    x : float or ndarray
        Distance from the boundary [m]
    t : float or ndarray
        time [d]
    h : float
        amplitude of tidal fluctuation [m]
    S : float
        storage coefficient [-]
    k : float
        hydraulic conductivity [m/d]
    D : float
        aquifer thickness [m]
    tau : float
        tidal period [d]
    c : float
        leakance [d]

    Returns
    -------
    head : float
        head in the aquifer at distance x and time t [m]
    """
    beta = np.sqrt(S / (k * D))
    eta = 1 / (c * S)
    omega = 2 * np.pi / tau

    i = 1j
    a = np.real(np.sqrt(eta + i * omega))
    b = np.imag(np.sqrt(eta + i * omega))

    return h * np.exp(-beta * a * x) * np.sin(omega * t - beta * b * x)


@latexify_function(
    identifiers={
        "bruggeman_128_04": "varphi",
        "theta": "vartheta",
        "j": "i",  # not working :(
        "real": "Re",
        "imag": "Im",
    },
    reduce_assignments=False,
)
def bruggeman_128_04(
    x: float | NDArray[np.float64],
    t: float | NDArray[np.float64],
    h: float,
    S: float,
    k: float,
    D: float,
    tau: float,
    c: float,
    w: float,
) -> float | NDArray[np.float64]:
    """128.04 Tidal fluctuation open water, leaky aquifer with entrance resistance (x = 0).

    From Bruggeman 128.04

    Parameters
    ----------
    x : float or ndarray
        Distance from the boundary [m]
    t : float or ndarray
        time [d]
    h : float
        amplitude of tidal fluctuation [m]
    S : float
        storage coefficient [-]
    k : float
        hydraulic conductivity [m/d]
    D : float
        aquifer thickness [m]
    tau : float
        tidal period [d]
    c : float
        leakance [d]
    w : float
        entry resistance at x=0 [d]

    Returns
    -------
    head : float
        head in the aquifer at distance x and time t [m]
    """
    beta = np.sqrt(S / (k * D))
    eta = 1 / (c * S)
    omega = 2 * np.pi / tau
    theta = 1 / (beta**2 * k**2 * w**2)

    i = 1j
    a = np.real(np.sqrt(eta + i * omega))
    b = np.imag(np.sqrt(eta + i * omega))

    return (
        h
        * np.sqrt(theta)
        * np.exp(-beta * a * x)
        * np.sin(omega * t - beta * b * x - np.arctan(b / (a + np.sqrt(theta))))
        / (np.sqrt((a + np.sqrt(theta)) ** 2 + b**2))
    )


@latexify_function(
    identifiers={"bruggeman_133_16": "varphi"},
    reduce_assignments=False,
)
def bruggeman_133_16(
    x: float | NDArray[np.float64],
    t: float | NDArray[np.float64],
    b: float,
    S: float,
    k: float,
    D: float,
    p: float = 1.0,
    N: int = 10,
) -> float | NDArray[np.float64]:
    """133.16 Confined aquifer with zero head at x=b, zero flux at x=0 and a constant arbitrary precipitation p.

    From Bruggeman 133.16

    Parameters
    ----------
    x : float or ndarray
        Distance from the boundary [m]
    t : float or ndarray
        Time [d]
    b : float
        Half width of the aquifer [m]
    S : float
        Storage coefficient [-]
    k : float
        Hydraulic conductivity [m/d]
    D : float
        Aquifer thickness [m]
    p : float
        Arbitrary constant precipitation [m/d]
    N : int
        Number of terms in the series expansion to approximate the infinite sum,
        by default 10 [-]

    Returns
    -------
    head : float
        Head in the aquifer at distance x and time t [m]
    """
    beta = np.sqrt(S / (k * D))

    return p / (2 * k * D) * (b**2 - x**2) - 16 * p * b**2 / (np.pi**3 * k * D) * sum(
        (-1) ** n
        / (2 * n + 1) ** 3
        * np.cos((2 * n + 1) * np.pi * x / (2 * b))
        * np.exp(-(((2 * n + 1) * np.pi / (2 * beta * b)) ** 2) * t)
        for n in range(N)
    )


@latexify_function(
    identifiers={"bruggeman_133_17": "varphi"},
    reduce_assignments=False,
)
def bruggeman_133_17(
    x: float | NDArray[np.float64],
    b: float,
    k: float,
    D: float,
    p: float = 1.0,
) -> float | NDArray[np.float64]:
    """133.17 Confined aquifer with zero head at x=b, zero flux at x=0 and a constant arbitrary precipitation p. Steady state.

    From Bruggeman 133.17

    Parameters
    ----------
    x : float or ndarray
        Distance from the boundary [m]
    b : float
        Half width of the aquifer [m]
    k : float
        Hydraulic conductivity [m/d]
    D : float
        Aquifer thickness [m]
    p : float
        Arbitrary constant precipitation [m/d]

    Returns
    -------
    head : float
        Head in the aquifer at distance x [m]
    """

    return p / (2 * k * D) * (b**2 - x**2)
