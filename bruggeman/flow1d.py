import numpy as np
from scipy.special import erfc

from bruggeman.general import ierfc


def bruggeman_123_02(x, t, dh, k, D, S):
    """Solution for sudden rise of the water table in a confined aquifer.

    From Bruggeman 123.02

    Parameters
    ----------
    x : float
        Distance from the boundary [m]
    t : float
        Time since the start of the rise [d]
    dh : float
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
    return dh * erfc(u)


def bruggeman_123_03(x, t, a, k, D, S):
    """Solution for linear rise of the water table in a confined aquifer.

    From Bruggeman 123.03

    Parameters
    ----------
    x : float
        Distance from the boundary [m]
    t : float
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


def bruggeman_123_05_q(x, t, Q, k, D, S):
    """Solution for constant infiltration/pumping in a confined aquifer.

    From Olsthoorn, Th. 2006. Van Edelman naar Bruggeman. Stromingen 12 (2006) p5-11.

    Parameters
    ----------
    x : float
        Distance from the boundary [m]
    t : float
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
    s = 2 * Q * np.sqrt(t) / np.sqrt(k * D * S) * ierfc(u, 1) / (ierfc(0, 0))
    return s


def bruggeman_123_32():
    """The Polder function.

    From Bruggeman 123.32
    """
    # implement function (check Pastas)
    pass


def bruggeman_133_15():
    """The response function of :cite:t:`van_de_leur_study_1958`.

    From Bruggeman 133.15
    """
    # implement function (check Pastas)
    pass


def bruggeman_128_01(x, t, h, S, k, D, tau):
    """Tidal fluctuation open water, confined aquifer with open boundary (x = 0).

    From Bruggeman 128.01

    Parameters
    ----------
    x : float
        Distance from the boundary [m]
    t : float
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
    omega_accent = beta * np.sqrt(omega / 2)

    return h * np.exp(-omega_accent * x) * np.sin(omega * t - omega_accent * x)


def bruggeman_128_03(x, t, h, S, k, D, tau, c):
    """Tidal fluctuation open water, leaky aquifer with open boundary (x = 0).

    From Bruggeman 128.03

    Parameters
    ----------
    x : float
        Distance from the boundary [m]
    t : float
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

    a = np.real(np.sqrt(eta + 1j * omega))
    b = np.imag(np.sqrt(eta + 1j * omega))

    return h * np.exp(-beta * a * x) * np.sin(omega * t - beta * b * x)


def bruggeman_128_04(x, t, h, S, k, D, tau, c, w):
    """Tidal fluctuation open water, leaky aquifer with entrance resistance (x = 0).

    From Bruggeman 128.04

    Parameters
    ----------
    x : float
        Distance from the boundary [m]
    t : float
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
    theta = 1 / (np.power(beta, 2) * np.power(k, 2) * np.power(w, 2))

    a = np.real(np.sqrt(eta + 1j * omega))
    b = np.imag(np.sqrt(eta + 1j * omega))

    return (
        h
        * np.sqrt(theta)
        * np.exp(-beta * a * x)
        * np.sin(omega * t - beta * b * x - np.arctan(b / (a + np.sqrt(theta))))
        / (np.sqrt(np.power((a + np.sqrt(theta)), 2) + np.power(b, 2)))
    )
