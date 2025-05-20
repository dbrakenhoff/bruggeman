from numpy import arctan, cos, exp, float64, imag, pi, real, sin, sqrt
from numpy.typing import NDArray
from scipy.special import erfc

from bruggeman.general import ierfc, latexify_function


@latexify_function(identifiers={"bruggeman_123_02": "varphi"}, reduce_assignments=True)
def bruggeman_123_02(
    x: float | NDArray[float64],
    t: float | NDArray[float64],
    h: float,
    k: float,
    D: float,
    S: float,
) -> float | NDArray[float64]:
    """Solution for sudden rise of the water table in a confined aquifer.

    From Bruggeman 123.02

    Parameters
    ----------
    x : float
        Distance from the boundary [m]
    t : float
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
    beta = sqrt(S / (k * D))
    u = beta * x / (2 * sqrt(t))
    return h * erfc(u)


@latexify_function(identifiers={"bruggeman_123_03": "varphi"}, reduce_assignments=True)
def bruggeman_123_03(
    x: float | NDArray[float64],
    t: float | NDArray[float64],
    a: float,
    k: float,
    D: float,
    S: float,
) -> float | NDArray[float64]:
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
    beta = sqrt(S / (k * D))
    u = beta * x / (2 * sqrt(t))
    return a * t * ierfc(u, 2) / ierfc(0, 2)


@latexify_function(
    identifiers={"bruggeman_123_05_q": "varphi"}, reduce_assignments=False
)
def bruggeman_123_05_q(
    x: float | NDArray[float64],
    t: float | NDArray[float64],
    Q: float,
    k: float,
    D: float,
    S: float,
) -> float | NDArray[float64]:
    """Solution for constant infiltration/pumping in a confined aquifer.

    Probably equivalent to Bruggeman 124.03?

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
    beta = sqrt(S / (k * D))
    u = beta * x / (2 * sqrt(t))
    return 2 * Q * sqrt(t) / sqrt(k * D * S) * ierfc(u, 1) / (ierfc(0, 0))


def bruggeman_123_32():
    """The Polder function.

    From Bruggeman 123.32
    """
    # implement function (check Pastas)
    pass


@latexify_function(
    identifiers={
        "bruggeman_126_33": "varphi",
        # "lambda_": "lambda",  # 'r\lambda' causes problems in Jupyter notebooks
    },
    reduce_assignments=False,
)
def bruggeman_126_33(
    x: float | NDArray[float64],
    h: float,
    k: float,
    D: float,
    c: float,
    w: float,
) -> float | NDArray[float64]:
    """Leaky aquifer with entrance resistance. Steady state after head change.

    From Bruggeman 126.33

    Parameters
    ----------
    x : float
        Distance from the boundary [m]
    h : float
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
    lambda_ = sqrt(k * D * c)
    return h * lambda_ / (k * w + lambda_) * exp(-x / lambda_)


@latexify_function(
    identifiers={"bruggeman_128_01": "varphi"},
    reduce_assignments=False,
    escape_underscores=False,
)
def bruggeman_128_01(
    x: float | NDArray[float64],
    t: float | NDArray[float64],
    h: float,
    S: float,
    k: float,
    D: float,
    tau: float,
) -> float | NDArray[float64]:
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
    beta = sqrt(S / (k * D))
    omega = 2 * pi / tau
    omega_p = beta * sqrt(omega / 2)
    return h * exp(-omega_p * x) * sin(omega * t - omega_p * x)


@latexify_function(
    identifiers={"bruggeman_128_03": "varphi", "j": "i", "real": "Re", "imag": "Im"},
    reduce_assignments=False,
)
def bruggeman_128_03(
    x: float | NDArray[float64],
    t: float | NDArray[float64],
    h: float,
    S: float,
    k: float,
    D: float,
    tau: float,
    c: float,
) -> float | NDArray[float64]:
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
    beta = sqrt(S / (k * D))
    eta = 1 / (c * S)
    omega = 2 * pi / tau

    i = 1j
    a = real(sqrt(eta + i * omega))
    b = imag(sqrt(eta + i * omega))

    return h * exp(-beta * a * x) * sin(omega * t - beta * b * x)


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
    x: float | NDArray[float64],
    t: float | NDArray[float64],
    h: float,
    S: float,
    k: float,
    D: float,
    tau: float,
    c: float,
    w: float,
) -> float | NDArray[float64]:
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
    beta = sqrt(S / (k * D))
    eta = 1 / (c * S)
    omega = 2 * pi / tau
    theta = 1 / (beta**2 * k**2 * w**2)

    i = 1j
    a = real(sqrt(eta + i * omega))
    b = imag(sqrt(eta + i * omega))

    return (
        h
        * sqrt(theta)
        * exp(-beta * a * x)
        * sin(omega * t - beta * b * x - arctan(b / (a + sqrt(theta))))
        / (sqrt((a + sqrt(theta)) ** 2 + b**2))
    )


@latexify_function(
    identifiers={"bruggeman_133_16": "varphi"},
    reduce_assignments=False,
)
def bruggeman_133_16(
    x: float | NDArray[float64],
    t: float | NDArray[float64],
    L: float,
    S: float,
    k: float,
    D: float,
    p: float = 1.0,
    N: int = 10,
) -> float | NDArray[float64]:
    """Confined aquifer with zero head at x=b(=L/2), zero flux at x=0
    and a constant arbitrary precipitation p.

    From Bruggeman 133.16

    Parameters
    ----------
    x : float
        Distance from the boundary [m]
    t : float
        Time [d]
    L : float
        Length of the aquifer [m]
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
    beta = sqrt(S / (k * D))
    b = L / 2
    return p / (2 * k * D) * (b**2 - x**2) - 16 * p * b**2 / (pi**3 * k * D) * sum(
        (-1) ** n
        / (2 * n + 1) ** 3
        * cos((2 * n + 1) * pi * x / (2 * b))
        * exp(-(((2 * n + 1) * pi / (2 * beta * b)) ** 2) * t)
        for n in range(N)
    )
