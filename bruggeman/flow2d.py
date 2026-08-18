import numpy as np
from mpmath import ellipf, ellipk
from scipy.special import i0, i1, k0, k1, exp1

from bruggeman.general import W, latexify_function


@latexify_function(
    identifiers={
        "bruggeman_244_02": "varphi",
        "i0": "I_0",
        "i1": "I_1",
        "k0": "K_0",
        "k1": "K_1",
        "head": "varphi",
        "denominator": "Delta"
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
    """Circular basin with different permeable layers for basin and polder.

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

    denominator = (
        lambda_1 * i0(R / lambda_1) * k1(R / lambda_2)
        + lambda_2 * i1(R / lambda_1) * k0(R / lambda_2)
    )
    phi_1 = h - h * lambda_1 * k1(R / lambda_2) * i0(r_arr / lambda_1) / denominator
    phi_2 = h * lambda_2 * i1(R / lambda_1) * k0(r_arr / lambda_2) / denominator

    head = np.where(r_arr <= R, phi_1, phi_2)
    if np.ndim(r_arr) == 0:
        return float(head)
    return head


@latexify_function(
    identifiers={"bruggeman_355_19": "omega", "ellipf": "F", "ellipk": "K"},
    reduce_assignments=False,
)
def bruggeman_355_19(
    x: float, z: float, L: float, B: float, h: float, k: float, D: float
) -> float:
    """Drainage canal on a confined aquifer of finite thickness near open boundary.

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
    identifiers={
        "bruggeman_215_03": "varphi",
        "exp1": "E_1",
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
    """Continuous abstraction from a line source with constant discharge Q(t) = Q0.

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
    return Q0 / (4 * np.pi * k * D) * exp1((beta * r) ** 2 / (4 * t))


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
    """Hantush solution for constant discharge to a leaky confined aquifer.

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
        "k0": "K_0",
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
    """Steady-state Hantush solution for a leaky confined aquifer.

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
    return Q0 / (2 * np.pi * k * D) * k0(r / lambda_)


@latexify_function(
    identifiers={
        "bruggeman_241_25": "varphi",
        "lambda_": "lambda",
        "k0": "K_0",
        "i0": "I_0",
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
    """Steady-state solution for a pumping well at the centre of a 
    circular island with open vertical boundary at r = R. Constant 
    discharge.

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
    return Q0 / (2 * np.pi * k * D) * (
        k0(r / lambda_) - k0(R / lambda_) / i0(R / lambda_) * i0(r / lambda_)
    )


@latexify_function(
    identifiers={
        "bruggeman_241_27": "varphi",
        "lambda_": "lambda",
        "k0": "K_0",
        "k1": "K_1",
        "i0": "I_0",
        "i1": "I_1",
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
    """Steady-state solution for a pumping well at the centre of a 
    circular island with closed vertical boundary at r = R. Constant 
    discharge.

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
    return Q0 / (2 * np.pi * k * D) * (
        k0(r / lambda_) + k1(R / lambda_) / i1(R / lambda_) * i0(r / lambda_)
    )


@latexify_function(
    identifiers={"bruggeman_355_19_total_discharge": "q", "ellipf": "F", "ellipk": "K"},
    reduce_assignments=True,
)
def bruggeman_355_19_total_discharge(
    L: float, B: float, h: float, k: float, D: float
) -> float:
    """Total discharge to the canal in Bruggeman 355-19.

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
