import numpy as np
from mpmath import ellipf, ellipk
from scipy.special import i0, i1, k0, k1

from bruggeman.general import latexify_function


@latexify_function(
    identifiers={
        "bruggeman_244_02": "varphi",
        "i0": "I_0",
        "i1": "I_1",
        "k0": "K_0",
        "k1": "K_1",
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
        head difference between basin and surrounding polder [L]
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

    denominator = lambda_1 * i0(R / lambda_1) * k1(R / lambda_2) + lambda_2 * i1(
        R / lambda_1
    ) * k0(R / lambda_2)
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
