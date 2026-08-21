import numpy as np
from numpy import pi, sqrt, float64
from numpy.typing import NDArray
from scipy.special import k0

from bruggeman.general import latexify_function


@latexify_function(
    identifiers={
        "huisman_kemp": "varphi_1,2",
        "alpha_1": "\\alpha_1",
        "alpha_2": "\\alpha_2",
        "beta_1": "\\beta_1",
        "lambda_1": "\\lambda_1",
        "lambda_2": "\\lambda_2",
        "k0": "K_0",
    },
    reduce_assignments=False,
    escape_underscores=False,
)
def huisman_kemp(
    r: float | NDArray[float64],
    Q_1: float,
    Q_2: float,
    k_1: float,
    D_1: float,
    k_2: float,
    D_2: float,
    c_1: float,
    c_2: float,
) -> tuple[float | NDArray[float64], float | NDArray[float64]]:
    """Head drawdown in the two artesian intervals of a two-layer system.

    Parameters
    ----------
    r : float or NDArray[float64]
        radial distance from the well [L]
    Q_1 : float
        discharge in the upper artesian aquifer [L^3/T]
    Q_2 : float
        discharge in the lower artesian aquifer [L^3/T]
    k_1 : float
        hydraulic conductivity of the upper artesian aquifer [L/T]
    D_1 : float
        thickness of the upper artesian aquifer [L]
    k_2 : float
        hydraulic conductivity of the lower artesian aquifer [L/T]
    D_2 : float
        thickness of the lower artesian aquifer [L]
    c_1 : float
        hydraulic resistance above the upper artesian aquifer [T]
    c_2 : float
        hydraulic resistance between the two artesian aquifers [T]

    Returns
    -------
    tuple:
        varphi_1, varphi_2 : float or NDArray[float64]
        head drawdown in the upper and lower artesian intervals [L]
    """
    alpha_1 = 1 / (k_1 * D_1 * c_1)
    alpha_2 = 1 / (k_2 * D_2 * c_2)
    beta_1 = 1 / (k_1 * D_1 * c_2)

    lambda_1 = 0.5 * (
        (alpha_1 + alpha_2 + beta_1)
        + sqrt((alpha_1 + alpha_2 + beta_1) ** 2 - 4 * alpha_1 * alpha_2)
    )
    lambda_2 = 0.5 * (
        (alpha_1 + alpha_2 + beta_1)
        - sqrt((alpha_1 + alpha_2 + beta_1) ** 2 - 4 * alpha_1 * alpha_2)
    )

    varphi_1 = (
        Q_1 / (2 * pi * k_1 * D_1 * (lambda_1 - lambda_2))
        * (
            (lambda_1 - alpha_2) * k0(sqrt(lambda_1) * r)
            + (alpha_2 - lambda_2) * k0(sqrt(lambda_2) * r)
        )
        + Q_2 / (2 * pi * k_2 * D_2)
        * beta_1
        / (lambda_1 - lambda_2)
        * (-k0(sqrt(lambda_1) * r) + k0(sqrt(lambda_2) * r))
    )
    varphi_2 = (
        Q_1 / (2 * pi * k_1 * D_1)
        * alpha_2
        / (lambda_1 - lambda_2)
        * (-k0(sqrt(lambda_1) * r) + k0(sqrt(lambda_2) * r))
        + Q_2 / (2 * pi * k_2 * D_2)
        / (lambda_1 - lambda_2)
        * (
            (alpha_2 - lambda_2) * k0(sqrt(lambda_1) * r)
            + (lambda_1 - alpha_2) * k0(sqrt(lambda_2) * r)
        )
    )
    return varphi_1, varphi_2
