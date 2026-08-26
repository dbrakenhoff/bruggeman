"""CI: Multi-layer systems - one dimensional flow (solutions 710).

This module contains Bruggeman's analytical solutions for one-dimensional
flow in continuous multi-layer systems.

From Bruggeman (1999), Section C, solutions 710.

Note: These solutions work with multi-layer systems where parameters are
provided as arrays for each layer. The functions return heads and discharges
for all layers.
"""

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from bruggeman.latexify import latexify_function, latexify_matrix_equation, mexp, msqrt
from bruggeman.multilayer.funcs import _build_A_matrix, _build_B_matrix


@latexify_matrix_equation(
    {
        r"\varphi": lambda x, A, h: mexp(-x * msqrt(A)) @ h,
        "Q_x": lambda x, A, T, h: T @ mexp(-x * msqrt(A)) @ msqrt(A) @ h,
    },
    scalars=("x",),
    vectors=("h",),
    matrices=("A", "T"),
)
def bruggeman_710_12(
    x: float | NDArray[np.float64],
    h: NDArray[np.float64],
    k: NDArray[np.float64],
    D: NDArray[np.float64],
    c: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """710.12 Multiple aquifers with open boundary, sudden drawdown, steady-state.

    Sudden drawdown of the surface water level, which is kept constant
    thereafter. Head is a function of x only: phi = phi(x) = drawdown.

    From Bruggeman (1999) solution 710.12.

    Parameters
    ----------
    x : float or ndarray
        Distance from the boundary [m]
    h : ndarray
        Drawdown height for each aquifer [m]
    k : ndarray
        Hydraulic conductivity for each aquifer [m/d]
    D : ndarray
        Thickness for each aquifer [m]
    c : ndarray
        Resistance of confining beds [d]. Length should be len(k) + 1.
        Use np.inf for impermeable boundaries.

    Returns
    -------
    phi : ndarray
        Head in each aquifer at distance x [m]. Shape: (n_layers, n_x)
    Qx : ndarray
        Discharge in each aquifer at distance x [m^2/d]. Shape: (n_layers, n_x)
    """
    n_layers = len(k)
    x_arr = np.atleast_1d(x)
    n_x = len(x_arr)

    # Build A matrix and its square root
    A = _build_A_matrix(k, D, c)
    sqrtA = linalg.sqrtm(A)

    # T is diagonal transmissivity matrix
    T = np.diag(k * D)

    # Precompute eigenvalue decomposition for efficiency
    Lambda, V = np.linalg.eig(sqrtA)
    V_inv = linalg.inv(V)

    phi = np.zeros((n_layers, n_x))
    Qx = np.zeros((n_layers, n_x))

    for ix, xi in enumerate(x_arr):
        exp_diag = np.diag(np.exp(-xi * Lambda))
        exp_term = V @ exp_diag @ V_inv
        phi[:, ix] = exp_term @ h
        Qx[:, ix] = T @ exp_term @ sqrtA @ h

    if n_x == 1:
        return phi[:, 0], Qx[:, 0]
    return phi, Qx
