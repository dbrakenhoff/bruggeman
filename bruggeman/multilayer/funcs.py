from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import inv
from scipy.special import i0, i1, k0, k1


def _build_A_matrix(
    k: NDArray[np.float64],
    D: NDArray[np.float64],
    c: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Build the leakage matrix A for a multilayer system.

    From Bruggeman (1999), the matrix A is defined as:

    .. code-block:: text

        A = -diag(1./(kD(2:NLay).*c(2:NLay)), -1) +
            diag(1./(kD(1:NLay).*c(1:NLay)) + 1./(kD(1:NLay).*[c(2:NLay) Inf]), 0) -
            diag(1./(kD(1:NLay-1).*c(2:NLay)), 1)

    where kD is the transmissivity.

    Parameters
    ----------
    k : ndarray
        Hydraulic conductivity for each aquifer [m/d]
    D : ndarray
        Thickness for each aquifer [m]
    c : ndarray
        Resistance of confining beds [d]. Length should be len(k) + 1.
        c[0] is the resistance above the first aquifer,
        c[1:len(k)] is the resistance between aquifers,
        c[len(k)] is the resistance below the last aquifer.
        Use np.inf for impermeable boundaries.

    Returns
    -------
    A : ndarray
        Leakage matrix of shape (n_layers, n_layers) [1/d]
    """
    n_layers = len(k)
    kD = k * D  # Transmissivity [m^2/d]

    A = np.zeros((n_layers, n_layers))

    # Diagonal elements: 1/(kD[i]*c[i]) + 1/(kD[i]*c[i+1])
    # where c[n_layers] = inf
    for i in range(n_layers):
        term1 = 1.0 / (kD[i] * c[i]) if c[i] != np.inf else 0.0
        term2 = (
            1.0 / (kD[i] * c[i + 1]) if (i + 1 < len(c) and c[i + 1] != np.inf) else 0.0
        )
        A[i, i] = term1 + term2

    # Sub-diagonal elements: -1/(kD[i]*c[i+1])
    for i in range(n_layers - 1):
        if c[i + 1] != np.inf:
            A[i, i + 1] = -1.0 / (kD[i] * c[i + 1])
            A[i + 1, i] = -1.0 / (kD[i + 1] * c[i + 1])

    return A


def _build_B_matrix(
    S: NDArray[np.float64],
    c: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Build the AA matrix for transient vertical flow solutions.

    From Bruggeman 710.01, AA is used for the case where the dimension
    of A is 1/d but the vector h has elements h/(cS).

    From Bruggeman (1999), the matrix AA is defined as:

    .. code-block:: text

        AA = -diag(1/(S[1:n_layers]*c[1:n_layers]), -1) +
             diag(1/(S[:n_layers]*c[:n_layers]) + 1/(S[:n_layers]*c[1:n_layers+1]), 0) -
             diag(1/(S[:n_layers-1]*c[1:n_layers]), 1)

    Parameters
    ----------
    S : ndarray
        Elastic storage coefficient for each aquifer [-]
    c : ndarray
        Resistance of confining beds [d]. Length should be len(S) + 1.

    Returns
    -------
    AA : ndarray
        Matrix for transient flow of shape (n_layers, n_layers)
    """
    n_layers = len(S)
    B = np.zeros((n_layers, n_layers))

    for i in range(n_layers):
        term1 = 1.0 / (S[i] * c[i]) if c[i] != np.inf else 0.0
        term2 = (
            1.0 / (S[i] * c[i + 1]) if (i + 1 < len(c) and c[i + 1] != np.inf) else 0.0
        )
        B[i, i] = term1 + term2

    for i in range(n_layers - 1):
        if c[i + 1] != np.inf:
            B[i, i + 1] = -1.0 / (S[i] * c[i + 1])
            B[i + 1, i] = -1.0 / (S[i + 1] * c[i + 1])

    return B


def _matrix_funm(
    x: float | NDArray[np.float64],
    sqrtA: NDArray[np.float64],
    func: Callable,
) -> NDArray[np.float64]:
    """Apply a function to the eigenvalues of x * sqrtA.

    Computes V @ diag(func(x * Lambda)) @ V_inv where sqrtA = V @ Lambda @ V_inv.

    Parameters
    ----------
    x : float or ndarray
        Scalar or array of x values
    sqrtA : ndarray
        Matrix square root of A
    func : callable
        Function to apply (e.g., np.sinh, np.cosh, np.exp)

    Returns
    -------
    result : ndarray
        Matrix function result. Shape: (n_layers, n_layers, n_x) or (n_layers, n_layers)
    """
    x_arr = np.atleast_1d(x)
    n_x = len(x_arr)
    n_layers = sqrtA.shape[0]

    # Eigenvalue decomposition
    eigval, eigvec = np.linalg.eig(sqrtA)
    eigvec_inv = inv(eigvec)

    values = func(x_arr[:, None] * eigval[None, :])
    result = np.einsum("ik,nk,kj->ijn", eigvec, values, eigvec_inv)

    if n_x == 1:
        return np.real_if_close(result[:, :, 0])
    return np.real_if_close(result)


def _matrix_bessel(
    r: float | NDArray[np.float64],
    sqrtA: NDArray[np.float64],
    bessel_func: Callable,
) -> NDArray[np.float64]:
    """Compute matrix Bessel function using eigenvalue decomposition.

    For a matrix M = r * sqrtA, we compute:
    V @ diag(bessel_func(eigvals(M))) @ V_inv

    Parameters
    ----------
    r : float or ndarray
        Scalar or array of r values
    sqrtA : ndarray
        Matrix square root of A
    bessel_func : callable
        Bessel function to apply (e.g., i0, i1, k0, k1)

    Returns
    -------
    bessel_matrix : ndarray
        Matrix Bessel function for each r value
    """
    r_arr = np.atleast_1d(r)
    n_r = len(r_arr)
    n_layers = sqrtA.shape[0]

    # Eigenvalue decomposition of sqrtA
    eigval, eigvec = np.linalg.eig(sqrtA)
    eigvec_inv = inv(eigvec)

    # Evaluate all radii in one call; the eigendecomposition is shared.
    values = bessel_func(r_arr[:, None] * eigval[None, :])
    result = np.einsum("ik,nk,kj->ijn", eigvec, values, eigvec_inv)

    if n_r == 1:
        return np.real_if_close(result[:, :, 0])
    return np.real_if_close(result)


def _matrix_bessel_i0(
    r: float | NDArray[np.float64],
    sqrtA: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute matrix modified Bessel function of the first kind, order 0."""
    return _matrix_bessel(r, sqrtA, i0)


def _matrix_bessel_i1(
    r: float | NDArray[np.float64],
    sqrtA: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute matrix modified Bessel function of the first kind, order 1."""
    return _matrix_bessel(r, sqrtA, i1)


def _matrix_bessel_k0(
    r: float | NDArray[np.float64],
    sqrtA: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute matrix modified Bessel function of the second kind, order 0."""
    return _matrix_bessel(r, sqrtA, k0)


def _matrix_bessel_k1(
    r: float | NDArray[np.float64],
    sqrtA: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute matrix modified Bessel function of the second kind, order 1."""
    return _matrix_bessel(r, sqrtA, k1)
