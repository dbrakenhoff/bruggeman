"""Commonly used functions in Bruggeman's solutions."""

import numpy as np
from numpy import clip, exp, float64, pi, sqrt
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.special import erfc


def ierfc(z: float, n: int) -> float:
    """Iterated integral complementary error function."""
    if n == -1:
        return 2 / sqrt(pi) * exp(-z * z)
    elif n == 0:
        return erfc(z)
    else:
        return clip(
            -z / n * ierfc(z, n - 1) + 1 / (2 * n) * ierfc(z, n - 2),
            a_min=0.0,
            a_max=None,
        )


def P(
    x: float | NDArray[float64],
    y: float | NDArray[float64],
) -> float | NDArray[float64]:
    """Bruggeman's Polder function for 1D flow in a semi-infinite aquifer."""
    return 1 / 2 * exp(2 * x) * erfc(x / y + y) + 1 / 2 * exp(-2 * x) * erfc(x / y - y)


def W(
    tau: float | NDArray[float64],
    rho: float | NDArray[float64],
) -> float | NDArray[float64]:
    r"""Hantush well function for leaky-aquifer flow.

    ..math::

        W(\tau, \rho) = \int_0^\tau \frac{1}{x} \exp\left(-x - \frac{\rho^2}{4x}\right) \, dx.
    """
    tau_arr = np.asarray(tau)
    rho_arr = np.asarray(rho)
    scalar = tau_arr.ndim == 0 and rho_arr.ndim == 0
    tau_b, rho_b = np.broadcast_arrays(tau_arr, rho_arr)

    def _w_single(tau_val: float, rho_val: float) -> float:
        if tau_val == 0:
            return 0.0
        result, _ = quad(
            lambda x: np.exp(-x - rho_val**2 / (4 * x)) / x,
            0.0,
            float(tau_val),
            limit=100,
            epsabs=1e-10,
            epsrel=1e-10,
        )
        return result

    vec = np.vectorize(_w_single, otypes=[float])
    out = vec(tau_b, rho_b)
    return float(out) if scalar else out
