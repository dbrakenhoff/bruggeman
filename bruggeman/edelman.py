import numpy as np
from scipy.special import erfc


def h_edelman(x, t, T, S, dh, t0=0.0):
    # from Analyical Groundwater Modeling, ch. 5
    u = np.sqrt(S * x**2 / (4 * T * (t - t0)))
    return dh * erfc(u)


def Qx_edelman(x, t, T, S, dh, t0=0.0):
    # from Analyical Groundwater Modeling, ch. 5
    u = np.sqrt(S * x**2 / (4 * T * (t - t0)))
    return T * dh * 2 * u / (x * np.sqrt(np.pi)) * np.exp(-(u**2))
