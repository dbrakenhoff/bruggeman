import numpy as np
from scipy.special import erfc

from bruggeman.general import ierfc


def bruggeman_123_02(x, t, Δh, k, H, S):
    """Solution for sudden rise of the water table in a confined aquifer.

    From Bruggeman 123.02
    """
    beta = np.sqrt(S / (k * H))
    u = beta * x / (2 * np.sqrt(t))
    return Δh * erfc(u)


def bruggeman_123_03(x, t, a, k, H, S):
    """Solution for linear rise of the water table in a confined aquifer.

    From Bruggeman 123.03
    """
    beta = np.sqrt(S / (k * H))
    u = beta * x / (2 * np.sqrt(t))
    return a * t * ierfc(u, 2) / ierfc(0, 2)


def bruggeman_123_05_q(x, t, b, k, H, S):
    """Solution for constant infiltration/pumping in a confined aquifer.

    From Olsthoorn, Th. 2006. Van Edelman naar Bruggeman. Stromingen 12 (2006) p5-11.
    """
    beta = np.sqrt(S / (k * H))
    u = beta * x / (2 * np.sqrt(t))
    s = 2 * b * np.sqrt(t) / np.sqrt(k * H * S) * ierfc(u, 1) / (ierfc(0, 0))
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
