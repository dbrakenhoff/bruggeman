import numpy as np
from scipy.special import erfc


def ierfc(z, n):
    """Iterated integral complementary error function."""
    if n == -1:
        return 2 / np.sqrt(np.pi) * np.exp(-z * z)
    elif n == 0:
        return erfc(z)
    else:
        result = -z / n * ierfc(z, n - 1) + 1 / (2 * n) * ierfc(z, n - 2)
        return np.clip(result, a_min=0.0, a_max=None)
