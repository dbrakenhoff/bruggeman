from numpy import exp, float64, pi, sqrt
from numpy.typing import NDArray
from scipy.special import erfc

from bruggeman.general import latexify_function


@latexify_function(
    identifiers={"h_edelman": "varphi"},
    reduce_assignments=True,
    escape_underscores=False,
)
def h_edelman(
    x: float | NDArray[float64],
    t: float | NDArray[float64],
    T: float,
    S: float,
    h: float,
    t_0: float = 0.0,
) -> float | NDArray[float64]:
    # from Analyical Groundwater Modeling, ch. 5
    u = sqrt(S * x**2 / (4 * T * (t - t_0)))
    return h * erfc(u)


@latexify_function(
    identifiers={"Qx_edelman": "Q_x"},
    reduce_assignments=True,
    escape_underscores=False,
)
def Qx_edelman(
    x: float | NDArray[float64],
    t: float | NDArray[float64],
    T: float,
    S: float,
    h: float,
    t_0: float = 0.0,
) -> float | NDArray[float64]:
    # from Analyical Groundwater Modeling, ch. 5
    u = sqrt(S * x**2 / (4 * T * (t - t_0)))
    return T * h * 2 * u / (x * sqrt(pi)) * exp(-(u**2))
