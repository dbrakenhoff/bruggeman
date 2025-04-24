from numpy import exp, pi, sqrt
from scipy.special import erfc

from bruggeman.general import latexify_function


@latexify_function(
    identifiers={"h_edelman": "varphi"},
    reduce_assignments=True,
    escape_underscores=False,
)
def h_edelman(x, t, T, S, h, t_0=0.0):
    # from Analyical Groundwater Modeling, ch. 5
    u = sqrt(S * x**2 / (4 * T * (t - t_0)))
    return h * erfc(u)


@latexify_function(
    identifiers={"Qx_edelman": "Q_x"},
    reduce_assignments=True,
    escape_underscores=False,
)
def Qx_edelman(x, t, T, S, h, t_0=0.0):
    # from Analyical Groundwater Modeling, ch. 5
    u = sqrt(S * x**2 / (4 * T * (t - t_0)))
    return T * h * 2 * u / (x * sqrt(pi)) * exp(-(u**2))
