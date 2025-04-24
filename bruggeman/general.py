from numpy import clip, exp, pi, sqrt
from scipy.special import erfc


def latexify_function(
    function=None,
    identifiers=None,
    use_math_symbols: bool = True,
    use_raw_function_name: bool = False,
    **kwargs,
):
    """
    Decorator to convert a function to LaTeX.

    Source: https://github.com/pastas/pastas/blob/dev/pastas/decorators.py#L168-L191

    Parameters
    ----------
    function : callable, optional
        function to decorate, by default None
    identifiers : dict, optional
        remap symbols or names to latex equivalents e.g. {"this_func": "phi"},
        by default None
    use_math_symbols : bool, optional
        use math symbols, e.g. greek symbols, by default True
    use_raw_function_name : bool, optional
        represent function name as is, do not map as latex, by default False
    """

    def latexify_decorator(f):
        try:
            import latexify

            docstring = str(f.__doc__)
            flatex = latexify.function(
                f,
                identifiers=identifiers,
                use_math_symbols=use_math_symbols,
                use_raw_function_name=use_raw_function_name,
                **kwargs,
            )
            flatex.__doc__ = docstring
            return flatex
        except ImportError:
            return f

    if function:
        fret = latexify_decorator(function)
        fret.__doc__ = function.__doc__
        return fret

    return latexify_decorator


@latexify_function()
def ierfc(z, n):
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
