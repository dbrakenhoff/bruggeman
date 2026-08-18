import ast
import inspect
import textwrap
from collections.abc import Callable

import numpy as np
from numpy import clip, exp, pi, sqrt, float64
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.special import erfc

_NP_TO_SYMPY = {
    "sqrt": "sqrt",
    "exp": "exp",
    "erfc": "erfc",
    "pi": "pi",
    "cos": "cos",
    "sin": "sin",
    "arctan": "atan",
    "arcsin": "asin",
    "tanh": "tanh",
    "real": "re",
    "imag": "im",
    "log": "log",
}


class _NpProxy:
    """Proxy for ``numpy`` that redirects attribute access to sympy equivalents.

    Allows symbolic evaluation of expressions written as ``np.tanh(...)``,
    ``np.pi``, etc. without importing numpy into the eval namespace.
    """

    def __getattr__(self, name: str):
        import sympy as sp

        sympy_name = _NP_TO_SYMPY.get(name, name)
        attr = getattr(sp, sympy_name, None)
        if attr is not None:
            return attr
        return sp.Function(name)


def _sympy_ns() -> dict:
    """Build a sympy evaluation namespace mapping numpy/scipy names to sympy."""
    import sympy as sp

    ns: dict = {k: getattr(sp, v) for k, v in _NP_TO_SYMPY.items()}
    ns["ierfc"] = sp.Function("ierfc")
    ns["clip"] = lambda x, a_min=None, a_max=None: x
    # Use unevaluated, capitalized Re/Im operators for clean LaTeX output
    ns["real"] = sp.Function("Re")
    ns["imag"] = sp.Function("Im")
    ns["np"] = _NpProxy()
    ns["_sp"] = sp
    return ns


class _SumRangeTransformer(ast.NodeTransformer):
    """Replace ``sum(f(n) for n in range(N))`` with ``_sp.Sum(f(n), (n, 0, N-1))``.

    This allows symbolic evaluation when N is a sympy Symbol rather than an int.
    The loop variable is recorded in ``self.loop_vars`` so callers can add it as
    a sympy Symbol to the evaluation namespace before eval().
    """

    def __init__(self) -> None:
        self.loop_vars: list[str] = []

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if not (
            isinstance(node.func, ast.Name)
            and node.func.id == "sum"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.GeneratorExp)
        ):
            return node
        gen = node.args[0]
        comps = gen.generators
        if not (
            len(comps) == 1
            and not comps[0].ifs
            and isinstance(comps[0].iter, ast.Call)
            and isinstance(comps[0].iter.func, ast.Name)
            and comps[0].iter.func.id == "range"
            and len(comps[0].iter.args) == 1
            and isinstance(comps[0].target, ast.Name)
        ):
            return node

        var_name = comps[0].target.id
        self.loop_vars.append(var_name)
        limit_node = comps[0].iter.args[0]

        upper = ast.BinOp(left=limit_node, op=ast.Sub(), right=ast.Constant(value=1))
        sym_var = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="_sp", ctx=ast.Load()),
                attr="Symbol",
                ctx=ast.Load(),
            ),
            args=[ast.Constant(value=var_name)],
            keywords=[],
        )
        bounds = ast.Tuple(elts=[sym_var, ast.Constant(value=0), upper], ctx=ast.Load())
        new_call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="_sp", ctx=ast.Load()),
                attr="Sum",
                ctx=ast.Load(),
            ),
            args=[gen.elt, bounds],
            keywords=[],
        )
        ast.fix_missing_locations(new_call)
        return new_call


def to_latex(
    func: Callable,
    identifiers: dict | None = None,
    reduce_assignments: bool = True,
) -> str | None:
    """Generate a LaTeX string for a function using sympy.

    Parameters
    ----------
    func : callable
        Function to convert.
    identifiers : dict, optional
        Map variable or function names to LaTeX symbol names,
        e.g. ``{"my_func": "varphi", "theta": "vartheta"}``.
    reduce_assignments : bool, optional
        If True (default), substitute all intermediate variables and return a
        single equation. If False, return an ``align*`` block showing each
        intermediate assignment followed by the final equation.
    """
    try:
        import sympy as sp
    except ImportError:
        return None

    sig = inspect.signature(func)

    # Map identifier keys that name callables in the function's globals
    # (e.g. ellipk, ellipf) to unevaluated sympy Functions named by their value.
    def _add_callable_stubs(ns: dict) -> None:
        for key, latex_name in (identifiers or {}).items():
            if key == func.__name__:
                continue
            if key in func.__globals__ and callable(func.__globals__[key]):
                ns[key] = sp.Function(latex_name)

    if not reduce_assignments:
        ns = _sympy_ns()
        _add_callable_stubs(ns)
        ns.update({name: sp.Symbol(name) for name in sig.parameters})
        try:
            source = textwrap.dedent(inspect.getsource(func))
            tree = ast.parse(source)
        except Exception:
            return None
        func_def = next(
            (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)), None
        )
        if func_def is None:
            return None

        equations = []
        transformer = _SumRangeTransformer()
        for stmt in func_def.body:
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
            ):
                var_name = stmt.targets[0].id
                try:
                    transformed = transformer.visit(stmt.value)
                    ast.fix_missing_locations(transformed)
                    for lv in transformer.loop_vars:
                        ns[lv] = sp.Symbol(lv)
                    transformer.loop_vars.clear()
                    val = eval(ast.unparse(transformed), ns)  # noqa: S307
                    if isinstance(val, complex):
                        # e.g. `i = 1j` — map to sympy I, don't display
                        ns[var_name] = sp.nsimplify(val)
                        continue
                    val = sp.nsimplify(val, rational=False)  # cleans up 1.0*I → I etc.
                    lhs_name = (identifiers or {}).get(var_name, var_name)
                    equations.append(sp.Eq(sp.Symbol(lhs_name), val))
                    ns[var_name] = sp.Symbol(lhs_name)
                except Exception:
                    pass
            elif isinstance(stmt, ast.Return) and stmt.value is not None:
                try:
                    transformed = transformer.visit(stmt.value)
                    ast.fix_missing_locations(transformed)
                    for lv in transformer.loop_vars:
                        ns[lv] = sp.Symbol(lv)
                    transformer.loop_vars.clear()
                    val = eval(ast.unparse(transformed), ns)  # noqa: S307
                    func_name = (identifiers or {}).get(func.__name__, func.__name__)
                    params = [sp.Symbol(p) for p in sig.parameters]
                    equations.append(sp.Eq(sp.Function(func_name)(*params), val))
                except Exception:
                    pass

        if not equations:
            return None
        # Use `aligned` (sub-environment) not `align*` (standalone display env).
        # `align*` nested inside any outer math delimiter causes MathJax to
        # raise "Erroneous nesting of equation structures".
        lines = [r"\begin{aligned}"]
        for eq in equations:
            lines.append("  " + sp.latex(eq).replace(" = ", r" &= ", 1) + r" \\")
        lines.append(r"\end{aligned}")
        return "\n".join(lines)

    # reduce_assignments=True: run the function with symbol arguments
    ns = _sympy_ns()
    _add_callable_stubs(ns)
    symbols = {name: sp.Symbol(name) for name in sig.parameters}
    patched = {k: v for k, v in ns.items() if k in func.__globals__}
    saved = {k: func.__globals__[k] for k in patched}
    func.__globals__.update(patched)
    try:
        expr = func(**symbols)
        lhs = sp.Symbol((identifiers or {}).get(func.__name__, func.__name__))
        return sp.latex(sp.Eq(lhs, expr))
    except Exception:
        return None
    finally:
        func.__globals__.update(saved)


def latexify_function(
    function: Callable | None = None,
    identifiers: dict | None = None,
    reduce_assignments: bool = True,
    **kwargs,
):
    """Decorator to render a function as LaTeX in Jupyter notebooks.

    Uses sympy to generate LaTeX and attaches a ``_repr_latex_`` method for
    Jupyter display. Silently falls back to the original function if conversion
    fails or sympy is not installed.

    Parameters
    ----------
    function : callable, optional
        function to decorate, by default None
    identifiers : dict, optional
        remap variable or function names to latex symbols,
        e.g. ``{"my_func": "varphi", "theta": "vartheta"}``, by default None
    reduce_assignments : bool, optional
        If True (default), show a single combined equation. If False, show each
        intermediate variable assignment followed by the final equation in an
        ``align*`` block.
    """

    def decorator(f):
        latex = to_latex(
            f, identifiers=identifiers, reduce_assignments=reduce_assignments
        )
        if latex:
            if reduce_assignments:
                f._repr_latex_ = lambda: f"$\\displaystyle {latex}$"
            else:
                # `aligned` is a sub-environment that must live inside $$...$$.
                f._repr_latex_ = lambda: f"$$\n{latex}\n$$"
        return f

    if function:
        return decorator(function)
    return decorator


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
    return 1/2 * exp(2 * x) * erfc(x / y + y) + 1/2 * exp(-2 * x) * erfc(x / y - y)


def W(
    tau: float | NDArray[float64],
    rho: float | NDArray[float64],
) -> float | NDArray[float64]:
    r"""Hantush well function for leaky-aquifer flow.

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
            lambda x: np.exp(-x - rho_val ** 2 / (4 * x)) / x,
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