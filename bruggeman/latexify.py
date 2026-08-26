"""Utilities for generating LaTeX representations of functions using sympy."""

import ast
import inspect
import textwrap
from collections.abc import Callable

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


_NP_PASSTHROUGH = frozenset({"asarray", "ascontiguousarray", "array", "atleast_1d"})


class _NpProxy:
    """Proxy for ``numpy`` that redirects attribute access to sympy equivalents.

    Allows symbolic evaluation of expressions written as ``np.tanh(...)``,
    ``np.pi``, etc. without importing numpy into the eval namespace.
    """

    def __getattr__(self, name: str):
        import sympy as sp

        if name in _NP_PASSTHROUGH:
            return lambda x, *args, **kwargs: x
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


def _is_matrix(obj) -> bool:
    return bool(getattr(obj, "is_Matrix", False))


class _MatrixFunc:
    """Lazily builds a ``MatrixExpr`` rendering ``func(arg)``, e.g. ``e^{-xA}``.

    Used by ``mexp``/``mcosh``/``msinh`` below, since plain ``sympy.exp`` of a
    matrix expression doesn't compose with matrix multiplication.
    """

    _TEMPLATES = {
        "exp": r"e^{{{}}}",
        "cosh": r"\cosh\left({}\right)",
        "sinh": r"\sinh\left({}\right)",
    }

    def __new__(cls, func_name: str, arg):
        import sympy as sp

        class _Impl(sp.MatrixExpr):
            def __new__(icls, name, matrix_arg):
                name_sym = name if isinstance(name, sp.Symbol) else sp.Symbol(str(name))
                return sp.Basic.__new__(icls, name_sym, sp.sympify(matrix_arg))

            @property
            def shape(self):
                return self.args[1].shape

            def _latex(self, printer):
                name = str(self.args[0])
                template = cls._TEMPLATES.get(name, name + r"\left({}\right)")
                return template.format(printer._print(self.args[1]))

        return _Impl(func_name, arg)


def mexp(matrix):
    """``e^{M}``, rendering nicely whether ``matrix`` is scalar or a matrix expr."""
    import sympy as sp

    return _MatrixFunc("exp", matrix) if _is_matrix(matrix) else sp.exp(matrix)


def mcosh(matrix):
    """``cosh(M)``, rendering nicely whether ``matrix`` is scalar or a matrix expr."""
    import sympy as sp

    return _MatrixFunc("cosh", matrix) if _is_matrix(matrix) else sp.cosh(matrix)


def msinh(matrix):
    """``sinh(M)``, rendering nicely whether ``matrix`` is scalar or a matrix expr."""
    import sympy as sp

    return _MatrixFunc("sinh", matrix) if _is_matrix(matrix) else sp.sinh(matrix)


def msqrt(matrix):
    """``sqrt(M)``, rendering nicely whether ``matrix`` is scalar or a matrix expr."""
    import sympy as sp

    return sp.sqrt(matrix)


def minv(matrix):
    """``M^{-1}``, rendering nicely whether ``matrix`` is scalar or a matrix expr."""
    return matrix.I if _is_matrix(matrix) else 1 / matrix


def latexify_matrix_equation(
    equations: dict[str, Callable],
    scalars: tuple[str, ...] = (),
    vectors: tuple[str, ...] = (),
    matrices: tuple[str, ...] = (),
):
    """Attach a hand-written matrix/vector LaTeX equation as ``_repr_latex_``.

    Unlike ``latexify_function``, this does not parse the decorated function's
    source. Instead each callable in ``equations`` is called with sympy
    symbols named after ``scalars``, column-vector ``MatrixSymbol``s named
    after ``vectors``, and square ``MatrixSymbol``s named after ``matrices``
    (all sharing a symbolic dimension), and must return the right-hand side.
    Useful for solutions whose numeric implementation (e.g. matrix
    exponentiation via eigendecomposition) can't be recovered symbolically
    from the source, but has a simple closed form.

    Parameters
    ----------
    equations : dict
        Maps the LaTeX name of the left-hand side (e.g. ``r"\\varphi"``) to a
        callable building the right-hand side, e.g.
        ``lambda x, A, h: mexp(-x * msqrt(A)) @ h``. Only the parameter names
        the callable declares are passed in.
    scalars, vectors, matrices : tuple of str, optional
        Names of the symbols available to the ``equations`` callables.
    """

    def decorator(f):
        try:
            import sympy as sp

            n = sp.Symbol("n", integer=True, positive=True)
            available = {s: sp.Symbol(s) for s in scalars}
            available.update({v: sp.MatrixSymbol(v, n, 1) for v in vectors})
            available.update({m: sp.MatrixSymbol(m, n, n) for m in matrices})

            lines = [r"\begin{aligned}"]
            for lhs, builder in equations.items():
                params = inspect.signature(builder).parameters
                kwargs = {k: v for k, v in available.items() if k in params}
                rhs = builder(**kwargs)
                lhs_sym = (
                    sp.MatrixSymbol(lhs, *rhs.shape)
                    if _is_matrix(rhs)
                    else sp.Symbol(lhs)
                )
                latex = sp.latex(sp.Eq(lhs_sym, rhs)).replace(" = ", r" &= ", 1)
                lines.append("  " + latex + r" \\")
            lines.append(r"\end{aligned}")
            latex = "\n".join(lines)
        except Exception:
            return f
        f._repr_latex_ = lambda: f"$$\n{latex}\n$$"
        return f

    return decorator


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
