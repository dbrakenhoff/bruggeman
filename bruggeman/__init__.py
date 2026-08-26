"""Python implementation of Bruggeman (1972) Analytical solutions for groundwater flow.

Table of contents (section letters and solution numbers refer to Bruggeman 1972, names
refer to sub-modules in bruggeman package):

- A. bruggeman.phreatic: Phreatic groundwater      10-100
- B. bruggeman.confined: Confined groundwater     100-700
- C. bruggeman.multilayer: Multi-layer systems    700-800
"""

import inspect
from typing import TYPE_CHECKING

import pandas as pd

# ruff: noqa: F401 F403
from bruggeman import confined, funcs, latexify, multilayer, other, phreatic
from bruggeman.__version__ import __version__
from bruggeman.confined.flow1d import *
from bruggeman.confined.flow2d.general import *
from bruggeman.confined.flow2d.radial import *
from bruggeman.multilayer.flow1d import *
from bruggeman.multilayer.radial import *
from bruggeman.other import *
from bruggeman.phreatic import *

if TYPE_CHECKING:
    from pandas import DataFrame


def _extract_bruggeman_number(func_name: str) -> str:
    """Extract the Bruggeman solution number from a function name.

    Parameters
    ----------
    func_name : str
        The function name, e.g., 'bruggeman_123_02' or 'bruggeman_710_12'.

    Returns
    -------
    str
        The Bruggeman solution number, e.g., '123.02' or '710.12'.
    """
    if not func_name.startswith("bruggeman_"):
        return ""
    # Remove 'bruggeman_' prefix (10 characters)
    rest = func_name[10:]
    # Split by underscores and take first two parts
    parts = rest.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return ""


def _get_submodule_path(module_name: str) -> str:
    """Convert a module path to a shorter submodule representation.

    Parameters
    ----------
    module_name : str
        The full module path, e.g., 'bruggeman.confined.flow1d'.

    Returns
    -------
    str
        The submodule path, e.g., 'confined.flow1d'.
    """
    if module_name.startswith("bruggeman."):
        return module_name[10:]  # Remove 'bruggeman.' prefix
    return module_name


def get_table_of_contents() -> pd.DataFrame:
    """Generate a table of contents DataFrame for all Bruggeman functions.

    This function scans all exported functions from the bruggeman package and
    creates a DataFrame containing the Bruggeman solution number, submodule path,
    function name, and the first line of each function's docstring.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with columns:
        - 'number': The Bruggeman solution number (e.g., '123.02') (index)
        - 'chapter': The chapter title (e.g., 'A. Phreatic groundwater')
        - 'submodule': The submodule path (e.g., 'confined.flow1d')
        - 'function': The function name (e.g., 'bruggeman_123_02')
        - 'description': The first line of the docstring.

    Examples
    --------
    >>> import bruggeman
    >>> toc = bruggeman.get_table_of_contents()
    >>> print(toc)
    """
    import re

    # Get all functions that start with 'bruggeman_' or are in 'other' module
    func_names = [
        x
        for x in globals()
        if (
            x.startswith("bruggeman_")
            or x in ["h_edelman", "Qx_edelman", "huisman_kemperman"]
        )
        and callable(globals()[x])
    ]

    data = []
    for func_name in sorted(func_names):
        func = globals()[func_name]
        module = inspect.getmodule(func)
        module_name = module.__name__ if module else "unknown"

        # Extract Bruggeman number
        bruggeman_num = _extract_bruggeman_number(func_name)

        # Get docstring
        doc = inspect.getdoc(func)
        if doc:
            first_line = doc.split("\n")[0].strip()
            # Remove the Bruggeman number from the beginning if present
            # Pattern: 123.02 Text... or 123.02Text...
            first_line = re.sub(r"^\d+\.\d+\s*", "", first_line).strip()
        else:
            first_line = ""

        submodule_path = _get_submodule_path(module_name)

        chapters = {
            "phreatic.phreatic": "A. Phreatic groundwater",
            "confined.flow1d": "BI. One-dimensional confined flow",
            "confined.flow2d.radial": "BII. Two-dimensional radial symmetric flow",
            "confined.flow2d.general": "BIII. General two-dimensional flow",
            "multilayer.flow1d": "C. Multi-layer systems",
            "multilayer.radial": "C. Multi-layer systems",
        }

        data.append(
            {
                "number": bruggeman_num,
                "chapter": chapters.get(submodule_path)
                if submodule_path in chapters
                else "",
                "submodule": submodule_path,
                "function": func_name,
                "description": first_line,
            }
        )

    df = pd.DataFrame(data).set_index("number")
    df = df.sort_index(
        key=lambda x: x.map(
            lambda s: tuple(map(int, s.split("."))) if s else (999999, 999999)
        )
    )
    return df
