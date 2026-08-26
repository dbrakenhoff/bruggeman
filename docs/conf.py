# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Bruggeman"
copyright = "2025, Davíd Brakenhoff"
author = "Davíd Brakenhoff"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    "sphinx.ext.mathjax",
    "sphinx_design",
    "sphinxcontrib.bibtex",
    "autoapi.extension",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

# -- myst_nb options ------------------------------------------------------------------

nb_execution_allow_errors = True  # Allow errors in notebooks, to see the error online
nb_execution_mode = "auto"
nb_merge_streams = True
myst_enable_extensions = ["dollarmath", "amsmath"]
myst_dmath_double_inline = True

# -- Options for BibTeX -------------------------------------------------------

bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"

# -- AutoAPI settings ---------------------------------------------------------
autoapi_dirs = ["../bruggeman"]
autoapi_root = "api"
autoapi_options = [
    "show-module-summary"
]
autoapi_own_page_level = "module"
suppress_warnings = ["autoapi"]

# Keep API signatures and section navigation compact by omitting module prefixes
# (e.g., show `River` instead of `timflow.steady.linesink.River`).
add_module_names = False

# Keep local object TOC entries compact in the right sidebar.
toc_object_entries_show_parents = "hide"