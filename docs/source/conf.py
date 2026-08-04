# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add project root to path so autodoc can find the package
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))

# -- Project information -----------------------------------------------------

project = "gwModels"
copyright = "2024, Tousif Islam"
author = "Tousif Islam"
release = "0.2.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Mock optional dependencies so autodoc works without them ----------------

autodoc_mock_imports = [
    "lal",
    "lalsimulation",
    "gwsurrogate",
    "pyseobnr",
    "torch",
    "nflows",
    "sklearn",
]

# -- Options for autodoc -----------------------------------------------------

autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "private-members": False,
}

# -- Options for Napoleon (NumPy/Google docstring support) -------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# -- Options for intersphinx -------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
}
html_static_path = ["_static"]

# -- Options for nbsphinx ---------------------------------------------------

nbsphinx_execute = "never"
