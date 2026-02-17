from __future__ import annotations
import os
import sys

# Repo root'u (bir üst klasör) Python path'e ekle ki `hdmrlib` import edilebilsin
sys.path.insert(0, os.path.abspath(".."))

project = "HDMR-Lib"
author = "hdmrlib"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_autodoc_typehints",
]

autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/hdmrlib/HDMR-Lib",
    "navbar_end": ["navbar-icon-links"],
}

myst_enable_extensions = ["colon_fence", "deflist", "fieldlist"]

