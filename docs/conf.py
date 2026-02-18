from __future__ import annotations

import os
import sys

# Repo root'u Python path'e ekle (hdmrlib import edilsin)
sys.path.insert(0, os.path.abspath(".."))

project = "HDMR-Lib"
author = "hdmrlib"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_autodoc_typehints",
    "sphinx_design",
    "sphinx_copybutton",
]

autosummary_generate = True
templates_path = ["_templates"]
exclude_patterns = ["_build"]

# Optional deps yüzünden docs build patlamasın:
autodoc_mock_imports = ["torch", "tensorflow"]

html_theme = "pydata_sphinx_theme"

html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/logo.jpg"
html_favicon = "_static/logo.jpg"

html_theme_options = {
    "github_url": "https://github.com/hdmrlib/HDMR-Lib",
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink"],
    "show_prev_next": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/hdmrlib/HDMR-Lib",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/hdmrlib/",
            "icon": "fa-brands fa-python",
        },
    ],
}

myst_enable_extensions = ["colon_fence", "deflist", "fieldlist"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

