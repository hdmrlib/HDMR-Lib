from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "HDMRLib"
author = "hdmrlib"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_gallery.gen_gallery",
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "sphinx_design",
    "sphinx_copybutton",
]

autosummary_generate = True
templates_path = ["_templates"]
exclude_patterns = ["_build"]
autodoc_mock_imports = ["torch", "tensorflow"]

html_theme = "pydata_sphinx_theme"

html_static_path = ["_static"]
html_js_files = ["custom.js"]
html_css_files = ["custom.css"]
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.jpg"
html_permalinks = False
html_show_sphinx = False

html_context = {
    "default_mode": "light",
}

html_theme_options = {
    "logo": {
        "text": "HDMRLib",
    },
    "navbar_start": ["navbar-logo", "navbar-nav"],
    "navbar_center": [],
    "navbar_end": ["search-field", "navbar-icon-links"],
    "secondary_sidebar_items": ["edit-this-page", "sourcelink"],
    "show_prev_next": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/hdmrlib/HDMRLib",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/hdmrlib/",
            "icon": "fa-brands fa-python",
        },
    ],
}

myst_enable_extensions = ["colon_fence", "deflist", "fieldlist", "html_image", "dollarmath"]

sphinx_gallery_conf = {
    "examples_dirs": "examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"plot_",
    "subsection_order": [
        "examples/general_examples",
        "examples/decomposition_workflows",
        "examples/research_examples",
    ],
    "within_subsection_order": "FileNameSortKey",
    "nested_sections": True,
    "thumbnail_size": (320, 220),
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}
