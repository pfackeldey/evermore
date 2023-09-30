from __future__ import annotations

import dilax

project = dilax.__name__
copyright = dilax.__copyright__
author = dilax.__author__
version = release = dilax.__version__

language = "en"

templates_path = ["_templates"]
html_static_path = ["_static"]
master_doc = "index"
source_suffix = ".rst"
pygments_style = "sphinx"
add_module_names = False

exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
]

html_title = f"{project} v{version}"
html_theme = "sphinx_book_theme"
html_theme_options = {
    "logo_only": True,
    "home_page_in_toc": True,
    "show_navbar_depth": 2,
    "show_toc_level": 2,
    "repository_url": "https://github.com/pfackeldey/dilax",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
}

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "autodocsumm",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_lfs_content",
]

myst_enable_extensions = [
    "colon_fence",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

nitpick_ignore = [
    ("py:class", "_io.StringIO"),
    ("py:class", "_io.BytesIO"),
]

always_document_param_types = True

autodoc_member_order = "bysource"


def setup(app):
    app.add_css_file("styles_sphinx_book_theme.css")
