# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys


sys.path.insert(0, os.path.abspath('../../arc'))
# -- Project information -----------------------------------------------------

project = 'ARC'
copyright = '2022, Databricks Inc'
author = 'Robert Whiffin, Marcell Ferencz, Milos Colic, Databricks Inc'

# The full version, including alpha/beta/rc tags
release = "v0.0.1"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_material",
    "nbsphinx",
    "sphinx_tabs.tabs",
    "sphinx.ext.githubpages",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.todo",
    "sphinx.ext.autodoc"
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

autodoc_typehints = "description"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', ".env"]
source_suffix = [".rst", ".md"]

pygments_style = 'sphinx'
nbsphinx_execute = 'never'
napoleon_use_admonition_for_notes = True
sphinx_tabs_disable_tab_closing = True
todo_include_todos = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'piccolo_theme'
html_theme_path = ['_theme']

html_logo = '_images/logo.png'

html_static_path = ['_static']
html_css_files = ["css/custom.css"]

