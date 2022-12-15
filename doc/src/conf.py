
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

project = 'cocohelper'
copyright = '2022, AILAB-BH'


# -- Extensions ---------------------------------------------------------------
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autodoc.typehints',

    'sphinx.ext.autosummary',  # Automatic generation of API -- please disable autoapi
    # 'autoapi.extension',     # Automatic generation of API -- please disable autosummary

    'sphinx_rtd_theme',  # Read The Docs theme
    'myst_parser',  # Parse markdown files
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.extlinks',
]

# -- Extensions Configurations ------------------------------------------------

# Autosummary configurations
autosummary_generate = False  # we are calling autosummary directly with ailab-apigen !
numpydoc_show_class_members = False

# Autodoc Configuration
autodoc_inherit_docstrings = False  # Disable docstring inheritance
autodoc_typehints = 'description'
autodoc_typehints_format = 'short'
autodoc_docstring_signature = True  # Enable overriding of function signatures in the first line of the docstring.
autodoc_class_signature = 'separated'  # "mixed" Display the signature with the class name, "separated" Display the signature as a method.
autodoc_member_order = 'groupwise'
autoclass_content = 'both'

# Autoapi configurations (currently not used)
autoapi_generate_api_docs = True
autoapi_keep_files = True
autoapi_add_toctree_entry = True
autoapi_type = 'python'
autoapi_dirs = ['../../src']
autoapi_options = ['members',
                   'undoc-members',
                   'private-members',
                   'show-inheritance',
                   'show-module-summary',
                   'special-members',
                   'imported-members']
autoapi_python_class_content = 'both'  # Select 'class' or 'init' docstring, or 'both' for both
autoapi_member_order = 'groupwise'

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False  # True: types processed and eventually will become links to the type documentation
napoleon_type_aliases = None
napoleon_attr_annotations = True

# MyST examples for .md file parsing, for more info: https://myst-parser.readthedocs.io/en/latest/syntax/optional.html
myst_enable_extensions = [
    "amsmath",  # enable LaTeX math environment
    "colon_fence",
    "deflist",
    "dollarmath",  # parsing of dollar $ and $$ encapsulated math
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    # "strikethrough",
    "substitution",
    "tasklist",
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# Patterns, relative to ailab_apigen-src directory, that match files and directories to ignore.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
pygments_style = 'sphinx'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_theme_options = {
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': -1,
    'includehidden': True,
    'titles_only': False
}


def autodoc_skip_member(app, what, name, obj, skip, options):
    """
    This method will return True when attribute 'name' starts with and ends with "__"
    """
    doskip = False
    # Skip magic methods
    if name.endswith('__') and name.startswith('__'):
        doskip = True

    return doskip

def setup(app):
    app.add_css_file('style.css')
    app.connect('autodoc-skip-member', autodoc_skip_member)


# autodoc_mock_imports = ['numpy', 'pandas', 'pycocotools', 'PIL', 'scipy', 'cv2', 'shapely', 'matplotlib']
