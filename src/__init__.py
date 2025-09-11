# In this __init__.py file, I'm doing two key things for my project structure.
#
# Package Marker:
# This file's primary role is to tell Python that the 'src' directory should be
# treated as a package. This is what allows me to organise my code into multiple
# files and use the relative dot imports (e.g., `from . import config`) that
# keep my project structure clean and maintainable.
# Centralise my SHAP Detection:
# I wanted to make the 'shap' library an optional dependency. To avoid writing
# a try/except block in every file that might use it, I've centralised the check
# here. I create a boolean variable, `HAS_SHAP`, which I can then import anywhere
# in my project. This makes the rest of my code cleaner

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# The `__all__` variable here defines the public API of my `src` package.
# It specifies which names are imported when I use a wildcard import like
# `from src import *`. I've set it to only expose my `HAS_SHAP` flag.
__all__ = ["HAS_SHAP"]
