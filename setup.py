#!/usr/bin/env python
"""Build shim. Everything declarative lives in pyproject.toml [project].

All this file still does is the one thing pyproject cannot: build the Cython
`monotonic_align` extension, which needs numpy's headers at build time.

It used to carry the whole package definition, and all of it was upstream's — name
`matcha-tts`, upstream author and URL, no licence field, and `install_requires` read
out of a `requirements.txt` that had drifted in both directions (see pyproject for
what was wrong with it). A wheel built from that metadata would have claimed to be
someone else's package. `make create-package` would then have pushed it to PyPI.
"""

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

exts = [
    Extension(
        name="matcha.utils.monotonic_align.core",
        sources=["matcha/utils/monotonic_align/core.pyx"],
    )
]

setup(
    include_dirs=[numpy.get_include()],
    ext_modules=cythonize(exts, language_level=3),
)
