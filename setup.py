"""Optional C++ kernels; the default package remains usable without a compiler."""
import os
from setuptools import setup

extensions = []
if os.environ.get('PDBLEND_BUILD_NATIVE') == '1':
    from pybind11.setup_helpers import Pybind11Extension
    extensions = [Pybind11Extension('pdblend._native', ['native/kernels.cpp'],
                                   cxx_std=17, extra_compile_args=['-ffp-contract=off'])]
setup(ext_modules=extensions)
