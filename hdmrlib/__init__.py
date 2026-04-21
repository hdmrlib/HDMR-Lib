# Copyright (c) 2025 HDMRLib Contributors
# SPDX-License-Identifier: MIT

"""
HDMRLib: High Dimensional Model Representation and Enhanced Multivariate Products Representation

A Python library for tensor decomposition using HDMR and EMPR methods with multi-backend support.
"""

__version__ = "0.1.2"
__author__ = "HDMRLib Contributors"
__license__ = "MIT"

from .hdmr import HDMR
from .empr import EMPR
from .backends import set_backend, get_backend, available_backends

__all__ = ['HDMR', 'EMPR', 'set_backend', 'get_backend', '__version__'] 
