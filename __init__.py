"""
HDMR-Lib: High Dimensional Model Representation and Enhanced Multivariate Products Representation

A Python library for tensor decomposition using HDMR and EMPR methods with multi-backend support.
"""

__version__ = "0.1.0"
__author__ = "HDMR-Lib Contributors"
__license__ = "MIT"

from hdmr import HDMR
from empr import EMPR
from backends import set_backend, get_backend

__all__ = ['HDMR', 'EMPR', 'set_backend', 'get_backend', '__version__'] 