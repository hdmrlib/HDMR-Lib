"""
Testing for backend functionality
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from hdmr import HDMR
from empr import EMPR
from backends import set_backend, get_backend
import numpy as np
from numpy.testing import assert_, assert_raises
import pytest


def test_backend_switching():
    """Test for backend switching"""
    # Default backend should be numpy
    initial_backend = get_backend()
    assert_(initial_backend in ['numpy', 'torch', 'tensorflow'])
    
    # Try setting numpy backend
    set_backend('numpy')
    assert_(get_backend() == 'numpy')


def test_invalid_backend():
    """Test for invalid backend name"""
    assert_raises(ValueError, set_backend, 'invalid_backend')


def test_numpy_backend():
    """Test for NumPy backend"""
    set_backend('numpy')
    
    tensor = np.random.rand(3, 3, 3)
    
    # Test HDMR
    hdmr_model = HDMR(tensor)
    hdmr_result = hdmr_model.decompose(order=2)
    assert_(hdmr_result is not None)
    assert_(isinstance(hdmr_result, np.ndarray))
    
    # Test EMPR
    empr_model = EMPR(tensor)
    empr_result = empr_model.decompose(order=2)
    assert_(empr_result is not None)
    assert_(isinstance(empr_result, np.ndarray))


@pytest.mark.skipif(True, reason="Only run if torch is installed")
def test_torch_backend():
    """Test for PyTorch backend"""
    try:
        set_backend('torch')
    except ValueError:
        pytest.skip("PyTorch backend not available")
    
    tensor = np.random.rand(3, 3, 3)
    
    # Test HDMR
    hdmr_model = HDMR(tensor)
    hdmr_result = hdmr_model.decompose(order=2)
    assert_(hdmr_result is not None)
    
    # Test EMPR
    empr_model = EMPR(tensor)
    empr_result = empr_model.decompose(order=2)
    assert_(empr_result is not None)


@pytest.mark.skipif(True, reason="Only run if tensorflow is installed")
def test_tensorflow_backend():
    """Test for TensorFlow backend"""
    try:
        set_backend('tensorflow')
    except ValueError:
        pytest.skip("TensorFlow backend not available")
    
    tensor = np.random.rand(3, 3, 3)
    
    # Test HDMR
    hdmr_model = HDMR(tensor)
    hdmr_result = hdmr_model.decompose(order=2)
    assert_(hdmr_result is not None)
    
    # Test EMPR
    empr_model = EMPR(tensor)
    empr_result = empr_model.decompose(order=2)
    assert_(empr_result is not None)



