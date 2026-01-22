"""
Testing for EMPR decomposition
"""

from hdmrlib import EMPR, set_backend
import numpy as np
from numpy.testing import assert_, assert_array_almost_equal, assert_raises


def test_empr_decompose():
    """Test for EMPR decomposition"""
    set_backend('numpy')
    
    # Test with 3D tensor
    tensor = np.random.rand(4, 4, 4)
    model = EMPR(tensor)
    result = model.decompose(order=2)
    
    assert_(result is not None)
    assert_(result.shape == tensor.shape)
    assert_(not np.isnan(result).any())
    assert_(not np.isinf(result).any())


def test_empr_components():
    """Test for EMPR component extraction"""
    set_backend('numpy')
    
    tensor = np.random.rand(3, 3, 3)
    model = EMPR(tensor)
    
    # Test component extraction
    components = model.components(max_order=2)
    
    assert_(isinstance(components, dict))
    assert_(len(components) > 0)
    
    # Check expected keys
    expected_keys = ['g_1', 'g_2', 'g_3', 'g_1,2', 'g_1,3', 'g_2,3']
    for key in expected_keys:
        assert_(key in components)


def test_empr_order_approximation():
    """Test for order approximation quality"""
    set_backend('numpy')
    
    tensor = np.random.rand(3, 3, 3)
    model = EMPR(tensor)
    
    # Decompose at different orders
    result_order1 = model.decompose(order=1)
    result_order2 = model.decompose(order=2)
    result_order3 = model.decompose(order=3)
    
    # Calculate errors
    error1 = np.linalg.norm(tensor - result_order1)
    error2 = np.linalg.norm(tensor - result_order2)
    error3 = np.linalg.norm(tensor - result_order3)
    
    # Higher order should have lower error
    assert_(error2 <= error1)
    assert_(error3 <= error2)
    assert_(error3 < 1e-10)  # Full order should be exact


def test_empr_supports():
    """Test for different support types"""
    set_backend('numpy')
    
    tensor = np.random.rand(3, 3, 3)
    support_types = ['ones', 'das']
    
    for support_type in support_types:
        model = EMPR(tensor, supports=support_type)
        result = model.decompose(order=2)
        
        assert_(result is not None)
        assert_(result.shape == tensor.shape)


def test_empr_das_vs_ones():
    """Test for DAS vs ones support comparison"""
    set_backend('numpy')
    
    tensor = np.random.rand(4, 4, 4)
    
    # Test with DAS supports
    model_das = EMPR(tensor, supports='das')
    result_das = model_das.decompose(order=2)
    error_das = np.linalg.norm(tensor - result_das)
    
    # Test with ones supports
    model_ones = EMPR(tensor, supports='ones')
    result_ones = model_ones.decompose(order=2)
    error_ones = np.linalg.norm(tensor - result_ones)
    
    # Both should produce valid results
    assert_(result_das.shape == tensor.shape)
    assert_(result_ones.shape == tensor.shape)
    assert_(error_das < 10.0)  # Reasonable error bound
    assert_(error_ones < 10.0)


def test_empr_custom_supports():
    """Test for custom support vectors"""
    set_backend('numpy')
    
    tensor = np.random.rand(3, 3, 3)
    custom_supports = [
        np.linspace(0, 1, 3).reshape(-1, 1),
        np.exp(np.linspace(-1, 1, 3)).reshape(-1, 1),
        np.sin(np.linspace(0, np.pi, 3)).reshape(-1, 1)
    ]
    
    model = EMPR(tensor, supports='custom', custom_supports=custom_supports)
    result = model.decompose(order=2)
    
    assert_(result is not None)
    assert_(result.shape == tensor.shape)


def test_empr_different_shapes():
    """Test for different tensor shapes"""
    set_backend('numpy')
    
    shapes = [(3, 3, 3), (4, 5, 6), (2, 2, 2, 2), (5, 5), (10, 10, 10)]
    
    for shape in shapes:
        tensor = np.random.rand(*shape)
        model = EMPR(tensor)
        result = model.decompose(order=min(2, len(shape)))
        
        assert_(result.shape == tensor.shape)


def test_empr_zero_tensor():
    """Test for zero tensor edge case"""
    set_backend('numpy')
    
    tensor = np.zeros((3, 3, 3))
    model = EMPR(tensor)
    result = model.decompose(order=2)
    
    # Result should be close to zero
    assert_(np.allclose(result, 0, atol=1e-10))


def test_empr_constant_tensor():
    """Test for constant tensor"""
    set_backend('numpy')
    
    constant_value = 5.0
    tensor = np.full((3, 3, 3), constant_value)
    model = EMPR(tensor)
    result = model.decompose(order=2)
    
    # Result should approximate the constant
    error = np.linalg.norm(tensor - result)
    assert_(error < 1e-6)
