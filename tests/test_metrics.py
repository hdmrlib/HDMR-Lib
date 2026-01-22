"""
Testing for metrics functions
"""

from hdmrlib.metrics import mean_squared_error, sensitivity_analysis
from hdmrlib import EMPR, set_backend
import numpy as np
from numpy.testing import assert_, assert_almost_equal, assert_raises


def test_mean_squared_error():
    """Test for MSE calculation"""
    set_backend('numpy')
    
    # Test with identical arrays
    tensor1 = np.random.rand(3, 3, 3)
    tensor2 = tensor1.copy()
    mse = mean_squared_error(tensor1, tensor2)
    assert_almost_equal(mse, 0.0, decimal=10)
    
    # Test with different arrays
    tensor3 = np.random.rand(3, 3, 3)
    mse = mean_squared_error(tensor1, tensor3)
    assert_(mse > 0)


def test_mean_squared_error_different_backends():
    """Test for MSE with different backend tensors"""
    set_backend('numpy')
    
    tensor = np.random.rand(3, 3, 3)
    model = EMPR(tensor)
    result = model.decompose(order=2)
    
    mse = mean_squared_error(tensor, result)
    assert_(mse >= 0)


def test_sensitivity_analysis_basic():
    """Test for basic sensitivity analysis"""
    set_backend('numpy')
    
    tensor = np.random.rand(4, 4, 4)
    model = EMPR(tensor)
    components = model.components(max_order=2)
    
    # Test with selected components
    result = sensitivity_analysis(tensor, components, ['g_1', 'g_2'], return_dict=True)
    
    assert_('individual_effects' in result)
    assert_('combined_effect' in result)
    assert_('g_1' in result['individual_effects'])
    assert_('g_2' in result['individual_effects'])


def test_sensitivity_analysis_combined():
    """Test for combined effect calculation"""
    set_backend('numpy')
    
    tensor = np.random.rand(3, 3, 3)
    model = EMPR(tensor)
    components = model.components(max_order=2)
    
    # Test combined effect
    result = sensitivity_analysis(tensor, components, ['g_1', 'g_2', 'g_1,2'], return_dict=True)
    
    # Combined should equal sum of individual effects
    combined = result['combined_effect']
    individual_sum = sum(result['individual_effects'].values())
    
    assert_almost_equal(combined, individual_sum, decimal=6)


def test_sensitivity_analysis_all_components():
    """Test for sensitivity analysis with all components"""
    set_backend('numpy')
    
    tensor = np.random.rand(3, 3, 3)
    model = EMPR(tensor)
    components = model.components(max_order=2)
    
    # Analyze all components
    component_list = list(components.keys())
    result = sensitivity_analysis(tensor, components, component_list, return_dict=True)
    
    # Total should be reasonable percentage
    total = result['combined_effect']
    assert_(0 <= total <= 100)


def test_sensitivity_analysis_invalid_component():
    """Test for invalid component name handling"""
    set_backend('numpy')
    
    tensor = np.random.rand(3, 3, 3)
    model = EMPR(tensor)
    components = model.components(max_order=2)
    
    # Should handle invalid component gracefully
    result = sensitivity_analysis(tensor, components, ['g_1', 'invalid_component'], return_dict=True)
    assert_('g_1' in result['individual_effects'])
    assert_('invalid_component' not in result['individual_effects'])

