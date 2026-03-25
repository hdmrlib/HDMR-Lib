"""
Testing for metrics functions
"""

from hdmrlib.metrics import mean_squared_error, sensitivity_analysis
from hdmrlib import EMPR, set_backend
import numpy as np
from numpy.testing import assert_, assert_almost_equal, assert_raises
import pytest


def test_mean_squared_error():
    """Test for MSE calculation"""
    set_backend('numpy')

    tensor1 = np.random.rand(3, 3, 3)
    tensor2 = tensor1.copy()
    mse = mean_squared_error(tensor1, tensor2)
    assert_almost_equal(mse, 0.0, decimal=10)

    tensor3 = np.random.rand(3, 3, 3)
    mse = mean_squared_error(tensor1, tensor3)
    assert_(mse > 0)


def test_mean_squared_error_with_reconstruction():
    """Test for MSE between original tensor and its reconstruction"""
    set_backend('numpy')
    np.random.seed(42)

    tensor = np.random.rand(3, 3, 3)
    model = EMPR(tensor)
    result = model.reconstruct(order=2)

    mse = mean_squared_error(tensor, result)
    assert_(mse >= 0)


def test_sensitivity_analysis_basic():
    """Test for basic sensitivity analysis"""
    set_backend('numpy')
    np.random.seed(42)

    tensor = np.random.rand(4, 4, 4)
    model = EMPR(tensor)
    components = model.components()

    result = sensitivity_analysis(tensor, components, ['g_1', 'g_2'], return_dict=True)

    assert_('individual_effects' in result)
    assert_('combined_effect' in result)
    assert_('g_1' in result['individual_effects'])
    assert_('g_2' in result['individual_effects'])


def test_sensitivity_analysis_combined():
    """Test for combined effect calculation"""
    set_backend('numpy')
    np.random.seed(42)

    tensor = np.random.rand(3, 3, 3)
    model = EMPR(tensor)
    components = model.components()

    result = sensitivity_analysis(tensor, components, ['g_1', 'g_2', 'g_1,2'], return_dict=True)

    combined = result['combined_effect']
    individual_sum = sum(result['individual_effects'].values())

    assert_almost_equal(combined, individual_sum, decimal=6)


def test_sensitivity_analysis_all_components():
    """Test for sensitivity analysis with all components"""
    set_backend('numpy')
    np.random.seed(42)

    tensor = np.random.rand(3, 3, 3)
    model = EMPR(tensor)
    components = model.components()

    component_list = [k for k in components.keys() if k != 'g_0']
    result = sensitivity_analysis(tensor, components, component_list, return_dict=True)

    total = result['combined_effect']
    assert_(total >= 0)


def test_sensitivity_analysis_invalid_component():
    """Test for invalid component name handling"""
    set_backend('numpy')
    np.random.seed(42)

    tensor = np.random.rand(3, 3, 3)
    model = EMPR(tensor)
    components = model.components()

    result = sensitivity_analysis(tensor, components, ['g_1', 'invalid_component'], return_dict=True)
    assert_('g_1' in result['individual_effects'])
    assert_('invalid_component' not in result['individual_effects'])


def test_sensitivity_analysis_print_mode():
    """Test for sensitivity analysis print mode (return_dict=False)"""
    set_backend('numpy')
    np.random.seed(42)

    tensor = np.random.rand(3, 3, 3)
    model = EMPR(tensor)
    components = model.components()

    result = sensitivity_analysis(tensor, components, ['g_1', 'g_2'], return_dict=False)
    assert_(result is None)


def test_sensitivity_analysis_with_g0():
    """Test for sensitivity analysis including g_0 component"""
    set_backend('numpy')
    np.random.seed(42)

    tensor = np.random.rand(3, 3, 3)
    model = EMPR(tensor)
    components = model.components()
    g0_value = components['g_0']

    result = sensitivity_analysis(
        tensor, components, ['g_0', 'g_1'], g0=g0_value, return_dict=True
    )

    assert_('g_0' in result['individual_effects'])
    assert_('g_1' in result['individual_effects'])


def test_sensitivity_analysis_g0_without_param():
    """Test that g_0 analysis without g0 param skips it with warning"""
    set_backend('numpy')
    np.random.seed(42)

    tensor = np.random.rand(3, 3, 3)
    model = EMPR(tensor)
    components = model.components()

    result = sensitivity_analysis(
        tensor, components, ['g_0', 'g_1'], return_dict=True
    )
    assert_('g_0' not in result['individual_effects'])
    assert_('g_1' in result['individual_effects'])


def _is_torch_available():
    try:
        import torch
        return True
    except ImportError:
        return False


def _is_tensorflow_available():
    try:
        import tensorflow
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _is_torch_available(), reason="PyTorch is not installed")
def test_mse_with_torch_tensors():
    """Test MSE computation when inputs are torch tensors"""
    import torch

    t1 = torch.rand(3, 3, 3, dtype=torch.float64)
    t2 = t1.clone()

    mse = mean_squared_error(t1, t2)
    assert_almost_equal(mse, 0.0, decimal=10)

    t3 = torch.rand(3, 3, 3, dtype=torch.float64)
    mse = mean_squared_error(t1, t3)
    assert_(mse > 0)


@pytest.mark.skipif(not _is_tensorflow_available(), reason="TensorFlow is not installed")
def test_mse_with_tf_tensors():
    """Test MSE computation when inputs are tensorflow tensors"""
    import tensorflow as tf

    np.random.seed(42)
    arr = np.random.rand(3, 3, 3)
    t1 = tf.constant(arr, dtype=tf.float64)
    t2 = tf.constant(arr, dtype=tf.float64)

    mse = mean_squared_error(t1, t2)
    assert_almost_equal(mse, 0.0, decimal=10)

    t3 = tf.constant(np.random.rand(3, 3, 3), dtype=tf.float64)
    mse = mean_squared_error(t1, t3)
    assert_(mse > 0)


@pytest.mark.skipif(not _is_torch_available(), reason="PyTorch is not installed")
def test_sensitivity_analysis_with_torch_tensors():
    """Test sensitivity analysis when components are torch tensors"""
    import torch

    set_backend('torch')
    np.random.seed(42)
    tensor = np.random.rand(3, 3, 3)

    model = EMPR(tensor)
    components = model.components()
    torch_tensor = torch.tensor(tensor, dtype=torch.float64)

    result = sensitivity_analysis(
        torch_tensor, components, ['g_1', 'g_2'], return_dict=True
    )
    assert_('g_1' in result['individual_effects'])
    assert_('g_2' in result['individual_effects'])

    set_backend('numpy')


@pytest.mark.skipif(not _is_tensorflow_available(), reason="TensorFlow is not installed")
def test_sensitivity_analysis_with_tf_tensors():
    """Test sensitivity analysis when components are tensorflow tensors"""
    import tensorflow as tf

    set_backend('tensorflow')
    np.random.seed(42)
    tensor = np.random.rand(3, 3, 3)

    model = EMPR(tensor)
    components = model.components()
    tf_tensor = tf.constant(tensor, dtype=tf.float64)

    result = sensitivity_analysis(
        tf_tensor, components, ['g_1', 'g_2'], return_dict=True
    )
    assert_('g_1' in result['individual_effects'])
    assert_('g_2' in result['individual_effects'])

    set_backend('numpy')
