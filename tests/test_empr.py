# Copyright (c) 2025 HDMRLib Contributors
# SPDX-License-Identifier: MIT

"""
Testing for EMPR decomposition
"""

from hdmrlib import EMPR, set_backend
import numpy as np
from numpy.testing import assert_, assert_array_almost_equal, assert_raises


def test_empr_reconstruct():
    """Test for EMPR reconstruction"""
    set_backend('numpy')
    np.random.seed(42)

    tensor = np.random.rand(4, 4, 4)
    model = EMPR(tensor)
    result = model.reconstruct(order=2)

    assert_(result is not None)
    assert_(result.shape == tensor.shape)
    assert_(not np.isnan(result).any())
    assert_(not np.isinf(result).any())


def test_empr_components():
    """Test for EMPR component extraction"""
    set_backend('numpy')
    np.random.seed(42)

    tensor = np.random.rand(3, 3, 3)
    model = EMPR(tensor)

    components = model.components()

    assert_(isinstance(components, dict))
    assert_(len(components) > 0)

    expected_keys = ['g_0', 'g_1', 'g_2', 'g_3', 'g_1,2', 'g_1,3', 'g_2,3']
    for key in expected_keys:
        assert_(key in components)


def test_empr_components_subset():
    """Test for EMPR component extraction with element filter"""
    set_backend('numpy')
    np.random.seed(42)

    tensor = np.random.rand(3, 3, 3)
    model = EMPR(tensor)

    subset = model.components(elements=['g_1', 'g_2'])
    assert_(len(subset) == 2)
    assert_('g_1' in subset)
    assert_('g_2' in subset)
    assert_('g_3' not in subset)


def test_empr_order_approximation():
    """Test for order approximation quality"""
    set_backend('numpy')
    np.random.seed(42)

    tensor = np.random.rand(3, 3, 3)
    model = EMPR(tensor)

    result_order1 = model.reconstruct(order=1)
    result_order2 = model.reconstruct(order=2)
    result_order3 = model.reconstruct(order=3)

    error1 = np.linalg.norm(tensor - result_order1)
    error2 = np.linalg.norm(tensor - result_order2)
    error3 = np.linalg.norm(tensor - result_order3)

    assert_(error2 <= error1)
    assert_(error3 <= error2)
    assert_(error3 < 1e-10)


def test_empr_supports():
    """Test for different support types"""
    set_backend('numpy')
    np.random.seed(42)

    tensor = np.random.rand(3, 3, 3)
    support_types = ['ones', 'das']

    for support_type in support_types:
        model = EMPR(tensor, supports=support_type)
        result = model.reconstruct(order=2)

        assert_(result is not None)
        assert_(result.shape == tensor.shape)


def test_empr_das_vs_ones():
    """Test for DAS vs ones support comparison"""
    set_backend('numpy')
    np.random.seed(42)

    tensor = np.random.rand(4, 4, 4)

    model_das = EMPR(tensor, supports='das')
    result_das = model_das.reconstruct(order=2)
    error_das = np.linalg.norm(tensor - result_das)

    model_ones = EMPR(tensor, supports='ones')
    result_ones = model_ones.reconstruct(order=2)
    error_ones = np.linalg.norm(tensor - result_ones)

    assert_(result_das.shape == tensor.shape)
    assert_(result_ones.shape == tensor.shape)
    assert_(error_das < 10.0)
    assert_(error_ones < 10.0)


def test_empr_custom_supports():
    """Test for custom support vectors"""
    set_backend('numpy')
    np.random.seed(42)

    tensor = np.random.rand(3, 3, 3)
    custom_supports = [
        np.linspace(0, 1, 3).reshape(-1, 1),
        np.exp(np.linspace(-1, 1, 3)).reshape(-1, 1),
        np.sin(np.linspace(0, np.pi, 3)).reshape(-1, 1)
    ]

    model = EMPR(tensor, supports='custom', custom_supports=custom_supports)
    result = model.reconstruct(order=2)

    assert_(result is not None)
    assert_(result.shape == tensor.shape)


def test_empr_different_shapes():
    """Test for different tensor shapes"""
    set_backend('numpy')
    np.random.seed(42)

    shapes = [(3, 3, 3), (4, 5, 6), (2, 2, 2, 2), (5, 5), (10, 10, 10)]

    for shape in shapes:
        tensor = np.random.rand(*shape)
        model = EMPR(tensor)
        result = model.reconstruct(order=min(2, len(shape)))

        assert_(result.shape == tensor.shape)


def test_empr_zero_tensor():
    """Test for zero tensor edge case"""
    set_backend('numpy')

    tensor = np.zeros((3, 3, 3))
    model = EMPR(tensor)
    result = model.reconstruct(order=2)

    assert_(np.allclose(result, 0, atol=1e-10))


def test_empr_constant_tensor():
    """Test for constant tensor -- full order should give exact reconstruction"""
    set_backend('numpy')

    constant_value = 5.0
    tensor = np.full((3, 3, 3), constant_value)
    model = EMPR(tensor)

    result_full = model.reconstruct(order=3)
    error_full = np.linalg.norm(tensor - result_full)
    assert_(error_full < 1e-10)


def test_empr_default_order():
    """Test that default order=2 is used when no order is passed to reconstruct"""
    set_backend('numpy')
    np.random.seed(42)

    tensor = np.random.rand(3, 3, 3)
    model = EMPR(tensor, order=2)
    result_default = model.reconstruct()
    result_explicit = model.reconstruct(order=2)

    assert_array_almost_equal(result_default, result_explicit)


def test_empr_attributes():
    """Test that EMPR exposes dimensions, weights, and support_vectors"""
    set_backend('numpy')
    np.random.seed(42)

    tensor = np.random.rand(4, 5, 6)
    model = EMPR(tensor)

    assert_(model.dimensions == (4, 5, 6))
    assert_(len(model.weights) == 3)
    assert_(len(model.support_vectors) == 3)


def test_empr_singleton_dimension():
    """Test that singleton dimensions are squeezed automatically"""
    set_backend('numpy')
    np.random.seed(42)

    tensor = np.random.rand(3, 1, 3)
    model = EMPR(tensor)
    assert_(1 not in model.dimensions)
    result = model.reconstruct(order=2)
    assert_(result is not None)
    assert_(not np.isnan(result).any())


def test_empr_custom_supports_as_lists():
    """Test custom supports provided as plain Python lists (not ndarrays)"""
    set_backend('numpy')
    np.random.seed(42)

    tensor = np.random.rand(3, 3, 3)
    custom_supports = [
        [[1.0], [2.0], [3.0]],
        [[1.0], [2.0], [3.0]],
        [[1.0], [2.0], [3.0]],
    ]

    model = EMPR(tensor, supports='custom', custom_supports=custom_supports)
    result = model.reconstruct(order=2)
    assert_(result is not None)
    assert_(result.shape == tensor.shape)


def test_empr_custom_supports_none_raises():
    """Test that custom support type without supports raises ValueError"""
    set_backend('numpy')
    tensor = np.random.rand(3, 3, 3)
    assert_raises(ValueError, EMPR, tensor, supports='custom')


def test_empr_custom_supports_wrong_length_raises():
    """Test that wrong number of custom supports raises ValueError"""
    set_backend('numpy')
    tensor = np.random.rand(3, 3, 3)
    assert_raises(
        ValueError, EMPR, tensor,
        supports='custom', custom_supports=[np.ones((3, 1))]
    )


def test_empr_calculate_mse_via_model():
    """Test the internal calculate_mse method"""
    set_backend('numpy')
    np.random.seed(42)

    tensor = np.random.rand(3, 3, 3)
    model = EMPR(tensor)
    mse = model._model.calculate_mse(2)
    assert_(isinstance(mse, float))
    assert_(mse >= 0)
