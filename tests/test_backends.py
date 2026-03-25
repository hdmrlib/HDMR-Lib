"""
Testing for backend functionality
"""

from hdmrlib import HDMR, EMPR, set_backend, get_backend
from hdmrlib.backends import available_backends
import numpy as np
from numpy.testing import assert_, assert_array_almost_equal
import pytest


def test_backend_switching():
    """Test for backend switching"""
    initial_backend = get_backend()
    assert_(initial_backend in ['numpy', 'torch', 'tensorflow'])

    set_backend('numpy')
    assert_(get_backend() == 'numpy')


def test_invalid_backend():
    """Test for invalid backend name"""
    with pytest.raises(ValueError):
        set_backend('invalid_backend')


def test_available_backends():
    """Test that available_backends returns a list containing numpy"""
    backends = available_backends()
    assert_(isinstance(backends, list))
    assert_('numpy' in backends)


def test_tf_alias():
    """Test that 'tf' is accepted as alias for 'tensorflow'"""
    backends = available_backends()
    if 'tensorflow' in backends:
        set_backend('tf')
        assert_(get_backend() == 'tensorflow')
        set_backend('numpy')
    else:
        pytest.skip("TensorFlow not installed")


def test_numpy_backend():
    """Test for NumPy backend"""
    set_backend('numpy')
    np.random.seed(42)

    tensor = np.random.rand(3, 3, 3)

    hdmr_model = HDMR(tensor)
    hdmr_result = hdmr_model.reconstruct(order=2)
    assert_(hdmr_result is not None)
    assert_(isinstance(hdmr_result, np.ndarray))

    empr_model = EMPR(tensor)
    empr_result = empr_model.reconstruct(order=2)
    assert_(empr_result is not None)
    assert_(isinstance(empr_result, np.ndarray))


def _is_torch_available():
    try:
        import torch
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _is_torch_available(), reason="PyTorch is not installed")
def test_torch_backend():
    """Test for PyTorch backend"""
    import torch

    try:
        set_backend('torch')
    except (ValueError, ImportError):
        pytest.skip("PyTorch backend not available")

    np.random.seed(42)
    tensor = np.random.rand(3, 3, 3)

    hdmr_model = HDMR(tensor)
    hdmr_result = hdmr_model.reconstruct(order=2)
    assert_(hdmr_result is not None)
    assert_(torch.is_tensor(hdmr_result))

    empr_model = EMPR(tensor)
    empr_result = empr_model.reconstruct(order=2)
    assert_(empr_result is not None)
    assert_(torch.is_tensor(empr_result))

    set_backend('numpy')


def _is_tensorflow_available():
    try:
        import tensorflow
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _is_tensorflow_available(), reason="TensorFlow is not installed")
def test_tensorflow_backend():
    """Test for TensorFlow backend"""
    import tensorflow as tf

    try:
        set_backend('tensorflow')
    except (ValueError, ImportError):
        pytest.skip("TensorFlow backend not available")

    np.random.seed(42)
    tensor = np.random.rand(3, 3, 3)

    hdmr_model = HDMR(tensor)
    hdmr_result = hdmr_model.reconstruct(order=2)
    assert_(hdmr_result is not None)

    empr_model = EMPR(tensor)
    empr_result = empr_model.reconstruct(order=2)
    assert_(empr_result is not None)

    set_backend('numpy')


@pytest.mark.skipif(not _is_torch_available(), reason="PyTorch is not installed")
def test_cross_backend_numpy_vs_torch():
    """Test that NumPy and Torch backends produce equivalent results"""
    np.random.seed(42)
    tensor = np.random.rand(3, 3, 3)

    set_backend('numpy')
    np_result = HDMR(tensor).reconstruct(order=2)

    try:
        set_backend('torch')
    except (ValueError, ImportError):
        pytest.skip("Torch backend not available")

    import torch
    torch_result = HDMR(tensor).reconstruct(order=2)
    torch_result_np = torch_result.detach().cpu().numpy()

    assert_array_almost_equal(np_result, torch_result_np, decimal=10)
    set_backend('numpy')


@pytest.mark.skipif(not _is_tensorflow_available(), reason="TensorFlow is not installed")
def test_cross_backend_numpy_vs_tensorflow():
    """Test that NumPy and TensorFlow backends produce equivalent results"""
    np.random.seed(42)
    tensor = np.random.rand(3, 3, 3)

    set_backend('numpy')
    np_result = HDMR(tensor).reconstruct(order=2)

    try:
        set_backend('tensorflow')
    except (ValueError, ImportError):
        pytest.skip("TensorFlow backend not available")

    tf_result = HDMR(tensor).reconstruct(order=2)
    tf_result_np = np.asarray(tf_result)

    assert_array_almost_equal(np_result, tf_result_np, decimal=10)
    set_backend('numpy')


def test_numpy_hdmr_dtype_float32():
    """Test that float32 input is accepted and produces valid results"""
    set_backend('numpy')
    tensor = np.random.rand(3, 3, 3).astype(np.float32)
    model = HDMR(tensor)
    result = model.reconstruct(order=2)
    assert_(result is not None)
    assert_(result.shape == tensor.shape)
    assert_(not np.isnan(result).any())


def test_numpy_hdmr_dtype_float16():
    """Test that float16 input is accepted and produces valid results"""
    set_backend('numpy')
    tensor = np.random.rand(3, 3, 3).astype(np.float16)
    model = HDMR(tensor)
    result = model.reconstruct(order=2)
    assert_(result is not None)
    assert_(result.shape == tensor.shape)
    assert_(not np.isnan(result).any())


def test_numpy_hdmr_dtype_int():
    """Test that integer input is accepted and produces valid results"""
    set_backend('numpy')
    tensor = np.random.randint(0, 10, size=(3, 3, 3))
    model = HDMR(tensor)
    result = model.reconstruct(order=2)
    assert_(result is not None)
    assert_(result.shape == tensor.shape)


def test_numpy_hdmr_list_input():
    """Test that plain Python list input is accepted"""
    set_backend('numpy')
    tensor_list = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    model = HDMR(tensor_list)
    result = model.reconstruct(order=2)
    assert_(result is not None)
    assert_(result.shape == (2, 2, 2))


def test_numpy_empr_dtype_float32():
    """Test that float32 input works for EMPR"""
    set_backend('numpy')
    tensor = np.random.rand(3, 3, 3).astype(np.float32)
    model = EMPR(tensor)
    result = model.reconstruct(order=2)
    assert_(result is not None)
    assert_(result.shape == tensor.shape)
    assert_(not np.isnan(result).any())


@pytest.mark.skipif(not _is_torch_available(), reason="PyTorch is not installed")
def test_torch_dtype_float32():
    """Test that torch float32 input is accepted"""
    import torch

    try:
        set_backend('torch')
    except (ValueError, ImportError):
        pytest.skip("Torch backend not available")

    tensor = np.random.rand(3, 3, 3).astype(np.float32)
    model = HDMR(tensor)
    result = model.reconstruct(order=2)
    assert_(result is not None)
    assert_(not torch.isnan(result).any())
    set_backend('numpy')


# ============================================================
# Torch backend: weight types, support types, singleton, mse
# ============================================================

@pytest.mark.skipif(not _is_torch_available(), reason="PyTorch is not installed")
def test_torch_hdmr_weight_types():
    """Test all HDMR weight types on torch backend"""
    try:
        set_backend('torch')
    except (ValueError, ImportError):
        pytest.skip("Torch backend not available")

    np.random.seed(42)
    tensor = np.random.rand(3, 3, 3)

    for weight in ['avg', 'gaussian', 'chebyshev']:
        model = HDMR(tensor, weight=weight)
        result = model.reconstruct(order=2)
        assert_(result is not None)

    set_backend('numpy')


@pytest.mark.skipif(not _is_torch_available(), reason="PyTorch is not installed")
def test_torch_hdmr_custom_weights():
    """Test HDMR custom weights on torch backend (list and tensor inputs)"""
    import torch

    try:
        set_backend('torch')
    except (ValueError, ImportError):
        pytest.skip("Torch backend not available")

    np.random.seed(42)
    tensor = np.random.rand(3, 3, 3)

    custom_weights_list = [
        np.ones((3, 1)),
        np.ones((3, 1)),
        np.ones((3, 1)),
    ]
    model = HDMR(tensor, weight='custom', custom_weights=custom_weights_list)
    assert_(model.reconstruct(order=2) is not None)

    custom_weights_torch = [
        torch.ones(3, 1, dtype=torch.float64),
        torch.ones(3, 1, dtype=torch.float64),
        torch.ones(3, 1, dtype=torch.float64),
    ]
    model2 = HDMR(tensor, weight='custom', custom_weights=custom_weights_torch)
    assert_(model2.reconstruct(order=2) is not None)

    set_backend('numpy')


@pytest.mark.skipif(not _is_torch_available(), reason="PyTorch is not installed")
def test_torch_hdmr_das_supports():
    """Test HDMR with DAS supports on torch backend"""
    import torch

    try:
        set_backend('torch')
    except (ValueError, ImportError):
        pytest.skip("Torch backend not available")

    np.random.seed(42)
    tensor = np.random.rand(3, 3, 3)

    model = HDMR(tensor, supports='das')
    result = model.reconstruct(order=2)
    assert_(result is not None)
    assert_(result.shape == torch.Size([3, 3, 3]))

    set_backend('numpy')


@pytest.mark.skipif(not _is_torch_available(), reason="PyTorch is not installed")
def test_torch_hdmr_custom_supports():
    """Test HDMR with custom supports on torch backend (list and tensor inputs)"""
    import torch

    try:
        set_backend('torch')
    except (ValueError, ImportError):
        pytest.skip("Torch backend not available")

    np.random.seed(42)
    tensor = np.random.rand(3, 3, 3)

    custom_supports_list = [
        np.linspace(0, 1, 3).reshape(-1, 1),
        np.linspace(0, 1, 3).reshape(-1, 1),
        np.linspace(0, 1, 3).reshape(-1, 1),
    ]
    model = HDMR(tensor, supports='custom', custom_supports=custom_supports_list)
    assert_(model.reconstruct(order=2) is not None)

    custom_supports_torch = [
        torch.linspace(0, 1, 3).reshape(-1, 1).to(torch.float64),
        torch.linspace(0, 1, 3).reshape(-1, 1).to(torch.float64),
        torch.linspace(0, 1, 3).reshape(-1, 1).to(torch.float64),
    ]
    model2 = HDMR(tensor, supports='custom', custom_supports=custom_supports_torch)
    assert_(model2.reconstruct(order=2) is not None)

    set_backend('numpy')


@pytest.mark.skipif(not _is_torch_available(), reason="PyTorch is not installed")
def test_torch_empr_supports():
    """Test EMPR with different support types on torch backend"""
    try:
        set_backend('torch')
    except (ValueError, ImportError):
        pytest.skip("Torch backend not available")

    np.random.seed(42)
    tensor = np.random.rand(3, 3, 3)

    for supports in ['ones', 'das']:
        model = EMPR(tensor, supports=supports)
        result = model.reconstruct(order=2)
        assert_(result is not None)

    set_backend('numpy')


@pytest.mark.skipif(not _is_torch_available(), reason="PyTorch is not installed")
def test_torch_empr_custom_supports():
    """Test EMPR with custom supports on torch backend"""
    import torch

    try:
        set_backend('torch')
    except (ValueError, ImportError):
        pytest.skip("Torch backend not available")

    np.random.seed(42)
    tensor = np.random.rand(3, 3, 3)

    custom_supports_list = [
        np.linspace(0, 1, 3).reshape(-1, 1),
        np.linspace(0, 1, 3).reshape(-1, 1),
        np.linspace(0, 1, 3).reshape(-1, 1),
    ]
    model = EMPR(tensor, supports='custom', custom_supports=custom_supports_list)
    assert_(model.reconstruct(order=2) is not None)

    custom_supports_torch = [
        torch.linspace(0, 1, 3).reshape(-1, 1).to(torch.float64),
        torch.linspace(0, 1, 3).reshape(-1, 1).to(torch.float64),
        torch.linspace(0, 1, 3).reshape(-1, 1).to(torch.float64),
    ]
    model2 = EMPR(tensor, supports='custom', custom_supports=custom_supports_torch)
    assert_(model2.reconstruct(order=2) is not None)

    set_backend('numpy')


@pytest.mark.skipif(not _is_torch_available(), reason="PyTorch is not installed")
def test_torch_singleton_dimension():
    """Test singleton dimension handling on torch backend"""
    try:
        set_backend('torch')
    except (ValueError, ImportError):
        pytest.skip("Torch backend not available")

    np.random.seed(42)
    tensor = np.random.rand(3, 1, 3)

    hdmr_model = HDMR(tensor)
    assert_(1 not in hdmr_model.dimensions)
    assert_(hdmr_model.reconstruct(order=2) is not None)

    empr_model = EMPR(tensor)
    assert_(1 not in empr_model.dimensions)
    assert_(empr_model.reconstruct(order=2) is not None)

    set_backend('numpy')


@pytest.mark.skipif(not _is_torch_available(), reason="PyTorch is not installed")
def test_torch_calculate_mse():
    """Test internal calculate_mse on torch backend"""
    try:
        set_backend('torch')
    except (ValueError, ImportError):
        pytest.skip("Torch backend not available")

    np.random.seed(42)
    tensor = np.random.rand(3, 3, 3)

    hdmr_mse = HDMR(tensor)._model.calculate_mse(2)
    assert_(isinstance(hdmr_mse, float))
    assert_(hdmr_mse >= 0)

    empr_mse = EMPR(tensor)._model.calculate_mse(2)
    assert_(isinstance(empr_mse, float))
    assert_(empr_mse >= 0)

    set_backend('numpy')


# ============================================================
# TensorFlow backend: weight types, support types, singleton, mse
# ============================================================

@pytest.mark.skipif(not _is_tensorflow_available(), reason="TensorFlow is not installed")
def test_tf_hdmr_weight_types():
    """Test all HDMR weight types on tensorflow backend"""
    try:
        set_backend('tensorflow')
    except (ValueError, ImportError):
        pytest.skip("TensorFlow backend not available")

    np.random.seed(42)
    tensor = np.random.rand(3, 3, 3)

    for weight in ['avg', 'gaussian', 'chebyshev']:
        model = HDMR(tensor, weight=weight)
        result = model.reconstruct(order=2)
        assert_(result is not None)

    set_backend('numpy')


@pytest.mark.skipif(not _is_tensorflow_available(), reason="TensorFlow is not installed")
def test_tf_hdmr_custom_weights():
    """Test HDMR custom weights on tensorflow backend (list and tensor inputs)"""
    import tensorflow as tf

    try:
        set_backend('tensorflow')
    except (ValueError, ImportError):
        pytest.skip("TensorFlow backend not available")

    np.random.seed(42)
    tensor = np.random.rand(3, 3, 3)

    custom_weights_list = [
        np.ones((3, 1)),
        np.ones((3, 1)),
        np.ones((3, 1)),
    ]
    model = HDMR(tensor, weight='custom', custom_weights=custom_weights_list)
    assert_(model.reconstruct(order=2) is not None)

    custom_weights_tf = [
        tf.ones((3, 1), dtype=tf.float64),
        tf.ones((3, 1), dtype=tf.float64),
        tf.ones((3, 1), dtype=tf.float64),
    ]
    model2 = HDMR(tensor, weight='custom', custom_weights=custom_weights_tf)
    assert_(model2.reconstruct(order=2) is not None)

    set_backend('numpy')


@pytest.mark.skipif(not _is_tensorflow_available(), reason="TensorFlow is not installed")
def test_tf_hdmr_das_supports():
    """Test HDMR with DAS supports on tensorflow backend"""
    try:
        set_backend('tensorflow')
    except (ValueError, ImportError):
        pytest.skip("TensorFlow backend not available")

    np.random.seed(42)
    tensor = np.random.rand(3, 3, 3)

    model = HDMR(tensor, supports='das')
    result = model.reconstruct(order=2)
    assert_(result is not None)

    set_backend('numpy')


@pytest.mark.skipif(not _is_tensorflow_available(), reason="TensorFlow is not installed")
def test_tf_hdmr_custom_supports():
    """Test HDMR with custom supports on tensorflow backend"""
    import tensorflow as tf

    try:
        set_backend('tensorflow')
    except (ValueError, ImportError):
        pytest.skip("TensorFlow backend not available")

    np.random.seed(42)
    tensor = np.random.rand(3, 3, 3)

    custom_supports_list = [
        np.linspace(0, 1, 3).reshape(-1, 1),
        np.linspace(0, 1, 3).reshape(-1, 1),
        np.linspace(0, 1, 3).reshape(-1, 1),
    ]
    model = HDMR(tensor, supports='custom', custom_supports=custom_supports_list)
    assert_(model.reconstruct(order=2) is not None)

    custom_supports_tf = [
        tf.constant(np.linspace(0, 1, 3).reshape(-1, 1), dtype=tf.float64),
        tf.constant(np.linspace(0, 1, 3).reshape(-1, 1), dtype=tf.float64),
        tf.constant(np.linspace(0, 1, 3).reshape(-1, 1), dtype=tf.float64),
    ]
    model2 = HDMR(tensor, supports='custom', custom_supports=custom_supports_tf)
    assert_(model2.reconstruct(order=2) is not None)

    set_backend('numpy')


@pytest.mark.skipif(not _is_tensorflow_available(), reason="TensorFlow is not installed")
def test_tf_empr_supports():
    """Test EMPR with different support types on tensorflow backend"""
    try:
        set_backend('tensorflow')
    except (ValueError, ImportError):
        pytest.skip("TensorFlow backend not available")

    np.random.seed(42)
    tensor = np.random.rand(3, 3, 3)

    for supports in ['ones', 'das']:
        model = EMPR(tensor, supports=supports)
        result = model.reconstruct(order=2)
        assert_(result is not None)

    set_backend('numpy')


@pytest.mark.skipif(not _is_tensorflow_available(), reason="TensorFlow is not installed")
def test_tf_empr_custom_supports():
    """Test EMPR with custom supports on tensorflow backend"""
    import tensorflow as tf

    try:
        set_backend('tensorflow')
    except (ValueError, ImportError):
        pytest.skip("TensorFlow backend not available")

    np.random.seed(42)
    tensor = np.random.rand(3, 3, 3)

    custom_supports_list = [
        np.linspace(0, 1, 3).reshape(-1, 1),
        np.linspace(0, 1, 3).reshape(-1, 1),
        np.linspace(0, 1, 3).reshape(-1, 1),
    ]
    model = EMPR(tensor, supports='custom', custom_supports=custom_supports_list)
    assert_(model.reconstruct(order=2) is not None)

    custom_supports_tf = [
        tf.constant(np.linspace(0, 1, 3).reshape(-1, 1), dtype=tf.float64),
        tf.constant(np.linspace(0, 1, 3).reshape(-1, 1), dtype=tf.float64),
        tf.constant(np.linspace(0, 1, 3).reshape(-1, 1), dtype=tf.float64),
    ]
    model2 = EMPR(tensor, supports='custom', custom_supports=custom_supports_tf)
    assert_(model2.reconstruct(order=2) is not None)

    set_backend('numpy')


@pytest.mark.skipif(not _is_tensorflow_available(), reason="TensorFlow is not installed")
def test_tf_singleton_dimension():
    """Test singleton dimension handling on tensorflow backend"""
    try:
        set_backend('tensorflow')
    except (ValueError, ImportError):
        pytest.skip("TensorFlow backend not available")

    np.random.seed(42)
    tensor = np.random.rand(3, 1, 3)

    hdmr_model = HDMR(tensor)
    assert_(1 not in tuple(hdmr_model.dimensions))
    assert_(hdmr_model.reconstruct(order=2) is not None)

    empr_model = EMPR(tensor)
    assert_(1 not in tuple(empr_model.dimensions))
    assert_(empr_model.reconstruct(order=2) is not None)

    set_backend('numpy')


@pytest.mark.skipif(not _is_tensorflow_available(), reason="TensorFlow is not installed")
def test_tf_calculate_mse():
    """Test internal calculate_mse on tensorflow backend"""
    try:
        set_backend('tensorflow')
    except (ValueError, ImportError):
        pytest.skip("TensorFlow backend not available")

    np.random.seed(42)
    tensor = np.random.rand(3, 3, 3)

    hdmr_mse = HDMR(tensor)._model.calculate_mse(2)
    assert_(isinstance(hdmr_mse, float))
    assert_(hdmr_mse >= 0)

    empr_mse = EMPR(tensor)._model.calculate_mse(2)
    assert_(isinstance(empr_mse, float))
    assert_(empr_mse >= 0)

    set_backend('numpy')


# ============================================================
# Torch backend: validation errors
# ============================================================

@pytest.mark.skipif(not _is_torch_available(), reason="PyTorch is not installed")
def test_torch_hdmr_custom_weights_none_raises():
    """Test custom weight without weights raises on torch backend"""
    try:
        set_backend('torch')
    except (ValueError, ImportError):
        pytest.skip("Torch backend not available")

    tensor = np.random.rand(3, 3, 3)
    with pytest.raises(ValueError):
        HDMR(tensor, weight='custom')

    set_backend('numpy')


@pytest.mark.skipif(not _is_torch_available(), reason="PyTorch is not installed")
def test_torch_hdmr_custom_weights_wrong_length_raises():
    """Test wrong number of custom weights raises on torch backend"""
    try:
        set_backend('torch')
    except (ValueError, ImportError):
        pytest.skip("Torch backend not available")

    tensor = np.random.rand(3, 3, 3)
    with pytest.raises(ValueError):
        HDMR(tensor, weight='custom', custom_weights=[np.ones((3, 1))])

    set_backend('numpy')


@pytest.mark.skipif(not _is_torch_available(), reason="PyTorch is not installed")
def test_torch_hdmr_custom_supports_none_raises():
    """Test custom support without supports raises on torch backend"""
    try:
        set_backend('torch')
    except (ValueError, ImportError):
        pytest.skip("Torch backend not available")

    tensor = np.random.rand(3, 3, 3)
    with pytest.raises(ValueError):
        HDMR(tensor, supports='custom')

    set_backend('numpy')


@pytest.mark.skipif(not _is_torch_available(), reason="PyTorch is not installed")
def test_torch_hdmr_custom_supports_wrong_length_raises():
    """Test wrong number of custom supports raises on torch backend"""
    try:
        set_backend('torch')
    except (ValueError, ImportError):
        pytest.skip("Torch backend not available")

    tensor = np.random.rand(3, 3, 3)
    with pytest.raises(ValueError):
        HDMR(tensor, supports='custom', custom_supports=[np.ones((3, 1))])

    set_backend('numpy')


@pytest.mark.skipif(not _is_torch_available(), reason="PyTorch is not installed")
def test_torch_empr_custom_supports_none_raises():
    """Test EMPR custom support without supports raises on torch backend"""
    try:
        set_backend('torch')
    except (ValueError, ImportError):
        pytest.skip("Torch backend not available")

    tensor = np.random.rand(3, 3, 3)
    with pytest.raises(ValueError):
        EMPR(tensor, supports='custom')

    set_backend('numpy')


@pytest.mark.skipif(not _is_torch_available(), reason="PyTorch is not installed")
def test_torch_empr_custom_supports_wrong_length_raises():
    """Test EMPR wrong number of custom supports raises on torch backend"""
    try:
        set_backend('torch')
    except (ValueError, ImportError):
        pytest.skip("Torch backend not available")

    tensor = np.random.rand(3, 3, 3)
    with pytest.raises(ValueError):
        EMPR(tensor, supports='custom', custom_supports=[np.ones((3, 1))])

    set_backend('numpy')


# ============================================================
# TensorFlow backend: validation errors
# ============================================================

@pytest.mark.skipif(not _is_tensorflow_available(), reason="TensorFlow is not installed")
def test_tf_hdmr_custom_weights_none_raises():
    """Test custom weight without weights raises on tf backend"""
    try:
        set_backend('tensorflow')
    except (ValueError, ImportError):
        pytest.skip("TensorFlow backend not available")

    tensor = np.random.rand(3, 3, 3)
    with pytest.raises(ValueError):
        HDMR(tensor, weight='custom')

    set_backend('numpy')


@pytest.mark.skipif(not _is_tensorflow_available(), reason="TensorFlow is not installed")
def test_tf_hdmr_custom_weights_wrong_length_raises():
    """Test wrong number of custom weights raises on tf backend"""
    try:
        set_backend('tensorflow')
    except (ValueError, ImportError):
        pytest.skip("TensorFlow backend not available")

    tensor = np.random.rand(3, 3, 3)
    with pytest.raises(ValueError):
        HDMR(tensor, weight='custom', custom_weights=[np.ones((3, 1))])

    set_backend('numpy')


@pytest.mark.skipif(not _is_tensorflow_available(), reason="TensorFlow is not installed")
def test_tf_hdmr_custom_supports_none_raises():
    """Test custom support without supports raises on tf backend"""
    try:
        set_backend('tensorflow')
    except (ValueError, ImportError):
        pytest.skip("TensorFlow backend not available")

    tensor = np.random.rand(3, 3, 3)
    with pytest.raises(ValueError):
        HDMR(tensor, supports='custom')

    set_backend('numpy')


@pytest.mark.skipif(not _is_tensorflow_available(), reason="TensorFlow is not installed")
def test_tf_hdmr_custom_supports_wrong_length_raises():
    """Test wrong number of custom supports raises on tf backend"""
    try:
        set_backend('tensorflow')
    except (ValueError, ImportError):
        pytest.skip("TensorFlow backend not available")

    tensor = np.random.rand(3, 3, 3)
    with pytest.raises(ValueError):
        HDMR(tensor, supports='custom', custom_supports=[np.ones((3, 1))])

    set_backend('numpy')


@pytest.mark.skipif(not _is_tensorflow_available(), reason="TensorFlow is not installed")
def test_tf_empr_custom_supports_none_raises():
    """Test EMPR custom support without supports raises on tf backend"""
    try:
        set_backend('tensorflow')
    except (ValueError, ImportError):
        pytest.skip("TensorFlow backend not available")

    tensor = np.random.rand(3, 3, 3)
    with pytest.raises(ValueError):
        EMPR(tensor, supports='custom')

    set_backend('numpy')


@pytest.mark.skipif(not _is_tensorflow_available(), reason="TensorFlow is not installed")
def test_tf_empr_custom_supports_wrong_length_raises():
    """Test EMPR wrong number of custom supports raises on tf backend"""
    try:
        set_backend('tensorflow')
    except (ValueError, ImportError):
        pytest.skip("TensorFlow backend not available")

    tensor = np.random.rand(3, 3, 3)
    with pytest.raises(ValueError):
        EMPR(tensor, supports='custom', custom_supports=[np.ones((3, 1))])

    set_backend('numpy')
