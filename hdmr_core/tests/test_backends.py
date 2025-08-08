import hdmr_core as hdmr
import numpy as np
import pytest

backends = ["numpy", "torch", "tensorflow", "cupy"]

@pytest.mark.parametrize("backend", backends)
def test_backend_decompose(backend):
    try:
        hdmr.set_backend(backend)
    except Exception:
        pytest.skip(f"Backend {backend} not available.")
    
    tensor = np.random.rand(3, 3, 3)
    
    # Test EMPR for all backends
    try:
        model = hdmr.EMPR(tensor)
        result = model.decompose(order=2)
        assert result is not None
    except NotImplementedError:
        pytest.skip(f"EMPR not implemented for backend {backend}")
    
    # Test HDMR for all except CuPy
    if backend != "cupy":
        try:
            model = hdmr.HDMR(tensor)
            result = model.decompose(order=2)
            assert result is not None
        except NotImplementedError:
            pytest.skip(f"HDMR not implemented for backend {backend}")

def test_backend_management():
    # Test set_backend and get_backend
    hdmr.set_backend("numpy")
    assert hdmr.get_backend() == "numpy"
    
    hdmr.set_backend("torch")
    assert hdmr.get_backend() == "torch"
    
    # Test invalid backend
    with pytest.raises(ValueError):
        hdmr.set_backend("invalid_backend") 