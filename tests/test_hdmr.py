import hdmr_core as hdmr
import numpy as np

def test_hdmr_decompose():
    hdmr.set_backend('numpy')
    tensor = np.random.rand(3, 3, 3)
    model = hdmr.HDMR(tensor)
    result = model.decompose(order=2)
    assert result is None  # Placeholder, update when implemented 