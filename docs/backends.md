# Backends

HDMR-Lib can run on multiple backends. Some are optional dependencies.

## Selecting a backend

```python
from hdmrlib.backends import set_backend, get_backend

set_backend("numpy")
print(get_backend())
```

Common values:
- `"numpy"` (always available)
- `"torch"` (requires PyTorch)
- `"tensorflow"` (requires TensorFlow)

## What if Torch/TensorFlow is not installed?

If an optional backend is missing, importing or selecting it will fail.

That is expected and intentional:
- the base installation stays lightweight,
- heavy dependencies remain opt-in.

To enable:
- `pip install -r requirements-torch.txt`
- `pip install -r requirements-tensorflow.txt`

