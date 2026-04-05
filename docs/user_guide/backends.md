# Backends

## Check the Current Backend

```python
from hdmrlib import get_backend

print(get_backend())
```

The library keeps one active backend at a time.

## Select a Backend

```python
from hdmrlib import set_backend

set_backend("numpy")
set_backend("torch")
set_backend("tensorflow")
```

Backend names are case-insensitive.

`"tf"` can also be used as an alias for `"tensorflow"`.

```python
set_backend("tf")
```

## List Available Backends

```python
from hdmrlib.backends import available_backends

print(available_backends())
```

This returns the backends that are currently available in the environment.

## Default Backend

NumPy is used by default when available. Otherwise, the first available backend is selected.

## Input Conversion

The active backend converts input data internally:

- NumPy backend converts inputs to NumPy arrays
- PyTorch backend converts inputs to Torch tensors
- TensorFlow backend converts inputs to TensorFlow tensors

All backend implementations use `float64` internally.

## Missing Backends

If you select a backend that is not installed, the library raises an import error.

For example:

- `set_backend("torch")` raises an error if the Torch backend is not available
- `set_backend("tensorflow")` raises an error if the TensorFlow backend is not available

## Notes

- only one backend is active at a time
- available backends depend on installed dependencies
