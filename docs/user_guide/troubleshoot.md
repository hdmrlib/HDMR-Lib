# Troubleshoot

## Backend Not Available

If you select a backend that is not installed, the library raises an import error.

```python
from hdmrlib import set_backend

set_backend("torch")
```

Install the missing backend and try again.

```bash
pip install hdmrlib[torch]
```

The same applies to TensorFlow.

```bash
pip install hdmrlib[tensorflow]
```

## Unknown Backend Name

If the backend name is not recognized, the library raises a value error.

```python
set_backend("unknown_backend")
```

Use one of the supported backend names:

- `numpy`
- `torch`
- `tensorflow`
- `tf`

## No Available Backend

If no supported backend is available, the library raises a runtime error when it tries to create a backend instance.

Make sure that at least one supported backend is installed.

## Singleton Dimensions

If the input tensor contains singleton dimensions, the backend implementation squeezes them and prints a warning.

For example, an input with shape `(1, 10, 10)` is converted to `(10, 10)` internally.

Check the tensor shape before running the decomposition if the exact dimensional structure matters.

## Missing Custom Supports

If you use `supports="custom"` without providing `custom_supports`, the library raises a value error.

```python
from hdmrlib import EMPR

empr = EMPR(X, order=2, supports="custom")
```

Provide one support vector per input dimension.

## Invalid Number of Custom Supports

If the number of custom supports does not match the number of tensor dimensions, the library raises a value error.

```python
custom_supports = [np.ones((10, 1))]
```

For a two-dimensional tensor, provide two support vectors.

## Missing Custom Weights in HDMR

If you use `weight="custom"` in `HDMR` without providing `custom_weights`, the library raises a value error.

```python
from hdmrlib import HDMR

hdmr = HDMR(X, order=2, weight="custom")
```

Provide one weight vector per input dimension.

## Invalid Number of Custom Weights

If the number of custom weights does not match the number of tensor dimensions, the library raises a value error.

For a three-dimensional tensor, provide three weight vectors.

## Component Key Errors

`components(elements=[...])` expects existing component keys.

```python
components = empr.components(elements=["g_1", "g_2", "g_1,2"])
```

Check available keys first:

```python
print(empr.components().keys())
```

Use the returned keys exactly as they appear.
