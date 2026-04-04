# Reconstruct Data

## Reconstruct with the Decomposition Order

Both `EMPR` and `HDMR` provide `reconstruct()`.

```python
import numpy as np
from hdmrlib import EMPR

X = np.random.rand(10, 10)

empr = EMPR(X, order=2)
X_reconstructed = empr.reconstruct()
```

If `order` is not provided, `reconstruct()` uses the decomposition order stored in the object.

## Reconstruct Lower-Order Approximations

```python
X_order1 = empr.reconstruct(order=1)
```

This reconstructs the data using terms up to order 1.

## HDMR

The same pattern applies to `HDMR`.

```python
from hdmrlib import HDMR

hdmr = HDMR(X, order=2)
X_reconstructed = hdmr.reconstruct()
X_order1 = hdmr.reconstruct(order=1)
```

## Notes

- `reconstruct()` returns an approximation tensor
- `order=None` uses the object's stored decomposition order
- lower-order reconstruction can be requested explicitly

