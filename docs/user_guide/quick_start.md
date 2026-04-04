# Quick Start

## EMPR

```python
import numpy as np
from hdmrlib import EMPR

X = np.random.rand(10, 10)

empr = EMPR(X, order=2)

X_reconstructed = empr.reconstruct()
components = empr.components()
```

This computes an EMPR decomposition of `X` up to order 2, reconstructs the data from the decomposition, and returns the component terms.

## Lower-Order Reconstruction

```python
X_order1 = empr.reconstruct(order=1)
```

This reconstructs `X` using only lower-order terms up to order 1.

## Selected Components

```python
selected = empr.components(elements=[(0,), (1,), (0, 1)])
```

This returns only the requested components.

## HDMR

```python
import numpy as np
from hdmrlib import HDMR

X = np.random.rand(10, 10)

hdmr = HDMR(X, order=2)

X_reconstructed = hdmr.reconstruct()
components = hdmr.components()
```

The same usage pattern applies to HDMR.
