# Decompose Data

## HDMR

```python
import numpy as np
from hdmrlib import HDMR

X = np.random.rand(10, 10)

hdmr = HDMR(X, order=2)
```

`HDMR` uses the same top-level call pattern.

## HDMR Weight and Support Options

```python
hdmr = HDMR(X, order=2, weight="avg", supports="ones")
```

Supported weight modes:

- `"avg"` or `"average"`
- `"gaussian"`
- `"chebyshev"`
- `"custom"`

Supported support modes:

- `"das"`
- `"ones"`
- `"custom"`

Use `custom_weights` when `weight="custom"` and `custom_supports` when `supports="custom"`.

```python
custom_weights = [
    np.ones((10, 1)),
    np.ones((10, 1)),
]

custom_supports = [
    np.ones((10, 1)),
    np.ones((10, 1)),
]

hdmr = HDMR(
    X,
    order=2,
    weight="custom",
    custom_weights=custom_weights,
    supports="custom",
    custom_supports=custom_supports,
)
```

## EMPR

```python
import numpy as np
from hdmrlib import EMPR

X = np.random.rand(10, 10)

empr = EMPR(X, order=2)
```

`EMPR` takes the input tensor as the first argument and the decomposition order as `order`.

## EMPR Support Options

```python
empr = EMPR(X, order=2, supports="das")
```

Supported support modes:

- `"das"`
- `"ones"`
- `"custom"`

Use `custom_supports` when `supports="custom"`.

```python
custom_supports = [
    np.ones((10, 1)),
    np.ones((10, 1)),
]

empr = EMPR(
    X,
    order=2,
    supports="custom",
    custom_supports=custom_supports,
)
```

## What You Get Back

Both calls return decomposition objects. The next steps are typically:

- inspect component terms
- reconstruct lower-order approximations
- compare different decomposition orders
