# Inspect Components

## Get All Components

Both `EMPR` and `HDMR` expose component terms through `components()`.

```python
import numpy as np
from hdmrlib import EMPR

X = np.random.rand(10, 10)

empr = EMPR(X, order=2)
components = empr.components()
```

`components` is a dictionary of component terms.

## Inspect Available Keys

```python
print(components.keys())
```

Component names follow the library's internal naming convention:

- `g_1`
- `g_2`
- `g_1,2`

The indices are one-based.

For a two-dimensional input, `g_1` and `g_2` are first-order terms, and `g_1,2` is the second-order interaction term.

## Select Specific Components

Use `elements` to request only specific component keys.

```python
selected = empr.components(elements=["g_1", "g_2", "g_1,2"])
```

`elements` must match the available component names exactly.

## HDMR

The same pattern applies to `HDMR`.

```python
from hdmrlib import HDMR

hdmr = HDMR(X, order=2)
components = hdmr.components()
selected = hdmr.components(elements=["g_1", "g_2", "g_1,2"])
```

## Notes

- `components()` returns the stored component dictionary
- component keys use one-based indexing
- selected keys must exist in the component dictionary

## Next

- **Reconstruct Data** shows how to build lower-order approximations
- **Work with Backends** covers backend-specific tensor behavior
- **Fundamentals** explains the meaning of component terms
