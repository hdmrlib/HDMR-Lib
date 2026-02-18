# Quickstart

This page shows a minimal end-to-end workflow:
1) choose a backend,
2) build an EMPR/HDMR model,
3) reconstruct,
4) inspect components.

## Minimal example

```python
import numpy as np

from hdmrlib.backends import set_backend
from hdmrlib.empr import EMPR

# 1) backend
set_backend("numpy")

# 2) data (toy example)
X = np.random.randn(128, 8)

# 3) fit / decompose
model = EMPR(X, order=2)   # adjust constructor args to your API
components = model.components()

# 4) reconstruct
X_hat = model.reconstruct()

print("reconstruction shape:", np.asarray(X_hat).shape)
print("components keys:", list(components.keys()))
```

## HDMR vs EMPR (intuition)

- **HDMR** represents a function by decomposing it into low-order interaction terms (e.g., 1st-order, 2nd-order, ...).
- **EMPR** is a related decomposition that emphasizes multiplicative interaction structure and is often convenient for tensor-like data.
- In practice, both provide a structured way to analyze contributions of variables and interactions, and both can be truncated by the chosen **order**.

