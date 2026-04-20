# Quick Start

HDMRLib supports two related decomposition formulations:

- **HDMR** is useful when you want to work explicitly with HDMR-based formulations
- **EMPR** provides the same high-level workflow through an alternative decomposition formulation

In both cases, the workflow is the same:

**input data** → **decomposition** → **components** → **reconstruction**

## End-to-end example with HDMR

```python
import numpy as np
from hdmrlib import HDMR

X = np.random.rand(10, 10)

hdmr = HDMR(X, order=2)

X_reconstructed = hdmr.reconstruct()
components = hdmr.components()
```

This computes an HDMR decomposition of `X` up to order 2.

- `hdmr.reconstruct()` returns a reconstruction based on the computed decomposition
- `hdmr.components()` returns the extracted component terms

## Understanding the outputs

The `components()` method returns the component functions produced by the decomposition. These are typically organized by term name, such as lower-order individual terms and interaction terms.

For example, a call such as:

```python
selected = hdmr.components(elements=["g_0", "g_1", "g_2", "g_1,2"])
```

returns only the requested components.

This is useful when you want to inspect specific lower-order terms or selected interactions rather than the full decomposition output.

## Reconstructing lower-order approximations

```python
X_order1 = hdmr.reconstruct(order=1)
```

This reconstructs `X` using only terms up to order 1.

Use `reconstruct(order=...)` when you want to study how much of the data can be approximated using only lower-order structure. This is especially useful when comparing simplified approximations against the full reconstruction.

## Running the same workflow with EMPR

```python
import numpy as np
from hdmrlib import EMPR

X = np.random.rand(10, 10)

empr = EMPR(X, order=2)

X_reconstructed = empr.reconstruct()
components = empr.components()
```

The same workflow applies to EMPR: compute a decomposition, inspect the extracted components, and reconstruct the data from the resulting representation.

## Choosing between HDMR and EMPR

As a starting point:

- use **HDMR** when you want to work specifically with HDMR-based formulations
- use **EMPR** when you want to compare an alternative formulation within the same workflow

Since both follow a consistent interface, it is easy to compare them within the same workflow.
