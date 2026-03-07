# Quick Start

This page provides a short introduction to the main workflow of **HDMR-Lib**.

For installation instructions, see the Installation page.

```{contents}
:local:
:depth: 2
```

## Overview

HDMR-Lib provides tools for high-dimensional model representation and related approximation methods for multivariate functions represented on tensor grids.

The library currently supports multiple computational backends and includes two main decomposition approaches:

- EMPR
- HDMR

## Basic usage pattern

A typical workflow in HDMR-Lib consists of the following steps:

1. prepare a tensor input
2. choose a computational backend
3. create an EMPR or HDMR model
4. reconstruct an approximation at a chosen order
5. inspect the resulting components

## Minimal example

```python
import numpy as np
import hdmrlib as h

X = np.random.rand(8, 8, 8)

model = h.EMPR(X, order=2)
approx = model.reconstruct()
components = model.components()
```

## What this example does

In this example:

- `X` is the input tensor
- `EMPR(X, order=2)` creates an EMPR model with second-order approximation as the default setting
- `reconstruct()` computes the approximation
- `components()` returns the available decomposition components

You can also override the approximation order at reconstruction time:

```python
approx_order_1 = model.reconstruct(order=1)
approx_order_2 = model.reconstruct(order=2)
```

You can retrieve only selected components by passing their keys:

```python
selected = model.components(elements=[(0,), (1,), (0, 1)])
```

## Backend selection

HDMR-Lib supports multiple computational backends. In most cases, NumPy is the best starting point for first-time users.

```python
import hdmrlib as h

h.set_backend("numpy")
print(h.get_backend())
print(h.available_backends())
```

The backend selector also accepts `"tf"` as an alias for TensorFlow.

## Using HDMR instead

The same workflow can also be used with HDMR:

```python
import numpy as np
import hdmrlib as h

X = np.random.rand(8, 8, 8)

model = h.HDMR(X, order=2)
approx = model.reconstruct()
components = model.components()
```

## Notes

Both `EMPR` and `HDMR` expose the following commonly used attributes:

- `dimensions`
- `weights`
- `support_vectors`

For example:

```python
print(model.dimensions)
print(model.weights)
print(model.support_vectors)
```

## Next steps

The following sections explain the main parts of the library in more detail:

- Overview of HDMR-Lib
- Computational Backends
- Working with Tensor Data
- EMPR Decomposition
- HDMR Decomposition
