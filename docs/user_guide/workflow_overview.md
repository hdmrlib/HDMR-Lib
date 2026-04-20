# Library Organization and Workflow Overview

This page provides a high-level overview of the main HDMRLib objects and the typical workflow used throughout the user guide.

HDMRLib is organized around a small set of core interfaces for decomposition and backend management. In most workflows, users will:

1. prepare input data
2. choose a backend if needed
3. create a decomposition object
4. inspect the extracted components
5. reconstruct lower-order approximations

## Core objects

The main user-facing decomposition classes are:

- **EMPR** for Enhanced Multivariate Products Representation
- **HDMR** for High-Dimensional Model Representation

Both follow the same high-level usage pattern. A decomposition object is created from input data and a target decomposition order.

```python
import numpy as np
from hdmrlib import EMPR, HDMR

X = np.random.rand(10, 10)

empr = EMPR(X, order=2)
hdmr = HDMR(X, order=2)
```

In both cases, the resulting object stores the computed decomposition and provides methods for inspecting or reconstructing it.

## Decomposition objects

A decomposition object represents the result of applying EMPR or HDMR to input data.

Once created, the object is typically used through two main methods:

- `components()` to access extracted component terms
- `reconstruct()` to rebuild the data from the decomposition

This gives a consistent workflow across both formulations:

```python
components = empr.components()
X_reconstructed = empr.reconstruct()
```

The same pattern applies to `hdmr`.

## Working with components

The `components()` method returns the component terms produced by the decomposition.

These components are typically organized by term names that indicate the corresponding lower-order structure or interaction terms. For example, users may request only selected terms:

```python
selected = empr.components(elements=["g_0", "g_1", "g_2", "g_1,2"])
```

Use `components()` when you want to inspect the decomposition itself rather than only its reconstruction.

## Working with reconstructions

The `reconstruct()` method returns a reconstruction of the original data based on the computed decomposition.

```python
X_full = empr.reconstruct()
X_order1 = empr.reconstruct(order=1)
```

A full reconstruction uses all computed terms up to the decomposition order. Passing `order=...` returns a lower-order approximation using only terms up to the requested order.

This is useful when studying how well lower-order structure explains the original data.

## Backend management

HDMRLib supports multiple computational backends. The main backend utilities are:

- `set_backend(...)` to select a backend
- `get_backend()` to inspect the currently active backend

For example:

```python
from hdmrlib import set_backend, get_backend

set_backend("numpy")
backend = get_backend()
```

This allows the same high-level workflow to be used across supported backends such as NumPy, PyTorch, and TensorFlow.

In many cases, the default backend is sufficient. Backend selection becomes more relevant when integrating HDMRLib into tensor-based or GPU-enabled workflows.

## Typical workflow

A typical HDMRLib workflow looks like this:

### 1. Prepare the input
Organize the input data into a supported tensor or array format.

### 2. Select a backend if needed
Use `set_backend(...)` when you want to work with a specific computational backend.

### 3. Create a decomposition object
Instantiate `EMPR(...)` or `HDMR(...)` with the desired decomposition order.

### 4. Inspect the component terms
Use `components()` to examine lower-order terms and selected interactions.

### 5. Reconstruct approximations
Use `reconstruct()` to obtain the full decomposition-based reconstruction or lower-order approximations.

## How to read the user guide

The user guide is organized around the main stages of this workflow:

- **Quick Start** gives a minimal end-to-end example
- **Prepare Input Data** explains input requirements and conventions
- **Run a Decomposition** describes the main decomposition interfaces and options
- **Inspect Components** explains how to access and interpret extracted terms
- **Reconstruct Data** shows how to build lower-order approximations
- **Backends** covers backend selection and usage
- **Troubleshooting** summarizes common issues and fixes

Users new to the library should usually start with **Quick Start** and then continue through the workflow pages in order.

## When to use the API Reference

The user guide explains the typical workflow and main concepts. The **API Reference** is most useful when you need detailed signatures, implementation-level options, or module-specific documentation.

For most first uses of the library, the workflow pages in the user guide should be sufficient.
