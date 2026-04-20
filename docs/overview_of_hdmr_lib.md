# Library Overview

HDMRLib is a Python library for constructing and analyzing high-dimensional decompositions of multivariate functions represented on tensor grids.

The library provides practical tools for approximating high-dimensional functions and studying their interaction structure through structured component representations.

## Core functionality

HDMRLib implements two decomposition methods:

- `EMPR`
- `HDMR`

Both methods construct decompositions from tensor inputs and provide utilities for reconstructing approximations and extracting component functions.

These decompositions make it possible to analyze how different variables and interactions contribute to the overall behavior of a multivariate function.

## Core objects

The primary user-facing objects in HDMRLib are the model classes:

- `EMPR`
- `HDMR`

These classes are initialized with a tensor input representing a sampled multivariate function.

```python
import numpy as np
import hdmrlib as h

X = np.random.rand(8, 8, 8)

model = h.EMPR(X, order=2)
approx = model.reconstruct()
components = model.components()
```

The same workflow applies to the `HDMR` model.

## Backend support

HDMRLib includes a backend abstraction layer that allows computations to run on different numerical frameworks.

Currently supported backends include:

- NumPy
- PyTorch
- TensorFlow

The backend can be selected through the public interface of the library.

## Library structure

At a high level, HDMRLib operates around three main elements:

1. **tensor inputs**, representing sampled multivariate functions
2. **decomposition models** (`EMPR` and `HDMR`)
3. **reconstruction and component analysis tools**

Together, these components allow users to construct structured approximations of high-dimensional functions and analyze their interaction structure.
