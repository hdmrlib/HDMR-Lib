# HDMR-Lib

**HDMR-Lib** is a Python library for **High-Dimensional Model Representation (HDMR)** and
**Enhanced Multivariate Products Representation (EMPR)** for tensor and function decomposition.

It is designed for studying high-dimensional structure through lower-order component
representations, with a unified interface across multiple computational backends.

## What is HDMR-Lib?

HDMR-Lib provides tools for decomposing multivariate functions and tensor-valued data into
interpretable lower-order terms. It supports both classical **HDMR** formulations and
**EMPR-based** representations within a common Python workflow.

The library is intended for research and scientific computing settings where decomposition,
interaction analysis, approximation, and backend flexibility are important.

## Core Features

- HDMR and EMPR-based decomposition
- Tensor and function-oriented workflows
- Multi-backend support through NumPy, PyTorch, and TensorFlow
- Component extraction for lower-order analysis
- A unified API for experimentation and research use

## Supported Backends

HDMR-Lib currently supports:

- **NumPy** for standard array-based workflows
- **PyTorch** for tensor computation and GPU-enabled pipelines
- **TensorFlow** for TensorFlow-based numerical workflows

```{toctree}
:maxdepth: 2
:hidden:

install
user_guide/index
fundamentals
examples
api
```

