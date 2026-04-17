# HDMR-Lib

**A unified Python library for HDMR and EMPR**

**HDMR-Lib** provides a consistent interface for decomposing high-dimensional tensors and multivariate functions into interpretable lower-order components. It supports both **High-Dimensional Model Representation (HDMR)** and **Enhanced Multivariate Products Representation (EMPR)** across **NumPy**, **PyTorch**, and **TensorFlow** backends.

```{image} _static/hdmrlib.svg
:alt: HDMR-Lib overview illustration
:width: 85%
:align: center
```

## Interpretable decomposition for high-dimensional structure

HDMR-Lib is designed for research and scientific computing workflows that require decomposition, approximation, and interaction analysis in high-dimensional settings. The library makes it easier to compute lower-order representations, inspect component terms, and work within a unified backend-flexible API.

## Core capabilities

### HDMR and EMPR in one interface
Work with classical HDMR and EMPR-based formulations through a consistent Python workflow.

### Lower-order component analysis
Decompose tensors and multivariate functions into interpretable lower-order terms for inspection and analysis.

### Multi-backend execution
Use the same workflow across NumPy, PyTorch, and TensorFlow, including tensor-based and GPU-enabled pipelines where supported.

## What is HDMR-Lib?

HDMR-Lib is a research-oriented library for studying high-dimensional structure through lower-order representations. It supports decomposition, reconstruction, and component extraction for tensor-valued data and multivariate functions, making it suitable for experimentation, analysis, and scientific software workflows.

The library is particularly useful when interaction structure, approximation quality, and computational flexibility are central to the problem setting.

## Supported backends

HDMR-Lib currently supports the following computational backends:

- **NumPy** for standard array-based workflows
- **PyTorch** for tensor computation and GPU-enabled pipelines
- **TensorFlow** for TensorFlow-based numerical workflows

## Explore the documentation

- Start with the **Installation** guide to set up the library
- Visit the **User Guide** for the main workflow
- See **Fundamentals** for the underlying decomposition concepts
- Browse the **Examples** gallery for practical use cases
- Use the **API Reference** for detailed documentation

```{toctree}
:maxdepth: 2
:hidden:

install
user_guide/index
fundamentals
auto_examples/index
api
team
```
