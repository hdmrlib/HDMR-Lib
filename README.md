[![PyPI version](https://img.shields.io/pypi/v/hdmrlib.svg)](https://pypi.org/project/hdmrlib/)
[![Docs](https://img.shields.io/badge/docs-online-31557f)](https://hdmrlib.github.io/HDMRLib/)
[![Python versions](https://img.shields.io/pypi/pyversions/hdmrlib.svg)](https://pypi.org/project/hdmrlib/)
[![Tests](https://img.shields.io/github/actions/workflow/status/hdmrlib/HDMRLib/tests.yml?branch=main&label=tests)](https://github.com/hdmrlib/HDMRLib/actions)
[![License](https://img.shields.io/github/license/hdmrlib/HDMRLib)](https://github.com/hdmrlib/HDMRLib/blob/main/LICENSE)

# HDMRLib

**HDMRLib** is an open-source Python library for **High-Dimensional Model Representation (HDMR)** and **Enhanced Multivariate Products Representation (EMPR)**. It provides a unified workflow for decomposition, component analysis, and lower-order reconstruction across **NumPy**, **PyTorch**, and **TensorFlow** backends.

## Features

- HDMR and EMPR in one library
- Unified decomposition workflow
- Component extraction and lower-order reconstruction
- NumPy, PyTorch, and TensorFlow support
- Documentation, examples, and tests

## Installation

Install the package from PyPI:

```bash
pip install hdmrlib
```

For development installation:

```bash
git clone https://github.com/hdmrlib/HDMRLib.git
cd HDMRLib
pip install -e .
```

For optional backend dependencies and development tools, see the documentation.

## Quick Example

```python
import numpy as np
from hdmrlib import HDMR

X = np.random.rand(10, 10)

model = HDMR(X, order=2)

X_reconstructed = model.reconstruct()
components = model.components()
```

This computes a second-order HDMR decomposition of `X`, reconstructs the data from the decomposition, and returns the extracted component terms.

The same workflow also applies to `EMPR`:

```python
from hdmrlib import EMPR

model = EMPR(X, order=2)
X_reconstructed = model.reconstruct()
components = model.components()
```

## Documentation

Full documentation is available at:

- **Documentation website:** https://hdmrlib.github.io/HDMRLib/
- **User Guide:** https://hdmrlib.github.io/HDMRLib/user_guide/index.html
- **API Reference:** https://hdmrlib.github.io/HDMRLib/api.html
- **Examples:** https://hdmrlib.github.io/HDMRLib/auto_examples/index.html
- **Fundamentals:** https://hdmrlib.github.io/HDMRLib/fundamentals.html

## Supported Backends

HDMRLib currently supports:

- **NumPy** for standard array-based workflows
- **PyTorch** for tensor computation and backend-integrated workflows
- **TensorFlow** for TensorFlow-based numerical workflows

Backend selection is handled through a unified interface, allowing the same decomposition workflow to be used across supported numerical libraries.

## Testing

To install development dependencies and run the test suite:

```bash
pip install -e ".[dev]"
python -m pytest
```

If you want to test optional backends as well:

```bash
pip install -e ".[dev,all]"
python -m pytest
```

## Citation

If you use **HDMRLib** in academic work, please cite the associated software or publication.

```bibtex
@software{hdmrlib,
  title = {HDMRLib: A Python Library for HDMR and EMPR},
  author = {Pınar Yalçın Güler and Muhammed Enis Şen and Buğra Eyidoğan},
  year = {2026},
  url = {https://github.com/hdmrlib/HDMRLib}
}
```

If a paper citation becomes available, it should be preferred here.

## Contributing

Contributions, bug reports, and feature suggestions are welcome.

Please use the GitHub issue tracker for bug reports and discussions, and open a pull request for proposed changes.

## License

This project is released under the terms of the license included in the repository.
