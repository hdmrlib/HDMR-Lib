# Installation

## Prerequisites

- Python 3.8 or higher

---

## Install from PyPI

Install the base package with NumPy backend:

```bash
pip install hdmrlib
```

---

## Optional Backends

HDMRLib supports multiple computational backends. You can install them via extras:

### PyTorch backend

```bash
pip install hdmrlib[torch]
```

### TensorFlow backend

```bash
pip install hdmrlib[tensorflow]
```

### All backends

```bash
pip install hdmrlib[all]
```

---

## Install from Source

Use this option if you want to work with the latest version of the codebase:

```bash
git clone https://github.com/hdmrlib/HDMRLib.git
cd HDMRLib
pip install -e .
```

---

## Development Installation

For development and contributing:

```bash
pip install -e ".[dev]"
```

---

## Verify Installation

You can verify the installation and backend setup:

```python
from hdmrlib import EMPR

# Create sample data
import numpy as np
X = np.random.rand(10, 10)

# Run decomposition
empr = EMPR(X, order=2)
components = empr.components()

print(type(components))
```
---

## Troubleshooting

### Backend not found

Ensure the backend is installed:

```bash
pip install hdmrlib[torch]
```

or

```bash
pip install hdmrlib[tensorflow]
```

---

### Import errors

Check installation:

```bash
pip show hdmrlib
```

If needed, reinstall:

```bash
pip install --upgrade --force-reinstall hdmrlib
```