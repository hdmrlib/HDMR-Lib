# Installation

## Install from PyPI

```bash
pip install hdmrlib
```

## Development install

```bash
git clone https://github.com/hdmrlib/HDMR-Lib.git
cd HDMR-Lib
pip install -e ".[dev]"
```

> If your project does not define extras yet, use:
>
> ```bash
> pip install -e .
> pip install -r requirements-dev.txt
> ```

## Optional backends

HDMR-Lib supports optional compute backends. If you do not install them, the corresponding backend will not be available.

### PyTorch backend

Install PyTorch first (choose the right command for your OS/CUDA), then:

```bash
pip install -r requirements-torch.txt
```

### TensorFlow backend

Install TensorFlow first (CPU or GPU build depending on your setup), then:

```bash
pip install -r requirements-tensorflow.txt
```

## Backend selection (quick example)

```python
from hdmrlib.backends import set_backend, get_backend

set_backend("numpy")   # or "torch" / "tensorflow" if installed
print(get_backend())
```
