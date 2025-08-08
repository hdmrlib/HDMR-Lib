from .numpy_backend import NumPyBackend
from .torch_backend import TorchBackend
from .tensorflow_backend import TensorFlowBackend
from .cupy_backend import CuPyBackend

_CURRENT_BACKEND = 'numpy'
_BACKEND_MAP = {
    'numpy': NumPyBackend(),
    'torch': TorchBackend(),
    'tensorflow': TensorFlowBackend(),
    #'cupy': CuPyBackend(),
}

def set_backend(name):
    global _CURRENT_BACKEND
    if name not in _BACKEND_MAP:
        raise ValueError(f"Unknown backend: {name}")
    _CURRENT_BACKEND = name

def get_backend():
    return _CURRENT_BACKEND

def get_backend_instance():
    return _BACKEND_MAP[_CURRENT_BACKEND] 