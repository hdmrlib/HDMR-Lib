_CURRENT_BACKEND = 'numpy'
_BACKEND_MAP = {}

def set_backend(name):
    global _CURRENT_BACKEND
    if name not in ['numpy', 'torch', 'tensorflow', 'cupy']:
        raise ValueError(f"Unknown backend: {name}")
    _CURRENT_BACKEND = name

def get_backend():
    return _CURRENT_BACKEND

def get_backend_instance():
    if _CURRENT_BACKEND not in _BACKEND_MAP:
        if _CURRENT_BACKEND == 'numpy':
            from .numpy_backend import NumPyBackend
            _BACKEND_MAP['numpy'] = NumPyBackend()
        elif _CURRENT_BACKEND == 'torch':
            from .torch_backend import TorchBackend
            _BACKEND_MAP['torch'] = TorchBackend()
        elif _CURRENT_BACKEND == 'tensorflow':
            from .tensorflow_backend import TensorFlowBackend
            _BACKEND_MAP['tensorflow'] = TensorFlowBackend()
        elif _CURRENT_BACKEND == 'cupy':
            from .cupy_backend import CuPyBackend
            _BACKEND_MAP['cupy'] = CuPyBackend()
    return _BACKEND_MAP[_CURRENT_BACKEND] 