try:
	from .numpy_backend import NumPyBackend
except Exception:
	NumPyBackend = None

try:
	from .torch_backend import TorchBackend
except Exception:
	TorchBackend = None

try:
	from .tensorflow_backend import TensorFlowBackend
except Exception:
	TensorFlowBackend = None

_BACKEND_MAP = {}
if NumPyBackend is not None:
	_BACKEND_MAP['numpy'] = NumPyBackend()
if TorchBackend is not None:
	_BACKEND_MAP['torch'] = TorchBackend()
if TensorFlowBackend is not None:
	_BACKEND_MAP['tensorflow'] = TensorFlowBackend()

_CURRENT_BACKEND = 'numpy' if 'numpy' in _BACKEND_MAP else (next(iter(_BACKEND_MAP.keys()), None))


def set_backend(name):
	global _CURRENT_BACKEND
	name = (name or "").lower()
	if name == "tf":
		name = "tensorflow"

	if name not in _BACKEND_MAP:
		if name == "torch":
			raise ImportError("Torch backend not available. Install with: pip install 'hdmrlib[torch]'")
		if name == "tensorflow":
			raise ImportError("TensorFlow backend not available. Install with: pip install 'hdmrlib[tensorflow]'")
		raise ValueError(f"Unknown backend: {name}")

	_CURRENT_BACKEND = name


def get_backend():
	return _CURRENT_BACKEND


def get_backend_instance():
	if _CURRENT_BACKEND is None:
		raise RuntimeError("No available backend. Please install a supported backend and try again.")
	return _BACKEND_MAP[_CURRENT_BACKEND] 

def available_backends():
	return list(_BACKEND_MAP.keys())
