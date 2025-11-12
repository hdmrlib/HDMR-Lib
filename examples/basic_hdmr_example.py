import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import directly from the root modules  
from hdmr import HDMR
from empr import EMPR
from backends import set_backend

# Create a compatibility object
class hdmr:
    HDMR = HDMR
    EMPR = EMPR
    set_backend = set_backend
import numpy as np

# Attempt to run HDMR on all available backends
backends_to_try = ['numpy', 'torch', 'tensorflow']

# Random test tensor
tensor = np.random.rand(5, 5, 5)

for backend in backends_to_try:
	try:
		hdmr.set_backend(backend)
		print(f"\nBackend: {backend}")
		model = hdmr.HDMR(tensor)
		result = model.decompose(order=3)
		print('HDMR decomposition result shape:', tuple(result.shape))
		# Compute MSE between original tensor and approximation
		def _to_numpy(x):
			try:
				import torch
				if torch.is_tensor(x):
					return x.detach().cpu().numpy()
			except Exception:
				pass
			try:
				import tensorflow as tf
				if isinstance(x, tf.Tensor):
					return x.numpy()
			except Exception:
				pass
			return np.array(x)
		mse = float(np.mean((tensor - _to_numpy(result)) ** 2))
		print('MSE:', mse)
		# For 3D tensors, also show component keys up to order 3
		if tensor.ndim == 3:
			comps = model.components(max_order=3)
			keys = ['g1','g2','g3','g12','g13','g23','g123']
			present = [k for k in keys if k in comps]
			print('Components available:', present)
	except Exception as e:
		print(f"Skipping backend {backend}: {e}") 