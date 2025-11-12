"""
Backend comparison example for HDMR-Lib
Demonstrates usage across different computational backends
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import time
from hdmr import HDMR
from empr import EMPR
from backends import set_backend
from metrics import mean_squared_error

print("=" * 60)
print("HDMR-Lib Backend Comparison")
print("=" * 60)

# Create test tensor
tensor = np.random.rand(8, 8, 8)
print(f"\nTest tensor shape: {tensor.shape}")

# ============================================
# HDMR Across Available Backends
# ============================================
print("\n" + "-" * 60)
print("HDMR Decomposition (NumPy, PyTorch, TensorFlow)")
print("-" * 60)

hdmr_backends = ['numpy', 'torch', 'tensorflow']

for backend_name in hdmr_backends:
    try:
        set_backend(backend_name)
        
        # Time the decomposition
        start_time = time.time()
        model = HDMR(tensor, weight='avg', supports='ones')
        result = model.decompose(order=2)
        elapsed_time = time.time() - start_time
        
        # Convert result to numpy
        try:
            import torch
            if torch.is_tensor(result):
                result_np = result.detach().cpu().numpy()
            else:
                result_np = np.asarray(result)
        except:
            try:
                import tensorflow as tf
                if isinstance(result, tf.Tensor):
                    result_np = result.numpy()
                else:
                    result_np = np.asarray(result)
            except:
                result_np = np.asarray(result)
        
        mse = np.mean((tensor - result_np) ** 2)
        
        print(f"\n{backend_name:>12}:")
        print(f"  Time: {elapsed_time:.4f}s")
        print(f"  MSE:  {mse:.6e}")
        
    except ValueError as e:
        print(f"\n{backend_name:>12}: Not available ({str(e)})")
    except Exception as e:
        print(f"\n{backend_name:>12}: Error - {type(e).__name__}")

print("\n" + "=" * 60)
print("Backend comparison completed!")
print("=" * 60)

