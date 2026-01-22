"""
GPU acceleration example for HDMR-Lib
Demonstrates CUDA support with PyTorch and TensorFlow backends
"""

import numpy as np
import time
from hdmrlib import HDMR, EMPR, set_backend
from hdmrlib.metrics import mean_squared_error

print("=" * 60)
print("HDMR-Lib GPU Acceleration Example")
print("=" * 60)

# Create a larger tensor for better GPU performance comparison
tensor_sizes = [(10, 10, 10), (20, 20, 20), (30, 30, 30)]

# ============================================
# PyTorch Backend (CPU vs CUDA)
# ============================================
print("\n" + "-" * 60)
print("PyTorch Backend - CPU vs CUDA")
print("-" * 60)

try:
    import torch
    
    if torch.cuda.is_available():
        print(f"\nCUDA Available: Yes")
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print(f"\nCUDA Available: No (using CPU only)")
    
    set_backend('torch')
    
    for size in tensor_sizes:
        tensor = np.random.rand(*size)
        print(f"\n  Tensor size: {size}")
        
        # Run EMPR decomposition
        start_time = time.time()
        model = EMPR(tensor, supports='das')
        result = model.decompose(order=2)
        elapsed_time = time.time() - start_time
        
        # Convert to numpy for MSE
        if torch.is_tensor(result):
            result_np = result.detach().cpu().numpy()
        else:
            result_np = np.asarray(result)
        
        mse = mean_squared_error(tensor, result_np)
        
        device_used = "CUDA" if torch.cuda.is_available() else "CPU"
        print(f"    Device: {device_used}")
        print(f"    Time: {elapsed_time:.4f}s")
        print(f"    MSE: {mse:.6e}")

except ImportError:
    print("\nPyTorch not available. Install with: pip install torch")
except Exception as e:
    print(f"\nError with PyTorch: {type(e).__name__}: {e}")

# ============================================
# TensorFlow Backend (GPU support)
# ============================================
print("\n" + "-" * 60)
print("TensorFlow Backend - GPU Support")
print("-" * 60)

try:
    import tensorflow as tf
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\nGPUs Available: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
    else:
        print(f"\nGPUs Available: No (using CPU only)")
    
    set_backend('tensorflow')
    
    for size in tensor_sizes:
        tensor = np.random.rand(*size)
        print(f"\n  Tensor size: {size}")
        
        # Run EMPR decomposition
        start_time = time.time()
        model = EMPR(tensor, supports='das')
        result = model.decompose(order=2)
        elapsed_time = time.time() - start_time
        
        # Convert to numpy for MSE
        if isinstance(result, tf.Tensor):
            result_np = result.numpy()
        else:
            result_np = np.asarray(result)
        
        mse = mean_squared_error(tensor, result_np)
        
        device_used = "GPU" if gpus else "CPU"
        print(f"    Device: {device_used}")
        print(f"    Time: {elapsed_time:.4f}s")
        print(f"    MSE: {mse:.6e}")

except ImportError:
    print("\nTensorFlow not available. Install with: pip install tensorflow")
except Exception as e:
    print(f"\nError with TensorFlow: {type(e).__name__}: {e}")

# ============================================
# Performance Comparison
# ============================================
print("\n" + "=" * 60)
print("GPU Acceleration Tips")
print("=" * 60)

print("""
1. PyTorch CUDA Support:
   - Install: pip install torch
   - CUDA will be used automatically if available
   - Check: torch.cuda.is_available()

2. TensorFlow GPU Support:
   - Install: pip install tensorflow (GPU support included)
   - GPUs are used automatically if available
   - Check: tf.config.list_physical_devices('GPU')

3. Performance Tips:
   - GPU acceleration is most beneficial for larger tensors (>100³)
   - For small tensors, CPU might be faster due to overhead
   - Ensure CUDA and cuDNN are properly installed
   - Monitor GPU memory usage with nvidia-smi

4. Installation:
   - CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
   - cuDNN: https://developer.nvidia.com/cudnn
   - Verify: nvidia-smi (check CUDA version)
""")

print("=" * 60)
print("Example completed!")
print("=" * 60)

