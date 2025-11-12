# HDMR-Lib Examples

This directory contains example scripts demonstrating the usage of HDMR-Lib.

## Examples

### 1. `basic_usage.py`
**Introduction to HDMR and EMPR**

Demonstrates:
- Basic HDMR decomposition with different weight types
- Basic EMPR decomposition with support vectors
- Component extraction
- MSE calculation

**Run:**
```bash
python examples/basic_usage.py
```

### 2. `advanced_usage.py`
**Advanced Features and Configurations**

Demonstrates:
- Different weight types for HDMR (avg, gaussian, chebyshev)
- Different support types for EMPR (ones, das)
- Custom support vectors
- Sensitivity analysis with metrics module
- Working with known analytical functions

**Run:**
```bash
python examples/advanced_usage.py
```

### 3. `backend_comparison.py`
**Multi-Backend Support**

Demonstrates:
- Running decompositions across all backends (NumPy, PyTorch, TensorFlow)
- Performance timing comparison
- Backend-specific tensor handling
- HDMR and EMPR support on all backends

**Run:**
```bash
python examples/backend_comparison.py
```

### 4. `gpu_acceleration.py`
**GPU/CUDA Support**

Demonstrates:
- CUDA acceleration with PyTorch backend
- GPU support with TensorFlow backend
- Performance comparison between CPU and GPU
- Device detection and configuration
- Tips for optimal GPU usage

**Run:**
```bash
python examples/gpu_acceleration.py
```

## Requirements

Basic examples require only NumPy:
```bash
pip install numpy
```

For all backends (CPU):
```bash
pip install numpy torch tensorflow
```

For GPU acceleration:
```bash
# PyTorch with CUDA (choose your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
# or
pip install torch --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1

# TensorFlow (GPU support included by default)
pip install tensorflow
```

**Note:** GPU acceleration requires NVIDIA GPU with CUDA support. Check https://pytorch.org/ for PyTorch installation options.

## Quick Start

Start with `basic_usage.py` to learn the fundamentals, then explore the other examples based on your needs.

