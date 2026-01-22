"""
Basic usage example for HDMR-Lib
Demonstrates simple HDMR and EMPR decomposition
"""

import numpy as np
from hdmrlib import HDMR, EMPR, set_backend
from hdmrlib.metrics import mean_squared_error

# Set backend
set_backend('numpy')

# Create a sample 3D tensor
print("=" * 60)
print("HDMR-Lib Basic Usage Example")
print("=" * 60)

tensor = np.random.rand(5, 5, 5)
print(f"\nOriginal tensor shape: {tensor.shape}")

# ============================================
# HDMR Decomposition
# ============================================
print("\n" + "-" * 60)
print("HDMR (High Dimensional Model Representation)")
print("-" * 60)

hdmr_model = HDMR(tensor, weight='avg', supports='ones')
hdmr_result = hdmr_model.decompose(order=2)

print(f"Decomposition shape: {hdmr_result.shape}")
print(f"MSE: {mean_squared_error(tensor, hdmr_result):.6e}")

# Get components
hdmr_components = hdmr_model.components(max_order=2)
print(f"Number of components: {len(hdmr_components)}")
print(f"Component keys: {list(hdmr_components.keys())}")

# ============================================
# EMPR Decomposition
# ============================================
print("\n" + "-" * 60)
print("EMPR (Enhanced Multivariate Products Representation)")
print("-" * 60)

empr_model = EMPR(tensor, supports='das')
empr_result = empr_model.decompose(order=2)

print(f"Decomposition shape: {empr_result.shape}")
print(f"MSE: {mean_squared_error(tensor, empr_result):.6e}")

# Get components
empr_components = empr_model.components(max_order=2)
print(f"Number of components: {len(empr_components)}")
print(f"Component keys: {list(empr_components.keys())}")

print("\n" + "=" * 60)
print("Example completed successfully!")
print("=" * 60)

