# Copyright (c) 2025 HDMRLib Contributors
# SPDX-License-Identifier: MIT

"""
Basic usage example for HDMRLib
Demonstrates simple HDMR and EMPR decomposition
"""

import numpy as np
from hdmrlib import HDMR, EMPR, set_backend
from hdmrlib.metrics import mean_squared_error

# Set backend
set_backend('numpy')

# Create a sample 3D tensor
print("=" * 60)
print("HDMRLib Basic Usage Example")
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
hdmr_result = hdmr_model.reconstruct(order=2)

print(f"Reconstruction shape: {hdmr_result.shape}")
print(f"MSE: {mean_squared_error(tensor, hdmr_result):.6e}")

# Get components
hdmr_components = hdmr_model.components()
print(f"Number of components: {len(hdmr_components)}")
print(f"Component keys: {list(hdmr_components.keys())}")

# ============================================
# EMPR Decomposition
# ============================================
print("\n" + "-" * 60)
print("EMPR (Enhanced Multivariate Products Representation)")
print("-" * 60)

empr_model = EMPR(tensor, supports='das')
empr_result = empr_model.reconstruct(order=2)

print(f"Reconstruction shape: {empr_result.shape}")
print(f"MSE: {mean_squared_error(tensor, empr_result):.6e}")

# Get components
empr_components = empr_model.components()
print(f"Number of components: {len(empr_components)}")
print(f"Component keys: {list(empr_components.keys())}")

print("\n" + "=" * 60)
print("Example completed successfully!")
print("=" * 60)
