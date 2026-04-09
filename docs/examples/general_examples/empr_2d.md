"""
EMPR on a 2D Tensor
===================

Run EMPR on a small 2D tensor, reconstruct the approximation, and inspect
the extracted component terms.
"""

import numpy as np
from hdmrlib import EMPR

# Create a small 2D tensor
X = np.random.rand(10, 10)

# Build the decomposition object
empr = EMPR(X, order=2)

# Reconstruct the tensor from the decomposition
X_reconstructed = empr.reconstruct()

# Get all component terms
components = empr.components()

print("Input shape:", X.shape)
print("Reconstructed shape:", X_reconstructed.shape)
print("Available component keys:", list(components.keys()))