"""
EMPR on a 2D Tensor
===================

Run EMPR on a small 2D tensor, reconstruct the approximation, and inspect
the extracted component terms.
"""

import numpy as np
from hdmrlib import EMPR

X = np.random.rand(10, 10)

empr = EMPR(X, order=2)
X_reconstructed = empr.reconstruct()
components = empr.components()

print("Input shape:", X.shape)
print("Component keys:", list(components.keys()))