"""
Lower-Order Reconstruction
==========================

Compare the full reconstruction with a lower-order reconstruction on the same
tensor.
"""

import matplotlib.pyplot as plt
import numpy as np

from hdmrlib import EMPR

x = np.linspace(0.0, 1.0, 32)
y = np.linspace(0.0, 1.0, 32)

X = (
    0.5
    + np.sin(np.pi * x)[:, None]
    + np.cos(np.pi * y)[None, :]
    + 0.35 * np.outer(x, y)
)

empr = EMPR(X, order=2)

X_full = empr.reconstruct()
X_order1 = empr.reconstruct(order=1)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

im0 = axes[0].imshow(X, aspect="auto")
axes[0].set_title("Original tensor")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(X_full, aspect="auto")
axes[1].set_title("Order-2 reconstruction")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

im2 = axes[2].imshow(X_order1, aspect="auto")
axes[2].set_title("Order-1 reconstruction")
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
