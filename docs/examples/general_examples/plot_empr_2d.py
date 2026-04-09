"""
HDMR on a 2D Tensor
===================

Run HDMR on a small 2D tensor and inspect the resulting component terms.
"""

import matplotlib.pyplot as plt
import numpy as np

from hdmrlib import HDMR


def to_display_array(component):
    """Convert a component to a 2D array for visualization."""
    arr = np.asarray(component)

    if arr.ndim == 0:
        return arr.reshape(1, 1)

    if arr.ndim == 1:
        return arr[:, None]

    if arr.ndim == 2:
        return arr

    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2:
        return arr

    # Fallback for higher-dimensional outputs
    return arr.reshape(arr.shape[0], -1)


x = np.linspace(0.0, 1.0, 32)
y = np.linspace(0.0, 1.0, 32)

X = (
    0.5
    + np.sin(np.pi * x)[:, None]
    + np.cos(np.pi * y)[None, :]
    + 0.25 * np.outer(x, y)
)

hdmr = HDMR(X, order=2)
components = hdmr.components()

print("Input shape:", X.shape)
print("Available component keys:", list(components.keys()))

# Expected keys for a 2D tensor decomposition
component_keys = ["g_1", "g_2", "g_1,2"]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.ravel()

# Original tensor
im0 = axes[0].imshow(X, aspect="auto")
axes[0].set_title("Original tensor")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

# Component terms
for ax, key in zip(axes[1:], component_keys):
    component = to_display_array(components[key])
    im = ax.imshow(component, aspect="auto")
    ax.set_title(key)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()