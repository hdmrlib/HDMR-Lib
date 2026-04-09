"""
EMPR on an RGB Image
====================

Load an RGB image as a 3D tensor, run EMPR, reconstruct the tensor, and compare
the reconstruction with the original image.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hdmrlib import EMPR


image_path = Path("docs/_static/examples/squirell.png")

# Load the image.
X = plt.imread(image_path)

# If the PNG has an alpha channel, drop it.
if X.ndim == 3 and X.shape[2] == 4:
    X = X[..., :3]

# This example is intended for RGB images.
if X.ndim != 3 or X.shape[2] != 3:
    raise ValueError(
        "Expected an RGB image with shape (height, width, 3). "
        f"Got shape {X.shape!r}."
    )

# Convert to float64 for the library.
X = np.asarray(X, dtype=np.float64)

# Optional: reduce image size if it is too large for docs builds.
max_size = 128
if X.shape[0] > max_size or X.shape[1] > max_size:
    step0 = max(1, X.shape[0] // max_size)
    step1 = max(1, X.shape[1] // max_size)
    X = X[::step0, ::step1, :]

empr = EMPR(X, order=2)
X_reconstructed = np.asarray(empr.reconstruct(), dtype=np.float64)
components = empr.components()

# Clip for display.
X_display = np.clip(X, 0.0, 1.0)
X_reconstructed_display = np.clip(X_reconstructed, 0.0, 1.0)

# Mean absolute error across RGB channels.
error_map = np.mean(np.abs(X_display - X_reconstructed_display), axis=2)

print("Image path:", image_path)
print("Input shape:", X.shape)
print("Reconstructed shape:", X_reconstructed.shape)
print("Available component keys:", list(components.keys()))

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(X_display)
axes[0].set_title("Original image")
axes[0].axis("off")

axes[1].imshow(X_reconstructed_display)
axes[1].set_title("EMPR reconstruction")
axes[1].axis("off")

im = axes[2].imshow(error_map)
axes[2].set_title("Mean absolute error")
axes[2].axis("off")
plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
