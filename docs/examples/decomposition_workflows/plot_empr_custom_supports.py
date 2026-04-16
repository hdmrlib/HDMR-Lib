"""
EMPR with Custom Supports
=========================

Run EMPR with user-defined support vectors and inspect how the chosen supports
affect the reconstruction.
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
    + 0.25 * np.outer(x, y)
)

support_x = np.linspace(0.5, 1.5, 32).reshape(-1, 1)
support_y = np.linspace(1.5, 0.5, 32).reshape(-1, 1)

empr = EMPR(
    X,
    order=2,
    supports="custom",
    custom_supports=[support_x, support_y],
)

X_reconstructed = np.asarray(empr.reconstruct(), dtype=np.float64)
abs_error = np.abs(X - X_reconstructed)

mae_by_row = np.mean(abs_error, axis=1)
mae_total = float(np.mean(abs_error))

mid_row = X.shape[0] // 2

print("Support shapes:", support_x.shape, support_y.shape)
print("Available component keys:", list(empr.components().keys()))
print("Mean absolute error:", mae_total)

fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

# Panel 1: custom supports
axes[0].plot(np.arange(support_x.shape[0]), support_x[:, 0], marker="o", label="support_x")
axes[0].plot(np.arange(support_y.shape[0]), support_y[:, 0], marker="s", label="support_y")
axes[0].set_title("Custom support vectors")
axes[0].set_xlabel("Index")
axes[0].set_ylabel("Support value")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Panel 2: one representative slice
axes[1].plot(X[mid_row, :], marker="o", label="Original slice")
axes[1].plot(X_reconstructed[mid_row, :], marker="s", label="Reconstructed slice")
axes[1].set_title(f"Row slice comparison (row={mid_row})")
axes[1].set_xlabel("Column index")
axes[1].set_ylabel("Value")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# Panel 3: error summary
axes[2].plot(mae_by_row, marker="o")
axes[2].set_title("Mean absolute error by row")
axes[2].set_xlabel("Row index")
axes[2].set_ylabel("Mean absolute error")
axes[2].grid(True, alpha=0.3)

plt.show()