"""
Compare Decomposition Orders
============================

Compare reconstruction quality across retained decomposition orders on the same
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

orders = [0, 1, 2]
reconstructions = [np.asarray(empr.reconstruct(order=o), dtype=np.float64) for o in orders]
mae_values = [float(np.mean(np.abs(X - Xr))) for Xr in reconstructions]
rmse_values = [float(np.sqrt(np.mean((X - Xr) ** 2))) for Xr in reconstructions)]

print("Orders:", orders)
print("MAE by order:", mae_values)
print("RMSE by order:", rmse_values)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

axes[0].plot(orders, mae_values, marker="o", label="MAE")
axes[0].plot(orders, rmse_values, marker="s", label="RMSE")
axes[0].set_title("Reconstruction error by retained order")
axes[0].set_xlabel("Retained order")
axes[0].set_ylabel("Error")
axes[0].set_xticks(orders)
axes[0].grid(True, alpha=0.3)
axes[0].legend()

order1_error = np.abs(X - reconstructions[1])
im = axes[1].imshow(order1_error, aspect="auto", interpolation="nearest")
axes[1].set_title("Absolute error for order-1 reconstruction")
fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

plt.show()