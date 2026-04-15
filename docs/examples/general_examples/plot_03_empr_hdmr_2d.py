"""
EMPR vs HDMR on a 2D Tensor
===========================

Run EMPR and HDMR on the same 2D tensor, compare the reconstructions,
and visualize the remaining approximation errors.
"""

import matplotlib.pyplot as plt
import numpy as np

from hdmrlib import EMPR, HDMR


x = np.linspace(0.0, 1.0, 32)
y = np.linspace(0.0, 1.0, 32)

X = (
    0.5
    + np.sin(np.pi * x)[:, None]
    + 0.8 * np.cos(np.pi * y)[None, :]
    + 0.5 * np.outer(x, y)
    + 0.3 * np.exp(-40.0 * ((x[:, None] - 0.7) ** 2 + (y[None, :] - 0.3) ** 2))
)

empr = EMPR(X, order=2)
X_empr = np.asarray(empr.reconstruct(), dtype=np.float64)

hdmr = HDMR(X, order=2)
X_hdmr = np.asarray(hdmr.reconstruct(), dtype=np.float64)

empr_abs_err = np.abs(X - X_empr)
hdmr_abs_err = np.abs(X - X_hdmr)
recon_diff = np.abs(X_empr - X_hdmr)

print("Input shape:", X.shape)
print("EMPR mean absolute error:", float(np.mean(empr_abs_err)))
print("HDMR mean absolute error:", float(np.mean(hdmr_abs_err)))
print("Mean absolute difference between reconstructions:", float(np.mean(recon_diff)))

vmin = min(X.min(), X_empr.min(), X_hdmr.min())
vmax = max(X.max(), X_empr.max(), X_hdmr.max())

err_vmax = max(empr_abs_err.max(), hdmr_abs_err.max(), recon_diff.max())

fig = plt.figure(figsize=(12, 7), constrained_layout=True)
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.05])

ax00 = fig.add_subplot(gs[0, 0])
ax01 = fig.add_subplot(gs[0, 1])
ax02 = fig.add_subplot(gs[0, 2])
cax0 = fig.add_subplot(gs[0, 3])

ax10 = fig.add_subplot(gs[1, 0])
ax11 = fig.add_subplot(gs[1, 1])
ax12 = fig.add_subplot(gs[1, 2])
cax1 = fig.add_subplot(gs[1, 3])

im0 = ax00.imshow(X, aspect="auto", vmin=vmin, vmax=vmax, interpolation="nearest")
ax00.set_title("Original tensor")

ax01.imshow(X_empr, aspect="auto", vmin=vmin, vmax=vmax, interpolation="nearest")
ax01.set_title("EMPR reconstruction")

ax02.imshow(X_hdmr, aspect="auto", vmin=vmin, vmax=vmax, interpolation="nearest")
ax02.set_title("HDMR reconstruction")

fig.colorbar(im0, cax=cax0)

im1 = ax10.imshow(empr_abs_err, aspect="auto", vmin=0.0, vmax=err_vmax, interpolation="nearest")
ax10.set_title("EMPR absolute error")

ax11.imshow(hdmr_abs_err, aspect="auto", vmin=0.0, vmax=err_vmax, interpolation="nearest")
ax11.set_title("HDMR absolute error")

ax12.imshow(recon_diff, aspect="auto", vmin=0.0, vmax=err_vmax, interpolation="nearest")
ax12.set_title("Absolute difference")

fig.colorbar(im1, cax=cax1)

plt.show()