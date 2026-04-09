"""
Inspect Component Terms
=======================

Inspect the available component keys and visualize the magnitude of each
component term.
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

empr = EMPR(X, order=2)
components = empr.components()

keys = list(components.keys())
norms = [float(np.linalg.norm(np.asarray(components[key]))) for key in keys]

print("Available component keys:", keys)

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(keys, norms)
ax.set_title("Component magnitudes")
ax.set_xlabel("Component key")
ax.set_ylabel("Frobenius norm")
plt.tight_layout()
plt.show()
