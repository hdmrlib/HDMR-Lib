"""
Advanced usage example for HDMR-Lib
Demonstrates different configurations and sensitivity analysis
"""

import numpy as np
from hdmrlib import HDMR, EMPR, set_backend
from hdmrlib.metrics import mean_squared_error, sensitivity_analysis

# Set backend
set_backend('numpy')

print("=" * 60)
print("HDMR-Lib Advanced Usage Example")
print("=" * 60)

# Create a known function for testing
x = np.linspace(0, 1, 6)
y = np.linspace(0, 1, 6)
z = np.linspace(0, 1, 6)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Function: f(x,y,z) = 2*x + 3*y + x*y + 0.5*z
tensor = 2*X + 3*Y + X*Y + 0.5*Z
print(f"\nTest function: f(x,y,z) = 2*x + 3*y + x*y + 0.5*z")
print(f"Tensor shape: {tensor.shape}")

# ============================================
# HDMR with Different Weight Types
# ============================================
print("\n" + "-" * 60)
print("HDMR with Different Weight Types")
print("-" * 60)

weight_types = ['avg', 'gaussian', 'chebyshev']

for weight in weight_types:
    model = HDMR(tensor, weight=weight, supports='ones')
    result = model.reconstruct(order=2)
    mse = mean_squared_error(tensor, result)
    print(f"{weight:>12}: MSE = {mse:.6e}")

# ============================================
# EMPR with Different Support Types
# ============================================
print("\n" + "-" * 60)
print("EMPR with Different Support Types")
print("-" * 60)

support_types = ['ones', 'das']

for supports in support_types:
    model = EMPR(tensor, supports=supports)
    result = model.reconstruct(order=2)
    mse = mean_squared_error(tensor, result)
    print(f"{supports:>12}: MSE = {mse:.6e}")

# ============================================
# Custom Supports Example
# ============================================
print("\n" + "-" * 60)
print("EMPR with Custom Support Vectors")
print("-" * 60)

custom_supports = [
    np.linspace(0, 1, 6).reshape(-1, 1),
    np.linspace(0, 1, 6).reshape(-1, 1),
    np.linspace(0, 1, 6).reshape(-1, 1)
]

model = EMPR(tensor, supports='custom', custom_supports=custom_supports)
result = model.reconstruct(order=2)
mse = mean_squared_error(tensor, result)
print(f"Custom supports MSE: {mse:.6e}")

# ============================================
# Sensitivity Analysis
# ============================================
print("\n" + "-" * 60)
print("Sensitivity Analysis")
print("-" * 60)

model = EMPR(tensor, supports='das')
components = model.components()

# Analyze specific components
print("\nAnalyzing components: g_1, g_2, g_1,2")
sensitivity_analysis(tensor, components, ['g_1', 'g_2', 'g_1,2'])

# Get results as dictionary
result_dict = sensitivity_analysis(tensor, components, ['g_1', 'g_2'], return_dict=True)
print(f"\nProgrammatic access:")
print(f"  g_1 effect: {result_dict['individual_effects']['g_1']:.2f}%")
print(f"  g_2 effect: {result_dict['individual_effects']['g_2']:.2f}%")
print(f"  Combined: {result_dict['combined_effect']:.2f}%")

print("\n" + "=" * 60)
print("Advanced example completed successfully!")
print("=" * 60)
