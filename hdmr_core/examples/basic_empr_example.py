import numpy as np
import hdmr_core as hdmr

# Create a sample tensor
tensor = np.random.rand(3, 3, 3)

# Set backend to numpy
hdmr.set_backend("numpy")

# Create EMPR model
model = hdmr.EMPR(tensor, supports="das")

# Decompose with order 2
result = model.decompose(order=2)

print(f"Original tensor shape: {tensor.shape}")
print(f"Decomposed result shape: {result.shape}")
print(f"Result type: {type(result)}") 