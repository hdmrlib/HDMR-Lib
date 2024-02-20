import torch

def calculate_g0_torch(G, n1, n2, n3):
    weights = torch.tensor([1/n1, 1/n2, 1/n3])
    return torch.tensordot(G, weights, dims=([0, 1, 2], [0, 1, 2]))

def calculate_gi_torch(G, g0, n1, n2, n3):
    weights1 = torch.tensor([1/n2, 1/n3])
    weights2 = torch.tensor([1/n1, 1/n3])
    weights3 = torch.tensor([1/n1, 1/n2])

    g1 = torch.tensordot(G, weights1, dims=([1, 2], [0, 1])) - g0 / n1
    g2 = torch.tensordot(G, weights2, dims=([0, 2], [0, 1])) - g0 / n2
    g3 = torch.tensordot(G, weights3, dims=([0, 1], [0, 1])) - g0 / n3

    return g1, g2, g3

# Example usage with sample data
G_torch = torch.rand(n1, n2, n3)
g0_torch = calculate_g0_torch(G_torch, n1, n2, n3)
g1_torch, g2_torch, g3_torch = calculate_gi_torch(G_torch, g0_torch, n1, n2, n3)

def calculate_gij_torch(G, g0, g1, g2, g3, n1, n2, n3):
    # Calculate g(12), g(13), and g(23)
    weight3 = torch.ones(n3) / n3
    g12 = torch.tensordot(G, weight3, dims=([2], [0])) - torch.outer(g1, torch.ones(n2)) - torch.outer(torch.ones(n1), g2) + g0

    weight2 = torch.ones(n2) / n2
    g13 = torch.tensordot(G, weight2, dims=([1], [0])) - torch.outer(g1, torch.ones(n3)) - torch.outer(torch.ones(n1), g3) + g0

    weight1 = torch.ones(n1) / n1
    g23 = torch.tensordot(G, weight1, dims=([0], [0])) - torch.outer(g2, torch.ones(n3)) - torch.outer(torch.ones(n2), g3) + g0

    return g12, g13, g23

# Calculate two-way components
g12_torch, g13_torch, g23_torch = calculate_gij_torch(G_torch, g0_torch, g1_torch, g2_torch, g3_torch, n1, n2, n3)


import torch

def calculate_gij_torch(G, g0, g1, g2, g3):
    n1, n2, n3 = G.shape

    # Convert g0, g1, g2, g3 to tensors if they are not already
    g0 = torch.tensor(g0) if not isinstance(g0, torch.Tensor) else g0
    g1 = torch.tensor(g1) if not isinstance(g1, torch.Tensor) else g1
    g2 = torch.tensor(g2) if not isinstance(g2, torch.Tensor) else g2
    g3 = torch.tensor(g3) if not isinstance(g3, torch.Tensor) else g3

    # Weights for averaging over each dimension
    weight1 = torch.ones(n1) / n1
    weight2 = torch.ones(n2) / n2
    weight3 = torch.ones(n3) / n3

    # Calculate g(12)
    g12 = torch.tensordot(G, weight3, dims=([2], [0])) - torch.outer(g1, torch.ones(n2)) - torch.outer(torch.ones(n1), g2) + g0

    # Calculate g(13)
    g13 = torch.tensordot(G, weight2, dims=([1], [0])) - torch.outer(g1, torch.ones(n3)) - torch.outer(torch.ones(n1), g3) + g0

    # Calculate g(23)
    g23 = torch.tensordot(G, weight1, dims=([0], [0])) - torch.outer(g2, torch.ones(n3)) - torch.outer(torch.ones(n2), g3) + g0

    return g12, g13, g23

# Example usage with sample data
n1, n2, n3 = 10, 10, 10
G_torch = torch.rand(n1, n2, n3)

# Assuming g0_torch, g1_torch, g2_torch, g3_torch are already calculated
g12_torch, g13_torch, g23_torch = calculate_gij_torch(G_torch, g0_torch, g1_torch, g2_torch, g3_torch)

print("Two-way EMPR components g(12), g(13), g(23):", g12_torch, g13_torch, g23_torch)

def calculate_gij_with_support(G, g0, s1, s2, s3, n1, n2, n3):
    # Initialize gij matrices
    g12 = np.zeros((n1, n2))
    g13 = np.zeros((n1, n3))
    g23 = np.zeros((n2, n3))

    # Calculate g12
    for i in range(n1):
        for j in range(n2):
            g12[i, j] = sum(G[i, j, k] * s3[k] for k in range(n3)) - g0 * s1[i] * s2[j]

    # Calculate g13
    for i in range(n1):
        for k in range(n3):
            g13[i, k] = sum(G[i, j, k] * s2[j] for j in range(n2)) - g0 * s1[i] * s3[k]

    # Calculate g23
    for j in range(n2):
        for k in range(n3):
            g23[j, k] = sum(G[i, j, k] * s1[i] for i in range(n1)) - g0 * s2[j] * s3[k]

    return g12, g13, g23

# Example usage
# Assuming s1, s2, s3 are the support vectors for each dimension
# and G is the 3D data cube
g12, g13, g23 = calculate_gij_with_support(G, g0, s1, s2, s3, n1, n2, n3)
