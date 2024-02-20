import numpy as np

def calculate_g0(G, n1, n2, n3):
    """Calculate the zero-way EMPR component g(0)"""
    g0 = 0
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                g0 += (1/n1) * (1/n2) * (1/n3) * G[i, j, k]  # Using constant weight vectors as per equation (9)
    return g0

def calculate_gi(G, g0, n1, n2, n3):
    """Calculate the one-way EMPR components g(i)"""
    g1 = np.zeros(n1)
    g2 = np.zeros(n2)
    g3 = np.zeros(n3)

    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                g1[i] += (1/n2) * (1/n3) * G[i, j, k] - g0 / n1
                g2[j] += (1/n1) * (1/n3) * G[i, j, k] - g0 / n2
                g3[k] += (1/n1) * (1/n2) * G[i, j, k] - g0 / n3

    return g1, g2, g3

# Sample Data and Dimension
n1, n2, n3 = 10, 10, 10  # Sample dimensions
G = np.random.rand(n1, n2, n3)  # Sample 3D data

# Calculate EMPR Components
g0 = calculate_g0(G, n1, n2, n3)
g1, g2, g3 = calculate_gi(G, g0, n1, n2, n3)

print("Zero-way EMPR component g(0):", g0)
print("One-way EMPR components g(1), g(2), g(3):", g1, g2, g3)
