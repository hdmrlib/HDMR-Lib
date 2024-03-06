import torch
from itertools import combinations

class NDEMPRCalculator:
    def __init__(self, G):
        self.G = G
        self.dimensions = G.shape
        self.support_vectors = self.calculate_support_vectors()
        self.weights = [1/dim for dim in self.dimensions]  # Calculate weights

    def calculate_support_vectors(self):
        support_vectors = []
        for dim_size in self.dimensions:
            s = torch.ones(dim_size,dtype=torch.float64)
            l2_norm = torch.norm(s, p=2)
            modified_s = (s * (dim_size ** 0.5)) / l2_norm
            support_vectors.append(modified_s.view(-1, 1))
        return support_vectors

    def calculate_g0(self):
        g0 = self.G
        for i, (s, w) in enumerate(zip(self.support_vectors, self.weights)):
            g0 = torch.tensordot(g0, s, dims=([0], [0])) * w
        return g0.item()

    def calculate_gi(self):
        gi_components = []
        g0 = self.calculate_g0() # Assuming g0 reduces to a scalar

        for i in range(len(self.dimensions)):
            ind=0
            temp_G = self.G.clone()
            for j, (s, w) in enumerate(zip(self.support_vectors, self.weights)):
                if j != i:
                    # Integrate over dimensions except the current one, adjusted by corresponding weight
                    temp_G = torch.tensordot(temp_G, s, dims=([ind], [0])) * w
                    print(temp_G.shape)
                else:
                    ind+=1

            # Normalize the gi component
            gi = temp_G.view(-1,1) - g0 * self.support_vectors[i]
            gi_components.append(gi.view(-1, 1))

        return gi_components

# Example usage
G = torch.rand(3, 4, 5)  # Example N-dimensional tensor
#G = tensor_reshaped
empr_calculator = NDEMPRCalculator(G)
g0 = empr_calculator.calculate_g0()
gi = empr_calculator.calculate_gi()
