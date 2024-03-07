import torch
import numpy as np
from itertools import combinations

class NDEMPRCalculator:
    def __init__(self, G):
        self.G = G.double()
        self.dimensions = G.shape
        self.support_vectors = self.calculate_support_vectors()
        self.weights = [1/dim for dim in self.dimensions]  # Calculate weights
        self.g0 = self.calculate_g0()
        self.g_components = {}

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

    def calculate_gi(self): # Deprecated 

        gi_components = []

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
            gi = temp_G.view(-1,1) - self.g0 * self.support_vectors[i]
            gi_components.append(gi.view(-1, 1))

        return gi_components
    
    def calculate_empr_component(self, involved_dims):

        # Calculation Parts:
        #
        #   g_k = ((G x s_i)... x s_j)                            [FIRST PART]
        #          - g_0 s_1...s_n                                [SECOND PART]
        #          - g_1s_2...s_n - ... - s_1...s_(n-1)g_n - ...  [THIRD PART]
        #    

        def convert_g_to_string(dims):
            return 'g_' + ''.join(map(str, list(map(lambda x:x+1, dims))))
    
        #INITIALIZATIONS
        G_component = self.G
        involved_dims = sorted(involved_dims)

        # FIRST PART
        ind=0
        for j, (s, w) in enumerate(zip(self.support_vectors, self.weights)):
            if j not in involved_dims:
                G_component = torch.tensordot(G_component, s, dims=([ind], [0])) * w
            else:
                ind+=1

        # SECOND PART
        subtracted = torch.squeeze(self.support_vectors[involved_dims[0]] * self.g0)
        for i in range(1, len(involved_dims)):
            subtracted = torch.einsum('...i, jk->...ij', subtracted, 
                                                         self.support_vectors[involved_dims[i]])
        # THIRD PART
        if len(involved_dims) > 1:
            for i in range(1, len(involved_dims)):
                    for g_combination in combinations(involved_dims, i):
                        s = [x for x in involved_dims if x not in g_combination]
                        term = torch.squeeze(self.g_components[convert_g_to_string(g_combination)])

                        for k in range(len(s)):
                            term = torch.einsum('...i, jk->...ij', term, 
                                                                   self.support_vectors[s[k]])
                            
                        subtracted += torch.permute(term, np.argsort(list(g_combination) + s).tolist())

        G_component = torch.squeeze(G_component)
        subtracted = torch.squeeze(subtracted)
        G_component -= subtracted

        self.g_components[convert_g_to_string(involved_dims)] = G_component

# Example usage
G = torch.rand(3, 4, 5, 6, 7)

empr_calculator = NDEMPRCalculator(G)
empr_calculator.calculate_empr_component([0])
empr_calculator.calculate_empr_component([1])
empr_calculator.calculate_empr_component([4])
empr_calculator.calculate_empr_component([0,1])
empr_calculator.calculate_empr_component([0,4])
empr_calculator.calculate_empr_component([1,4])
empr_calculator.calculate_empr_component([0,1,4])

print(empr_calculator.g_components)
