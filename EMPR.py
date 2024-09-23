
## EXPLAINABILITY - ORDER


import torch
import numpy as np
from itertools import combinations

class NDEMPRCalculator:

    def __init__(self, G, supports = 'das', custom_supports = None):
        
        if torch.cuda.is_available():
            G = G.to('cuda')
        else:
            print('Defaulted to CPU')

        self.G = G.double()
        self.dimensions = G.shape
        self.custom_supports = custom_supports
        self.support_vectors = self.initialize_support_vectors(supports)
        self.weights = [1/dim for dim in self.dimensions]  # Calculate EMPR weights
        self.g0 = self.calculate_g0()
        self.g_components = {}
        self.calculate_empr_component(np.arange(len(self.dimensions)))

    def initialize_support_vectors(self, supports):

        support_vectors = []

        if supports == 'das':

            for i in range(len(self.dimensions)):
                temp = self.G
                ind = 0
                for j in range(len(self.dimensions)):
                    if j != i:
                        temp = torch.mean(temp, ind)
                    else:
                        ind += 1
                temp = torch.unsqueeze(temp, -1)
                support_vectors.append(temp)                

        elif supports == 'ones':
            
            for dim_size in self.dimensions:
                s = torch.ones(dim_size, 1, dtype=torch.float64)
                l2_norm = torch.norm(s, p=2)
                modified_s = (s * (dim_size ** 0.5)) / l2_norm
                support_vectors.append(modified_s)

        elif supports == 'custom':

            if self.custom_supports is None:
                raise ValueError("Custom supports must be provided for 'custom' support type.")
            
            if len(self.custom_supports) != len(self.dimensions):
                raise ValueError("The number of custom supports must match the number of dimensions.")
            
            for w in self.custom_supports:
                if not torch.is_tensor(w):
                    s = torch.tensor(s, dtype=torch.float64)
                support_vectors.append(s)

        return support_vectors

    def calculate_g0(self):

        g0 = self.G
        for i, (s, w) in enumerate(zip(self.support_vectors, self.weights)):
            g0 = torch.tensordot(g0, s, dims=([0], [0])) * w
        return g0.item()
    
    def convert_g_to_string(self, dims):
        return 'g_' + ','.join(map(str, list(map(lambda x:x+1, dims))))
    
    def check_required_components(self, dims):

        # Check if all required g_components are calculated
        for i in range(1, len(dims)):
            for g_combination in combinations(dims, i):

                component_name = self.convert_g_to_string(g_combination)
                if component_name not in self.g_components.keys():

                    #print("\033[33mWarning,", component_name, "not found in previously calculated components...\033[0m")
                    #print("Adding", component_name, "to the queue")

                    self.calculate_empr_component(g_combination)

    def calculate_empr_component(self, involved_dims):

        # Calculation Parts:
        #
        #   g_k = ((G x s_i)... x s_j)                            [FIRST PART]
        #          - g_0 s_1...s_n                                [SECOND PART]
        #          - g_1s_2...s_n - ... - s_1...s_(n-1)g_n - ...  [THIRD PART]
        #    
    
        # CHECKING REQUIREMENTS
        self.check_required_components(involved_dims)

        # INITIALIZATIONS
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
                        term = torch.squeeze(self.g_components[self.convert_g_to_string(g_combination)])

                        for k in range(len(s)):
                            term = torch.einsum('...i, jk->...ij', term, 
                                                                   self.support_vectors[s[k]])
                            
                        subtracted += torch.permute(term, np.argsort(list(g_combination) + s).tolist())

        G_component = torch.squeeze(G_component)
        subtracted = torch.squeeze(subtracted)
        G_component = G_component.clone() - subtracted

        self.g_components[self.convert_g_to_string(involved_dims)] = G_component

    def calculate_approximation(self, order):

        # Calculation Parts:
        #
        #        = g_0 s_1...s_n +                                [FIRST PART]
        #          + g_1s_2...s_n + ... + s_1...s_(n-1)g_n +      [SECOND PART]
        #          + ... +
        #          + g_1...(n-1)s_n + ... + s_1g_2...n            [N'TH PART]
        #

        involved_dims = np.arange(len(self.dimensions))

        # FIRST PART
        overall_sum = torch.squeeze(self.support_vectors[involved_dims[0]] * self.g0)
        for i in range(1, len(involved_dims)):
            overall_sum = torch.einsum('...i, jk->...ij', overall_sum, 
                                                          self.support_vectors[involved_dims[i]])
            
        # SECOND-N'TH PART
        for i in range(1, order+1):
            for g_combination in combinations(involved_dims, i):
                s = [x for x in involved_dims if x not in g_combination]
                term = torch.squeeze(self.g_components[self.convert_g_to_string(g_combination)])

                for k in range(len(s)):
                    term = torch.einsum('...i, jk->...ij', term, 
                                                           self.support_vectors[s[k]])
                    
                overall_sum += torch.permute(term, np.argsort(list(g_combination) + s).tolist())

        return torch.squeeze(overall_sum)

    def calculate_mse(self, order):
        """
        Calculate the Mean Squared Error (MSE) between the original tensor and 
        the approximated tensor based on the given EMPR order.

        Args:
        order: int, the order of approximation.

        Returns:
        mse: float, the mean squared error.
        """
        # Calculate the approximation using the given order
        T_empr = self.calculate_approximation(order)

        # Compute the squared error
        squared_error = torch.norm(self.G - T_empr, p='fro')

        # Calculate MSE by dividing the squared error by the number of elements in the tensor
        num_elements = torch.numel(self.G)
        mse = squared_error / num_elements


        return mse.item()  # Return the MSE as a Python float

    def calculate_sensitivity_ratios(self, order, plot=False):
    
            G_component = self.G
            T_empr = self.calculate_approximation(order)
            ratios = {}
            # Compute g0 and add its ratio to the dictionary
            g0_square = hdmr_calculator.g0 ** 2
            original_norm_square = torch.norm(G_component) ** 2
            ratios['g_0'] = (g0_square / original_norm_square) * 100 
    
            for i in range(1, order + 1):
                for g_combination in combinations(np.arange(len(hdmr_calculator.dimensions)), i):
                    component_name = hdmr_calculator.convert_g_to_string(g_combination)
                    g_component = hdmr_calculator.g_components[component_name]
                    # Calculate the ratio for the current component
                    component_norm_square = torch.norm(g_component) ** 2
                    ratios[component_name] = (component_norm_square / original_norm_square) * 100  # Yüzde olarak hesapla
                    # Numpy array of tensor values
                    values_array = np.array([(component_name, value.item()) for component_name, value in ratios.items()])
                    # Total percenatage of all components
                    total_percentage = np.sum([value.item() for value in ratio.values()])
                    
            print("Sensivitiy Percentages:\n", values_array)
            print("Total percentage: %",total_percentage)
            
            # Plot if it is True
            if plot:
                component_names = list(ratios.keys())
                component_values = list(ratios.values())
                plt.figure(figsize=(10, 6))
                plt.bar(component_names, component_values, color='skyblue')
                for i, value in enumerate(component_values):
                    plt.text(i, value, f'{value:.2f}%', ha='center', va='bottom')  
                plt.xlabel('Components')
                plt.ylabel('Sensitivity Percentages (%)')
                plt.title('Component Sensitivity Percentages')
                plt.xticks(rotation=45)  
                plt.ylim(0, max(component_values)+10)  
                plt.grid(axis='y')  
                plt.show() 
    

# Example usage
torch.manual_seed(0)
G = torch.rand(3, 4, 5)

empr_calculator = NDEMPRCalculator(G, supports = 'das')

# ADD CUSTOM COMPONENTS []
#empr_calculator = NDEMPRCalculator(G, supports = 'custom', custom_supports = None)

'''
for key in empr_calculator.g_components.keys():
    print(key, " - shape :", empr_calculator.g_components[key].shape, "\n\t", empr_calculator.g_components[key])
'''

print("Original:")
print(G)
print("Approx:")
print(empr_calculator.calculate_approximation(order=3))
