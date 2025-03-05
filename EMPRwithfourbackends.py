import torch, time
import tensorflow as tf
import numpy as np
from itertools import combinations

class NDEMPRCalculator:

    def __init__(self, G, supports='das', custom_supports=None, backend=None):
        self.backend = backend

        # Error handling for unsupported backends
        if self.backend not in ['tensorflow', 'pytorch', 'numpy']:
            raise ValueError(f"Unsupported backend: {self.backend}. Supported backends are: 'tensorflow', 'pytorch', and 'numpy'.")

        # Convert G to the selected backend's tensor type
        if self.backend == 'tensorflow':
            try:
                G = tf.convert_to_tensor(G, dtype=tf.float64)
            except Exception as e:
                raise ValueError(f"TensorFlow error when converting G: {e}")
        elif self.backend == 'pytorch':
            try:
                if torch.cuda.is_available():
                    G = G.to('cuda')
                G = G.double()
            except Exception as e:
                raise ValueError(f"PyTorch error when converting G: {e}")
        elif self.backend == 'numpy':
            try:
                G = np.array(G, dtype=np.float64)
            except Exception as e:
                raise ValueError(f"NumPy error when converting G: {e}")

        self.G = G
        self.dimensions = G.shape
        self.custom_supports = custom_supports
        self.support_vectors = self.initialize_support_vectors(supports)
        self.weights = [1/dim for dim in self.dimensions]  # Calculate EMPR weights
        self.g0 = self.calculate_g0()
        self.g_components = {}

        # Ensure dimensions are correct before proceeding
        if len(self.dimensions) < 2:
            raise ValueError(f"Tensor G must have at least 2 dimensions, but got {len(self.dimensions)} dimensions.")

        start_time = time.time()
        self.calculate_empr_component(np.arange(len(self.dimensions)))
        self.excution_time = time.time()-start_time

        start_time = time.time()
        self.calculate_approximation(order=len(self.dimensions))
        self.approximation_time = time.time()-start_time

    def initialize_support_vectors(self, supports):
        support_vectors = []
        if supports == 'das':
            for i in range(len(self.dimensions)):
                temp = self.G
                ind = 0
                for j in range(len(self.dimensions)):
                    if j != i:
                        if self.backend == 'tensorflow':
                            temp = tf.reduce_mean(temp, axis=ind)
                        elif self.backend == 'pytorch':
                            temp = torch.mean(temp, ind)
                        elif self.backend == 'numpy':
                            temp = np.mean(temp, axis=ind)
                    else:
                        ind += 1
                if self.backend == 'tensorflow':
                    temp = tf.expand_dims(temp, -1)
                elif self.backend == 'pytorch':
                    temp = torch.unsqueeze(temp, -1)
                elif self.backend == 'numpy':
                    temp = np.expand_dims(temp, axis=-1)
                support_vectors.append(temp)

        elif supports == 'ones':
            for dim_size in self.dimensions:
                if self.backend == 'tensorflow':
                    s = tf.ones((dim_size, 1), dtype=tf.float64)
                elif self.backend == 'pytorch':
                    s = torch.ones(dim_size, 1, dtype=torch.float64)
                elif self.backend == 'numpy':
                    s = np.ones((dim_size, 1), dtype=np.float64)
                support_vectors.append(s)

        elif supports == 'custom':
            if self.custom_supports is None:
                raise ValueError("Custom supports must be provided for 'custom' support type.")
            if len(self.custom_supports) != len(self.dimensions):
                raise ValueError("The number of custom supports must match the number of dimensions.")
            for w in self.custom_supports:
                if self.backend == 'numpy':
                    support_vectors.append(np.array(w, dtype=np.float64))
                elif self.backend == 'tensorflow':
                    support_vectors.append(tf.convert_to_tensor(w, dtype=tf.float64))
                elif self.backend == 'pytorch':
                    support_vectors.append(torch.tensor(w, dtype=torch.float64))

        return support_vectors

    def calculate_g0(self):
        g0 = self.G
        for i, (s, w) in enumerate(zip(self.support_vectors, self.weights)):
            try:
                if self.backend == 'tensorflow':
                    g0 = tf.tensordot(g0, s, axes=[[0], [0]]) * w
                elif self.backend == 'pytorch':
                    g0 = torch.tensordot(g0, s, dims=([0], [0])) * w
                elif self.backend == 'numpy':
                    g0 = np.tensordot(g0, s, axes=([0], [0])) * w
            except Exception as e:
                raise ValueError(f"Error during tensordot operation in calculate_g0: {e}")

        # Ensure that g0 is returned as the correct type depending on the backend
        if self.backend == 'tensorflow':
            return g0.numpy()
        elif self.backend == 'pytorch':
            return g0.item()
        elif self.backend == 'numpy':
            return g0

    def convert_g_to_string(self, dims):
        return 'g_' + ','.join(map(str, list(map(lambda x: x + 1, dims))))

    def check_required_components(self, dims):
        for i in range(1, len(dims)):
            for g_combination in combinations(dims, i):
                component_name = self.convert_g_to_string(g_combination)
                if component_name not in self.g_components.keys():
                    self.calculate_empr_component(g_combination)

    def calculate_empr_component(self, involved_dims):
        """
        This handles the calculation of EMPR components for NumPy, TensorFlow, and PyTorch.
        """
        try:
            self.check_required_components(involved_dims)

            G_component = self.G
            ind = 0
            for j, (s, w) in enumerate(zip(self.support_vectors, self.weights)):
                if j not in involved_dims:
                    try:
                        if self.backend == 'tensorflow':
                            G_component = tf.tensordot(G_component, s, axes=[[ind], [0]]) * w
                        elif self.backend == 'pytorch':
                            G_component = torch.tensordot(G_component, s, dims=([ind], [0])) * w
                        elif self.backend == 'numpy':
                            G_component = np.tensordot(G_component, s, axes=([ind], [0])) * w
                    except Exception as e:
                        raise ValueError(f"Error during tensordot operation in calculate_empr_component for backend {self.backend}: {e}")
                else:
                    ind += 1

            # Subtraction part for each backend
            if self.backend == 'tensorflow':
                subtracted = tf.squeeze(self.support_vectors[involved_dims[0]] * self.g0)
                for i in range(1, len(involved_dims)):
                    subtracted = tf.einsum('...i,jk->...ij', subtracted, self.support_vectors[involved_dims[i]])
            elif self.backend == 'pytorch':
                subtracted = torch.squeeze(self.support_vectors[involved_dims[0]] * self.g0)
                for i in range(1, len(involved_dims)):
                    subtracted = torch.einsum('...i, jk->...ij', subtracted, self.support_vectors[involved_dims[i]])
            elif self.backend == 'numpy':
                subtracted = np.squeeze(self.support_vectors[involved_dims[0]] * self.g0)
                for i in range(1, len(involved_dims)):
                    subtracted = np.einsum('...i,jk->...ij', subtracted, self.support_vectors[involved_dims[i]])

            # Handle higher-order terms
            if len(involved_dims) > 1:
                for i in range(1, len(involved_dims)):
                    for g_combination in combinations(involved_dims, i):
                        s = [x for x in involved_dims if x not in g_combination]
                        component_name = self.convert_g_to_string(g_combination)
                        if component_name not in self.g_components:
                            self.calculate_empr_component(g_combination)
                        term = self.g_components[component_name]
                        for k in range(len(s)):
                            try:
                                if self.backend == 'tensorflow':
                                    term = tf.einsum('...i,jk->...ij', term, self.support_vectors[s[k]])
                                elif self.backend == 'pytorch':
                                    term = torch.einsum('...i, jk->...ij', term, self.support_vectors[s[k]])
                                elif self.backend == 'numpy':
                                    term = np.einsum('...i,jk->...ij', term, self.support_vectors[s[k]])
                            except Exception as e:
                                raise ValueError(f"Error during einsum operation for higher-order term in {self.backend}: {e}")
                        if self.backend == 'tensorflow':
                            subtracted += tf.transpose(term, perm=np.argsort(list(g_combination) + s).tolist())
                        elif self.backend == 'pytorch':
                            subtracted += torch.permute(term, np.argsort(list(g_combination) + s).tolist())
                        elif self.backend == 'numpy':
                            subtracted += np.transpose(term, axes=np.argsort(list(g_combination) + s))

            # Finalize the component calculation
            if self.backend == 'tensorflow':
                G_component = tf.squeeze(G_component)
                subtracted = tf.squeeze(subtracted)
                G_component = tf.identity(G_component) - subtracted
            elif self.backend == 'pytorch':
                G_component = torch.squeeze(G_component)
                subtracted = torch.squeeze(subtracted)
                G_component = G_component.clone() - subtracted
            elif self.backend == 'numpy':
                G_component = np.squeeze(G_component)
                subtracted = np.squeeze(subtracted)
                G_component = np.copy(G_component) - subtracted

            self.g_components[self.convert_g_to_string(involved_dims)] = G_component

        except Exception as e:
            raise RuntimeError(f"Error in calculate_empr_component for {involved_dims}: {e}")


    def calculate_approximation(self, order):
        try:
            involved_dims = np.arange(len(self.dimensions))
            overall_sum = self.support_vectors[involved_dims[0]] * self.g0

            for i in range(1, len(involved_dims)):
                if self.backend == 'tensorflow':
                    overall_sum = tf.einsum('...i,jk->...ij', overall_sum, self.support_vectors[involved_dims[i]])
                elif self.backend == 'pytorch':
                    overall_sum = torch.einsum('...i, jk->...ij', overall_sum, self.support_vectors[involved_dims[i]])
                elif self.backend == 'numpy':
                    overall_sum = np.einsum('...i,jk->...ij', overall_sum, self.support_vectors[involved_dims[i]])

            # Ensure no extra dimensions
            if self.backend == 'tensorflow':
                overall_sum = tf.squeeze(overall_sum)
            elif self.backend == 'pytorch':
                overall_sum = torch.squeeze(overall_sum)
            elif self.backend == 'numpy':
                overall_sum = np.squeeze(overall_sum)

            # Additional terms for higher order approximations
            for i in range(1, order + 1):
                for g_combination in combinations(involved_dims, i):
                    s = [x for x in involved_dims if x not in g_combination]
                    term = self.g_components[self.convert_g_to_string(g_combination)]
                    for k in range(len(s)):
                        if self.backend == 'tensorflow':
                            term = tf.einsum('...i,jk->...ij', term, self.support_vectors[s[k]])
                        elif self.backend == 'pytorch':
                            term = torch.einsum('...i, jk->...ij', term, self.support_vectors[s[k]])
                        elif self.backend == 'numpy':
                            term = np.einsum('...i,jk->...ij', term, self.support_vectors[s[k]])

                    # Ensure correct ordering and dimensionality
                    if self.backend == 'tensorflow':
                        term = tf.transpose(term, perm=np.argsort(list(g_combination) + s).tolist())
                        overall_sum += tf.squeeze(term)
                    elif self.backend == 'pytorch':
                        term = torch.permute(term, np.argsort(list(g_combination) + s).tolist())
                        overall_sum += torch.squeeze(term)
                    elif self.backend == 'numpy':
                        term = np.transpose(term, axes=np.argsort(list(g_combination) + s))
                        overall_sum += np.squeeze(term)

            if self.backend == 'tensorflow':
                return tf.squeeze(overall_sum)
            elif self.backend == 'pytorch':
                return torch.squeeze(overall_sum)
            elif self.backend == 'numpy':
                return np.squeeze(overall_sum)
            
        except Exception as e:
            raise RuntimeError(f"Error in calculate_approximation: {e}")

    def calculate_mse(self, order):
        try:
            T_empr = self.calculate_approximation(order)
            if self.backend == 'tensorflow':
                squared_error = tf.norm(self.G - T_empr, ord='euclidean')
                mse = squared_error / tf.size(self.G, out_type=tf.float64)
                return mse.numpy().item()
            elif self.backend == 'pytorch':
                squared_error = torch.norm(self.G - T_empr, p='fro')
                mse = squared_error / torch.numel(self.G)
                return mse.item()
            elif self.backend == 'numpy':
                squared_error = np.linalg.norm((self.G - T_empr).flatten(), ord=2)
                mse = squared_error / np.size(self.G)
                return mse
        except Exception as e:
            raise RuntimeError(f"Error in calculate_mse: {e}")
"""
import tensorflow as tf
import numpy as np

# Set random seed for reproducibility
tf.random.set_seed(0)

# Create a random tensor using TensorFlow instead of PyTorch
G = tf.random.uniform((3, 4, 5), dtype=tf.float64)

# Initialize the NDEMPRCalculator with the TensorFlow backend
empr_calculator = NDEMPRCalculator(G, supports='das', backend='tensorflow')

# Print the original tensor
print("Original:")
print(G)  # Convert TensorFlow tensor to NumPy for easy printing

# Calculate and print the approximation of the tensor using EMPR
print("Approximation (Order 3):")
approximation = empr_calculator.calculate_approximation(order=4)
print(approximation)  # Convert TensorFlow tensor to NumPy for easy printing

# Calculate the MSE for the approximation of order 3
mse = empr_calculator.calculate_mse(order=3)
print("Mean Squared Error:", mse)

"""

"""
# Example usage for NumPy backend
G = np.random.rand(3, 4, 5)

empr_calculator = NDEMPRCalculator(G, supports='das', backend='numpy')

# Calculate approximation
print("Original:")
print(G)
print("Approximation (Order 3):")
approximation = empr_calculator.calculate_approximation(order=4)
print(approximation)

# Calculate the MSE for the approximation of order 3
mse = empr_calculator.calculate_mse(order=3)
print("Mean Squared Error:", mse)

"""


"""
# Example usage for NumPy backend
torch.manual_seed(0)
G = torch.rand(3, 4, 5)
empr_calculator = NDEMPRCalculator(G, supports='das', backend='pytorch')
print(G)

# Calculate the MSE for the approximation of order 3
mse = empr_calculator.calculate_mse(order=7)
print("Mean Squared Error:", mse)
"""