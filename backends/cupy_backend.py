import cupy as cp
import numpy as np
from itertools import combinations
from .base import BaseBackend

class CuPyBackend(BaseBackend):
    # No HDMR implementation for CuPy
    def hdmr_decompose(self, tensor, order=2, **kwargs):
        raise NotImplementedError("HDMR decomposition is not implemented for the CuPy backend.")

    # EMPR implementation
    class _EMPR:
        def __init__(self, G, supports='das', custom_supports=None):
            self.G = cp.array(G, dtype=cp.float64)
            self.dimensions = self.G.shape
            self.custom_supports = custom_supports
            self.support_vectors = self.initialize_support_vectors(supports)
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
                            temp = cp.mean(temp, axis=ind)
                        else:
                            ind += 1
                    temp = cp.expand_dims(temp, -1)
                    support_vectors.append(temp)
            elif supports == 'ones':
                for dim_size in self.dimensions:
                    s = cp.ones((dim_size, 1), dtype=cp.float64)
                    support_vectors.append(s)
            elif supports == 'custom':
                if self.custom_supports is None:
                    raise ValueError("Custom supports must be provided for 'custom' support type.")
                if len(self.custom_supports) != len(self.dimensions):
                    raise ValueError("The number of custom supports must match the number of dimensions.")
                for s in self.custom_supports:
                    if not isinstance(s, cp.ndarray):
                        s = cp.array(s, dtype=cp.float64)
                    support_vectors.append(s)
            return support_vectors

        def calculate_g0(self):
            g0 = self.G
            for i, s in enumerate(self.support_vectors):
                g0 = cp.tensordot(g0, s, axes=([0], [0]))
            return float(cp.asnumpy(g0))

        def convert_g_to_string(self, dims):
            return 'g_' + ','.join(map(str, list(map(lambda x: x+1, dims))))

        def check_required_components(self, dims):
            for i in range(1, len(dims)):
                for g_combination in combinations(dims, i):
                    component_name = self.convert_g_to_string(g_combination)
                    if component_name not in self.g_components.keys():
                        self.calculate_empr_component(g_combination)

        def calculate_empr_component(self, involved_dims):
            self.check_required_components(involved_dims)
            G_component = self.G
            involved_dims = sorted(involved_dims)
            # First part
            ind = 0
            for j, s in enumerate(self.support_vectors):
                if j not in involved_dims:
                    G_component = cp.tensordot(G_component, s, axes=([ind], [0]))
                else:
                    ind += 1
            # Second part
            subtracted = cp.squeeze(self.support_vectors[involved_dims[0]] * self.g0)
            for i in range(1, len(involved_dims)):
                subtracted = cp.einsum('...i,jk->...ij', subtracted, 
                                     self.support_vectors[involved_dims[i]])
            # Third part
            if len(involved_dims) > 1:
                for i in range(1, len(involved_dims)):
                    for g_combination in combinations(involved_dims, i):
                        s = [x for x in involved_dims if x not in g_combination]
                        term = cp.squeeze(self.g_components[self.convert_g_to_string(g_combination)])
                        for k in range(len(s)):
                            term = cp.einsum('...i,jk->...ij', term, 
                                           self.support_vectors[s[k]])
                        subtracted += cp.transpose(term, np.argsort(list(g_combination) + s))
            G_component = cp.squeeze(G_component)
            subtracted = cp.squeeze(subtracted)
            G_component = G_component - subtracted
            self.g_components[self.convert_g_to_string(involved_dims)] = G_component

        def calculate_approximation(self, order):
            involved_dims = np.arange(len(self.dimensions))
            # First part
            overall_sum = cp.squeeze(self.support_vectors[involved_dims[0]] * self.g0)
            for i in range(1, len(involved_dims)):
                overall_sum = cp.einsum('...i,jk->...ij', overall_sum, 
                                      self.support_vectors[involved_dims[i]])
            # Second-N'th part
            for i in range(1, order+1):
                for g_combination in combinations(involved_dims, i):
                    s = [x for x in involved_dims if x not in g_combination]
                    term = cp.squeeze(self.g_components[self.convert_g_to_string(g_combination)])
                    for k in range(len(s)):
                        term = cp.einsum('...i,jk->...ij', term, 
                                       self.support_vectors[s[k]])
                    overall_sum += cp.transpose(term, np.argsort(list(g_combination) + s))
            return cp.squeeze(overall_sum)

        def calculate_mse(self, order):
            T_empr = self.calculate_approximation(order)
            squared_error = cp.linalg.norm(self.G - T_empr)
            num_elements = self.G.size
            mse = squared_error / num_elements
            return float(cp.asnumpy(mse))

    def empr_decompose(self, tensor, order=2, **kwargs):
        model = self._EMPR(tensor, **kwargs)
        return model.calculate_approximation(order) 