import numpy as np
from itertools import combinations
from .base import BaseBackend

class NumPyBackend(BaseBackend):
    # HDMR implementation
    class _HDMR:
        def __init__(self, G, weight="avg", custom_weights=None, supports='ones', custom_supports=None):
            self.G = np.array(G, dtype=np.float64)
            self.dimensions = G.shape
            self.custom_supports = custom_supports
            self.support_vectors = self.initialize_support_vectors(supports)
            self.custom_weights = custom_weights
            self.weights = self.initialize_weights(weight)
            self.g0 = self.calculate_g0()
            self.g_components = {}
            self.calculate_hdmr_component(np.arange(len(self.dimensions)))

        def initialize_weights(self, weight):
            weights = []
            if weight == 'average' or weight == 'avg':
                for dim_size in self.dimensions:
                    w = np.ones((dim_size, 1), dtype=np.float64)
                    l2_norm = np.linalg.norm(w, ord=2)
                    modified_w = (w * (dim_size ** 0.5)) / l2_norm
                    weights.append(modified_w / dim_size)
            elif weight == 'custom':
                if self.custom_weights is None:
                    raise ValueError("Custom weights must be provided for 'custom' weight type.")
                if len(self.custom_weights) != len(self.dimensions):
                    raise ValueError("The number of custom weights must match the number of dimensions.")
                for w in self.custom_weights:
                    if not isinstance(w, np.ndarray):
                        w = np.array(w, dtype=np.float64)
                    weights.append(w)
            elif weight == 'gaussian':
                for dim_size in self.dimensions:
                    w = np.random.randn(dim_size, 1).astype(np.float64)
                    l2_norm = np.linalg.norm(w, ord=2)
                    modified_w = (w * (dim_size ** 0.5)) / l2_norm
                    weights.append(modified_w / dim_size)
            elif weight == 'chebyshev':
                for dim_size in self.dimensions:
                    k = np.arange(1, dim_size + 1, dtype=np.float64)
                    w = np.cos((2 * k - 1) * np.pi / (2 * dim_size))
                    w = w.reshape(dim_size, 1)
                    l2_norm = np.linalg.norm(w, ord=2)
                    modified_w = (w * (dim_size ** 0.5)) / l2_norm
                    weights.append(modified_w / dim_size)
            return weights

        def initialize_support_vectors(self, supports):
            support_vectors = []
            if supports == 'das':
                for i in range(len(self.dimensions)):
                    temp = self.G
                    ind = 0
                    for j in range(len(self.dimensions)):
                        if j != i:
                            temp = np.mean(temp, axis=ind)
                        else:
                            ind += 1
                    temp = np.expand_dims(temp, axis=-1)
                    support_vectors.append(temp)
            elif supports == 'ones':
                for dim_size in self.dimensions:
                    s = np.ones((dim_size, 1), dtype=np.float64)
                    l2_norm = np.linalg.norm(s, ord=2)
                    modified_s = (s * (dim_size ** 0.5)) / l2_norm
                    support_vectors.append(modified_s)
            elif supports == 'custom':
                if self.custom_supports is None:
                    raise ValueError("Custom supports must be provided for 'custom' support type.")
                if len(self.custom_supports) != len(self.dimensions):
                    raise ValueError("The number of custom supports must match the number of dimensions.")
                for s in self.custom_supports:
                    if not isinstance(s, np.ndarray):
                        s = np.array(s, dtype=np.float64)
                    support_vectors.append(s)
            return support_vectors

        def calculate_g0(self):
            g0 = self.G
            for i, (s, w) in enumerate(zip(self.support_vectors, self.weights)):
                g0 = np.tensordot(g0, s * w, axes=([0], [0]))
            return float(g0)

        def convert_g_to_string(self, dims):
            return 'g_' + ','.join(map(str, list(map(lambda x: x+1, dims))))

        def check_required_components(self, dims):
            for i in range(1, len(dims)):
                for g_combination in combinations(dims, i):
                    component_name = self.convert_g_to_string(g_combination)
                    if component_name not in self.g_components.keys():
                        self.calculate_hdmr_component(g_combination)

        def calculate_hdmr_component(self, involved_dims):
            self.check_required_components(involved_dims)
            G_component = self.G
            involved_dims = sorted(involved_dims)
            # First part
            ind = 0
            for j, (s, w) in enumerate(zip(self.support_vectors, self.weights)):
                if j not in involved_dims:
                    G_component = np.tensordot(G_component, s * w, axes=([ind], [0]))
                else:
                    ind += 1
            # Second part
            subtracted = np.squeeze((self.support_vectors[involved_dims[0]] * self.weights[involved_dims[0]]) * self.g0)
            for i in range(1, len(involved_dims)):
                subtracted = np.einsum('...i,jk->...ij', subtracted, 
                                     (self.support_vectors[involved_dims[i]] * self.weights[involved_dims[i]]))
            # Third part
            if len(involved_dims) > 1:
                for i in range(1, len(involved_dims)):
                    for g_combination in combinations(involved_dims, i):
                        s = [x for x in involved_dims if x not in g_combination]
                        term = np.squeeze(self.g_components[self.convert_g_to_string(g_combination)])
                        for k in range(len(s)):
                            term = np.einsum('...i,jk->...ij', term, 
                                           (self.support_vectors[s[k]] * self.weights[s[k]]))
                        subtracted += np.transpose(term, axes=np.argsort(list(g_combination) + s))
            G_component = np.squeeze(G_component)
            subtracted = np.squeeze(subtracted)
            G_component = np.copy(G_component) - subtracted
            self.g_components[self.convert_g_to_string(involved_dims)] = G_component

        def calculate_approximation(self, order):
            involved_dims = np.arange(len(self.dimensions))
            # First part
            overall_sum = np.squeeze((self.support_vectors[involved_dims[0]] * self.weights[involved_dims[0]]) * self.g0)
            for i in range(1, len(involved_dims)):
                overall_sum = np.einsum('...i,jk->...ij', overall_sum, 
                                      (self.support_vectors[involved_dims[i]] * self.weights[involved_dims[i]]))
            # Second-N'th part
            for i in range(1, order+1):
                for g_combination in combinations(involved_dims, i):
                    s = [x for x in involved_dims if x not in g_combination]
                    term = np.squeeze(self.g_components[self.convert_g_to_string(g_combination)])
                    for k in range(len(s)):
                        term = np.einsum('...i,jk->...ij', term, 
                                       (self.support_vectors[s[k]] * self.weights[s[k]]))
                    overall_sum += np.transpose(term, axes=np.argsort(list(g_combination) + s))
            return np.squeeze(overall_sum)

        def calculate_mse(self, order):
            T_empr = self.calculate_approximation(order)
            squared_error = np.linalg.norm(self.G - T_empr, ord='fro')
            num_elements = self.G.size
            mse = squared_error / num_elements
            return float(mse)

    # EMPR implementation
    class _EMPR:
        def __init__(self, G, supports='das', custom_supports=None):
            self.G = np.array(G, dtype=np.float64)
            self.dimensions = G.shape
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
                            temp = np.mean(temp, axis=ind)
                        else:
                            ind += 1
                    temp = np.expand_dims(temp, -1)
                    support_vectors.append(temp)
            elif supports == 'ones':
                for dim_size in self.dimensions:
                    s = np.ones((dim_size, 1), dtype=np.float64)
                    support_vectors.append(s)
            elif supports == 'custom':
                if self.custom_supports is None:
                    raise ValueError("Custom supports must be provided for 'custom' support type.")
                if len(self.custom_supports) != len(self.dimensions):
                    raise ValueError("The number of custom supports must match the number of dimensions.")
                for s in self.custom_supports:
                    if not isinstance(s, np.ndarray):
                        s = np.array(s, dtype=np.float64)
                    support_vectors.append(s)
            return support_vectors

        def calculate_g0(self):
            g0 = self.G
            for i, s in enumerate(self.support_vectors):
                g0 = np.tensordot(g0, s, axes=([0], [0]))
            return float(g0)

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
                    G_component = np.tensordot(G_component, s, axes=([ind], [0]))
                else:
                    ind += 1
            # Second part
            subtracted = np.squeeze(self.support_vectors[involved_dims[0]] * self.g0)
            for i in range(1, len(involved_dims)):
                subtracted = np.einsum('...i,jk->...ij', subtracted, 
                                     self.support_vectors[involved_dims[i]])
            # Third part
            if len(involved_dims) > 1:
                for i in range(1, len(involved_dims)):
                    for g_combination in combinations(involved_dims, i):
                        s = [x for x in involved_dims if x not in g_combination]
                        term = np.squeeze(self.g_components[self.convert_g_to_string(g_combination)])
                        for k in range(len(s)):
                            term = np.einsum('...i,jk->...ij', term, 
                                           self.support_vectors[s[k]])
                        subtracted += np.transpose(term, np.argsort(list(g_combination) + s))
            G_component = np.squeeze(G_component)
            subtracted = np.squeeze(subtracted)
            G_component = G_component - subtracted
            self.g_components[self.convert_g_to_string(involved_dims)] = G_component

        def calculate_approximation(self, order):
            involved_dims = np.arange(len(self.dimensions))
            # First part
            overall_sum = np.squeeze(self.support_vectors[involved_dims[0]] * self.g0)
            for i in range(1, len(involved_dims)):
                overall_sum = np.einsum('...i,jk->...ij', overall_sum, 
                                      self.support_vectors[involved_dims[i]])
            # Second-N'th part
            for i in range(1, order+1):
                for g_combination in combinations(involved_dims, i):
                    s = [x for x in involved_dims if x not in g_combination]
                    term = np.squeeze(self.g_components[self.convert_g_to_string(g_combination)])
                    for k in range(len(s)):
                        term = np.einsum('...i,jk->...ij', term, 
                                       self.support_vectors[s[k]])
                    overall_sum += np.transpose(term, np.argsort(list(g_combination) + s))
            return np.squeeze(overall_sum)

        def calculate_mse(self, order):
            T_empr = self.calculate_approximation(order)
            squared_error = np.linalg.norm(self.G - T_empr)
            num_elements = self.G.size
            mse = squared_error / num_elements
            return float(mse)

    def hdmr_decompose(self, tensor, order=2, **kwargs):
        model = self._HDMR(tensor, **kwargs)
        return model.calculate_approximation(order)

    def empr_decompose(self, tensor, order=2, **kwargs):
        model = self._EMPR(tensor, **kwargs)
        return model.calculate_approximation(order)

    def hdmr_components(self, tensor, max_order=None, **kwargs):
        model = self._HDMR(tensor, **kwargs)
        num_dims = len(model.dimensions)
        if max_order is None:
            max_order = num_dims
        components = {}
        dims = list(range(num_dims))
        for r in range(1, min(max_order, num_dims) + 1):
            for comb in combinations(dims, r):
                key = 'g' + ''.join(str(i+1) for i in comb)
                components[key] = model.g_components[model.convert_g_to_string(comb)]
        return components

    def empr_components(self, tensor, max_order=None, **kwargs):
        model = self._EMPR(tensor, **kwargs)
        num_dims = len(model.dimensions)
        if max_order is None:
            max_order = num_dims
        components = {}
        dims = list(range(num_dims))
        for r in range(1, min(max_order, num_dims) + 1):
            for comb in combinations(dims, r):
                key = 'g' + ''.join(str(i+1) for i in comb)
                components[key] = model.g_components[model.convert_g_to_string(comb)]
        return components 