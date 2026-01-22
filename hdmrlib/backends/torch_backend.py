import torch
import numpy as np
from itertools import combinations
from .base import BaseBackend

class TorchBackend(BaseBackend):
    def __init__(self):
        """Initialize TorchBackend with device selection."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # HDMR implementation
    class _HDMR:
        def __init__(self, G, weight="avg", custom_weights=None, supports='ones', custom_supports=None, device='cpu'):
            self.device = device
            G = torch.tensor(G, dtype=torch.float64).to(device)
            self.G = G
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
                    w = torch.ones(dim_size, 1, dtype=torch.float64, device=self.device)
                    l2_norm = torch.norm(w, p=2)
                    modified_w = (w * (dim_size ** 0.5)) / l2_norm
                    weights.append(modified_w / dim_size)
            elif weight == 'custom':
                if self.custom_weights is None:
                    raise ValueError("Custom weights must be provided for 'custom' weight type.")
                if len(self.custom_weights) != len(self.dimensions):
                    raise ValueError("The number of custom weights must match the number of dimensions.")
                for w in self.custom_weights:
                    if not torch.is_tensor(w):
                        w = torch.tensor(w, dtype=torch.float64, device=self.device)
                    else:
                        w = w.to(self.device)
                    weights.append(w)
            elif weight == 'gaussian':
                for dim_size in self.dimensions:
                    w = torch.randn(dim_size, 1, dtype=torch.float64, device=self.device)
                    l2_norm = torch.norm(w, p=2)
                    modified_w = (w * (dim_size ** 0.5)) / l2_norm
                    weights.append(modified_w / dim_size)
            elif weight == 'chebyshev':
                for dim_size in self.dimensions:
                    k = torch.arange(1, dim_size + 1, dtype=torch.float64, device=self.device)
                    w = torch.cos((2 * k - 1) * np.pi / (2 * dim_size))
                    w = w.view(dim_size, 1)
                    l2_norm = torch.norm(w, p=2)
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
                            temp = torch.mean(temp, ind)
                        else:
                            ind += 1
                    temp = torch.unsqueeze(temp, -1)
                    support_vectors.append(temp)
            elif supports == 'ones':
                for dim_size in self.dimensions:
                    s = torch.ones(dim_size, 1, dtype=torch.float64, device=self.device)
                    l2_norm = torch.norm(s, p=2)
                    modified_s = (s * (dim_size ** 0.5)) / l2_norm
                    support_vectors.append(modified_s)
            elif supports == 'custom':
                if self.custom_supports is None:
                    raise ValueError("Custom supports must be provided for 'custom' support type.")
                if len(self.custom_supports) != len(self.dimensions):
                    raise ValueError("The number of custom supports must match the number of dimensions.")
                for s in self.custom_supports:
                    if not torch.is_tensor(s):
                        s = torch.tensor(s, dtype=torch.float64, device=self.device)
                    else:
                        s = s.to(self.device)
                    support_vectors.append(s)
            return support_vectors

        def calculate_g0(self):
            g0 = self.G
            for i, (s, w) in enumerate(zip(self.support_vectors, self.weights)):
                g0 = torch.tensordot(g0, s * w, dims=([0], [0]))
            return g0.item()

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
                    G_component = torch.tensordot(G_component, s * w, dims=([ind], [0]))
                else:
                    ind += 1
            # Second part
            subtracted = torch.squeeze((self.support_vectors[involved_dims[0]] * self.weights[involved_dims[0]]) * self.g0)
            for i in range(1, len(involved_dims)):
                subtracted = torch.einsum('...i, jk->...ij', subtracted, 
                                        (self.support_vectors[involved_dims[i]] * self.weights[involved_dims[i]]))
            # Third part
            if len(involved_dims) > 1:
                for i in range(1, len(involved_dims)):
                    for g_combination in combinations(involved_dims, i):
                        s = [x for x in involved_dims if x not in g_combination]
                        term = torch.squeeze(self.g_components[self.convert_g_to_string(g_combination)])
                        for k in range(len(s)):
                            term = torch.einsum('...i, jk->...ij', term, 
                                              (self.support_vectors[s[k]] * self.weights[s[k]]))
                        subtracted += torch.permute(term, np.argsort(list(g_combination) + s).tolist())
            G_component = torch.squeeze(G_component)
            subtracted = torch.squeeze(subtracted)
            G_component = G_component.clone() - subtracted
            self.g_components[self.convert_g_to_string(involved_dims)] = G_component

        def calculate_approximation(self, order):
            involved_dims = np.arange(len(self.dimensions))
            # First part
            overall_sum = torch.squeeze((self.support_vectors[involved_dims[0]] * self.weights[involved_dims[0]]) * self.g0)
            for i in range(1, len(involved_dims)):
                overall_sum = torch.einsum('...i, jk->...ij', overall_sum, 
                                         (self.support_vectors[involved_dims[i]] * self.weights[involved_dims[i]]))
            # Second-N'th part
            for i in range(1, order+1):
                for g_combination in combinations(involved_dims, i):
                    s = [x for x in involved_dims if x not in g_combination]
                    term = torch.squeeze(self.g_components[self.convert_g_to_string(g_combination)])
                    for k in range(len(s)):
                        term = torch.einsum('...i, jk->...ij', term, 
                                          (self.support_vectors[s[k]] * self.weights[s[k]]))
                    overall_sum += torch.permute(term, np.argsort(list(g_combination) + s).tolist())
            return torch.squeeze(overall_sum)

        def calculate_mse(self, order):
            T_empr = self.calculate_approximation(order)
            squared_error = torch.norm(self.G - T_empr, p='fro')
            num_elements = torch.numel(self.G)
            mse = squared_error / num_elements
            return mse.item()

    # EMPR implementation
    class _EMPR:
        def __init__(self, G, supports='das', custom_supports=None, device='cpu'):
            self.device = device
            self.G = torch.tensor(G, dtype=torch.float64, device=device)
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
                            temp = torch.mean(temp, dim=ind)
                        else:
                            ind += 1
                    temp = temp.unsqueeze(-1)
                    support_vectors.append(temp)
            elif supports == 'ones':
                for dim_size in self.dimensions:
                    s = torch.ones((dim_size, 1), dtype=torch.float64, device=self.device)
                    support_vectors.append(s)
            elif supports == 'custom':
                if self.custom_supports is None:
                    raise ValueError("Custom supports must be provided for 'custom' support type.")
                if len(self.custom_supports) != len(self.dimensions):
                    raise ValueError("The number of custom supports must match the number of dimensions.")
                for s in self.custom_supports:
                    if not isinstance(s, torch.Tensor):
                        s = torch.tensor(s, dtype=torch.float64, device=self.device)
                    else:
                        s = s.to(self.device)
                    support_vectors.append(s)
            return support_vectors

        def calculate_g0(self):
            g0 = self.G
            for i, s in enumerate(self.support_vectors):
                g0 = torch.tensordot(g0, s, dims=([0], [0]))
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
                    G_component = torch.tensordot(G_component, s, dims=([ind], [0]))
                else:
                    ind += 1
            # Second part
            subtracted = torch.squeeze(self.support_vectors[involved_dims[0]] * self.g0)
            for i in range(1, len(involved_dims)):
                subtracted = torch.einsum('...i,jk->...ij', subtracted, 
                                        self.support_vectors[involved_dims[i]])
            # Third part
            if len(involved_dims) > 1:
                for i in range(1, len(involved_dims)):
                    for g_combination in combinations(involved_dims, i):
                        s = [x for x in involved_dims if x not in g_combination]
                        term = torch.squeeze(self.g_components[self.convert_g_to_string(g_combination)])
                        for k in range(len(s)):
                            term = torch.einsum('...i,jk->...ij', term, 
                                              self.support_vectors[s[k]])
                        subtracted += term.permute(tuple(np.argsort(list(g_combination) + s)))
            G_component = torch.squeeze(G_component)
            subtracted = torch.squeeze(subtracted)
            G_component = G_component - subtracted
            self.g_components[self.convert_g_to_string(involved_dims)] = G_component

        def calculate_approximation(self, order):
            involved_dims = np.arange(len(self.dimensions))
            # First part
            overall_sum = torch.squeeze(self.support_vectors[involved_dims[0]] * self.g0)
            for i in range(1, len(involved_dims)):
                overall_sum = torch.einsum('...i,jk->...ij', overall_sum, 
                                      self.support_vectors[involved_dims[i]])
            # Second-N'th part
            for i in range(1, order+1):
                for g_combination in combinations(involved_dims, i):
                    s = [x for x in involved_dims if x not in g_combination]
                    term = torch.squeeze(self.g_components[self.convert_g_to_string(g_combination)])
                    for k in range(len(s)):
                        term = torch.einsum('...i,jk->...ij', term, 
                                          self.support_vectors[s[k]])
                    overall_sum += term.permute(tuple(np.argsort(list(g_combination) + s)))
            return torch.squeeze(overall_sum)

        def calculate_mse(self, order):
            T_empr = self.calculate_approximation(order)
            squared_error = torch.norm(self.G - T_empr, p=2)
            num_elements = torch.numel(self.G)
            mse = squared_error / num_elements
            return float(mse)

    def hdmr_decompose(self, tensor, order=2, **kwargs):
        model = self._HDMR(tensor, device=self.device, **kwargs)
        return model.calculate_approximation(order)

    def empr_decompose(self, tensor, order=2, **kwargs):
        model = self._EMPR(tensor, device=self.device, **kwargs)
        return model.calculate_approximation(order)

    def hdmr_components(self, tensor, max_order=None, **kwargs):
        model = self._HDMR(tensor, device=self.device, **kwargs)
        num_dims = len(model.dimensions)
        if max_order is None:
            max_order = num_dims
        components = {}
        dims = list(range(num_dims))
        for r in range(1, min(max_order, num_dims) + 1):
            for comb in combinations(dims, r):
                key = model.convert_g_to_string(comb)
                components[key] = model.g_components[key]
        return components

    def empr_components(self, tensor, max_order=None, **kwargs):
        model = self._EMPR(tensor, device=self.device, **kwargs)
        num_dims = len(model.dimensions)
        if max_order is None:
            max_order = num_dims
        components = {}
        dims = list(range(num_dims))
        for r in range(1, min(max_order, num_dims) + 1):
            for comb in combinations(dims, r):
                key = model.convert_g_to_string(comb)
                components[key] = model.g_components[key]
        return components 