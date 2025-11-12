import tensorflow as tf
import numpy as np
from itertools import combinations
from .base import BaseBackend

class TensorFlowBackend(BaseBackend):
    def __init__(self):
        """Initialize TensorFlowBackend with device selection."""
        # TensorFlow automatically uses GPU if available
        self.gpus = tf.config.list_physical_devices('GPU')
        if self.gpus:
            try:
                # Enable memory growth to avoid allocating all GPU memory
                for gpu in self.gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
    # HDMR implementation
    class _HDMR:
        def __init__(self, G, weight="avg", custom_weights=None, supports='ones', custom_supports=None):
            self.G = tf.convert_to_tensor(G, dtype=tf.float64)
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
                    w = tf.ones((dim_size, 1), dtype=tf.float64)
                    l2_norm = tf.norm(w, ord=2)
                    modified_w = (w * (dim_size ** 0.5)) / l2_norm
                    weights.append(modified_w / dim_size)
            elif weight == 'custom':
                if self.custom_weights is None:
                    raise ValueError("Custom weights must be provided for 'custom' weight type.")
                if len(self.custom_weights) != len(self.dimensions):
                    raise ValueError("The number of custom weights must match the number of dimensions.")
                for w in self.custom_weights:
                    if not isinstance(w, tf.Tensor):
                        w = tf.convert_to_tensor(w, dtype=tf.float64)
                    weights.append(w)
            elif weight == 'gaussian':
                for dim_size in self.dimensions:
                    w = tf.random.normal((dim_size, 1), dtype=tf.float64)
                    l2_norm = tf.norm(w, ord=2)
                    modified_w = (w * (dim_size ** 0.5)) / l2_norm
                    weights.append(modified_w / dim_size)
            elif weight == 'chebyshev':
                for dim_size in self.dimensions:
                    k = tf.range(1, dim_size + 1, dtype=tf.float64)
                    w = tf.cos((2 * k - 1) * np.pi / (2 * dim_size))
                    w = tf.reshape(w, (dim_size, 1))
                    l2_norm = tf.norm(w, ord=2)
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
                            temp = tf.reduce_mean(temp, axis=ind)
                        else:
                            ind += 1
                    temp = tf.expand_dims(temp, -1)
                    support_vectors.append(temp)
            elif supports == 'ones':
                for dim_size in self.dimensions:
                    s = tf.ones((dim_size, 1), dtype=tf.float64)
                    l2_norm = tf.norm(s, ord=2)
                    modified_s = (s * (dim_size ** 0.5)) / l2_norm
                    support_vectors.append(modified_s)
            elif supports == 'custom':
                if self.custom_supports is None:
                    raise ValueError("Custom supports must be provided for 'custom' support type.")
                if len(self.custom_supports) != len(self.dimensions):
                    raise ValueError("The number of custom supports must match the number of dimensions.")
                for s in self.custom_supports:
                    if not isinstance(s, tf.Tensor):
                        s = tf.convert_to_tensor(s, dtype=tf.float64)
                    support_vectors.append(s)
            return support_vectors

        def calculate_g0(self):
            g0 = self.G
            for i, (s, w) in enumerate(zip(self.support_vectors, self.weights)):
                g0 = tf.tensordot(g0, s * w, axes=[[0], [0]])
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
                    G_component = tf.tensordot(G_component, s * w, axes=[[ind], [0]])
                else:
                    ind += 1
            # Second part
            subtracted = tf.squeeze((self.support_vectors[involved_dims[0]] * self.weights[involved_dims[0]]) * self.g0)
            for i in range(1, len(involved_dims)):
                subtracted = tf.einsum('...i,jk->...ij', subtracted, 
                                     (self.support_vectors[involved_dims[i]] * self.weights[involved_dims[i]]))
            # Third part
            if len(involved_dims) > 1:
                for i in range(1, len(involved_dims)):
                    for g_combination in combinations(involved_dims, i):
                        s = [x for x in involved_dims if x not in g_combination]
                        term = tf.squeeze(self.g_components[self.convert_g_to_string(g_combination)])
                        for k in range(len(s)):
                            term = tf.einsum('...i,jk->...ij', term, 
                                           (self.support_vectors[s[k]] * self.weights[s[k]]))
                        subtracted += tf.transpose(term, perm=np.argsort(list(g_combination) + s))
            G_component = tf.squeeze(G_component)
            subtracted = tf.squeeze(subtracted)
            G_component = G_component - subtracted
            self.g_components[self.convert_g_to_string(involved_dims)] = G_component

        def calculate_approximation(self, order):
            involved_dims = np.arange(len(self.dimensions))
            # First part
            overall_sum = tf.squeeze((self.support_vectors[involved_dims[0]] * self.weights[involved_dims[0]]) * self.g0)
            for i in range(1, len(involved_dims)):
                overall_sum = tf.einsum('...i,jk->...ij', overall_sum, 
                                      (self.support_vectors[involved_dims[i]] * self.weights[involved_dims[i]]))
            # Second-N'th part
            for i in range(1, order+1):
                for g_combination in combinations(involved_dims, i):
                    s = [x for x in involved_dims if x not in g_combination]
                    term = tf.squeeze(self.g_components[self.convert_g_to_string(g_combination)])
                    for k in range(len(s)):
                        term = tf.einsum('...i,jk->...ij', term, 
                                       (self.support_vectors[s[k]] * self.weights[s[k]]))
                    overall_sum += tf.transpose(term, perm=np.argsort(list(g_combination) + s))
            return tf.squeeze(overall_sum)

        def calculate_mse(self, order):
            T_empr = self.calculate_approximation(order)
            squared_error = tf.norm(self.G - T_empr, ord='fro')
            num_elements = tf.size(self.G)
            mse = squared_error / tf.cast(num_elements, tf.float64)
            return float(mse)

    # EMPR implementation
    class _EMPR:
        def __init__(self, G, supports='das', custom_supports=None):
            self.G = tf.convert_to_tensor(G, dtype=tf.float64)
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
                            temp = tf.reduce_mean(temp, axis=ind)
                        else:
                            ind += 1
                    temp = tf.expand_dims(temp, -1)
                    support_vectors.append(temp)
            elif supports == 'ones':
                for dim_size in self.dimensions:
                    s = tf.ones((dim_size, 1), dtype=tf.float64)
                    support_vectors.append(s)
            elif supports == 'custom':
                if self.custom_supports is None:
                    raise ValueError("Custom supports must be provided for 'custom' support type.")
                if len(self.custom_supports) != len(self.dimensions):
                    raise ValueError("The number of custom supports must match the number of dimensions.")
                for s in self.custom_supports:
                    if not isinstance(s, tf.Tensor):
                        s = tf.convert_to_tensor(s, dtype=tf.float64)
                    support_vectors.append(s)
            return support_vectors

        def calculate_g0(self):
            g0 = self.G
            for i, s in enumerate(self.support_vectors):
                g0 = tf.tensordot(g0, s, axes=[[0], [0]])
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
                    G_component = tf.tensordot(G_component, s, axes=[[ind], [0]])
                else:
                    ind += 1
            # Second part
            subtracted = tf.squeeze(self.support_vectors[involved_dims[0]] * self.g0)
            for i in range(1, len(involved_dims)):
                subtracted = tf.einsum('...i,jk->...ij', subtracted, 
                                     self.support_vectors[involved_dims[i]])
            # Third part
            if len(involved_dims) > 1:
                for i in range(1, len(involved_dims)):
                    for g_combination in combinations(involved_dims, i):
                        s = [x for x in involved_dims if x not in g_combination]
                        term = tf.squeeze(self.g_components[self.convert_g_to_string(g_combination)])
                        for k in range(len(s)):
                            term = tf.einsum('...i,jk->...ij', term, 
                                           self.support_vectors[s[k]])
                        subtracted += tf.transpose(term, perm=tuple(np.argsort(list(g_combination) + s)))
            G_component = tf.squeeze(G_component)
            subtracted = tf.squeeze(subtracted)
            G_component = G_component - subtracted
            self.g_components[self.convert_g_to_string(involved_dims)] = G_component

        def calculate_approximation(self, order):
            involved_dims = np.arange(len(self.dimensions))
            # First part
            overall_sum = tf.squeeze(self.support_vectors[involved_dims[0]] * self.g0)
            for i in range(1, len(involved_dims)):
                overall_sum = tf.einsum('...i,jk->...ij', overall_sum, 
                                      self.support_vectors[involved_dims[i]])
            # Second-N'th part
            for i in range(1, order+1):
                for g_combination in combinations(involved_dims, i):
                    s = [x for x in involved_dims if x not in g_combination]
                    term = tf.squeeze(self.g_components[self.convert_g_to_string(g_combination)])
                    for k in range(len(s)):
                        term = tf.einsum('...i,jk->...ij', term, 
                                       self.support_vectors[s[k]])
                    overall_sum += tf.transpose(term, perm=tuple(np.argsort(list(g_combination) + s)))
            return tf.squeeze(overall_sum)

        def calculate_mse(self, order):
            T_empr = self.calculate_approximation(order)
            squared_error = tf.norm(self.G - T_empr, ord=2)
            num_elements = tf.size(self.G)
            mse = squared_error / tf.cast(num_elements, tf.float64)
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