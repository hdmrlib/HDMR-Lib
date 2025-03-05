import tensorflow as tf
import numpy as np
from itertools import combinations

class NDEMPRCalculator:

    def __init__(self, G, supports='das', custom_supports=None):
        
        if tf.config.list_physical_devices('GPU'):
            G = tf.convert_to_tensor(G, dtype=tf.float64)
            self.device = "/GPU:0"
        else:
            print('Defaulted to CPU')
            self.device = "/CPU:0"

        self.G = G
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
                for j in range(len(self.dimensions)):
                    if j != i:
                        temp = tf.reduce_mean(temp, axis=0)
                temp = tf.expand_dims(temp, axis=-1)
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

            for w in self.custom_supports:
                if not tf.is_tensor(w):
                    s = tf.convert_to_tensor(w, dtype=tf.float64)
                support_vectors.append(s)

        return support_vectors

    def calculate_g0(self):
        g0 = self.G
        for i, (s, w) in enumerate(zip(self.support_vectors, self.weights)):
            g0 = tf.tensordot(g0, s, axes=[[0], [0]]) * w
        return g0.numpy().item()

    def convert_g_to_string(self, dims):
        return 'g_' + ','.join(map(str, list(map(lambda x: x + 1, dims))))

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

        # FIRST PART
        for j, (s, w) in enumerate(zip(self.support_vectors, self.weights)):
            if j not in involved_dims:
                G_component = tf.tensordot(G_component, s, axes=[[0], [0]]) * w

        # SECOND PART
        subtracted = self.support_vectors[involved_dims[0]] * self.g0
        for i in range(1, len(involved_dims)):
            subtracted = tf.einsum('...i,jk->...ij', subtracted, self.support_vectors[involved_dims[i]])

        # THIRD PART
        if len(involved_dims) > 1:
            for i in range(1, len(involved_dims)):
                for g_combination in combinations(involved_dims, i):
                    s = [x for x in involved_dims if x not in g_combination]
                    term = self.g_components[self.convert_g_to_string(g_combination)]

                    for k in range(len(s)):
                        term = tf.einsum('...i,jk->...ij', term, self.support_vectors[s[k]])

                    subtracted += tf.transpose(term, np.argsort(list(g_combination) + s))

        G_component = tf.squeeze(G_component - subtracted)
        self.g_components[self.convert_g_to_string(involved_dims)] = G_component

    def calculate_approximation(self, order):
        involved_dims = np.arange(len(self.dimensions))

        # FIRST PART
        overall_sum = self.support_vectors[involved_dims[0]] * self.g0
        for i in range(1, len(involved_dims)):
            overall_sum = tf.einsum('...i,jk->...ij', overall_sum, self.support_vectors[involved_dims[i]])

        # SECOND-N'TH PART
        for i in range(1, order + 1):
            for g_combination in combinations(involved_dims, i):
                s = [x for x in involved_dims if x not in g_combination]
                term = self.g_components[self.convert_g_to_string(g_combination)]

                for k in range(len(s)):
                    term = tf.einsum('...i,jk->...ij', term, self.support_vectors[s[k]])

                overall_sum += tf.transpose(term, np.argsort(list(g_combination) + s))

        return tf.squeeze(overall_sum)
