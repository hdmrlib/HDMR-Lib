from EMPR import NDEMPRCalculator
from HDMR import NDHDMRCalculator
import torch
import numpy as np

def npouter(a, b):
    return np.einsum('ij,kd->ijk', a, b)

torch.manual_seed(0)

H = torch.rand(3, 4, 5)
hdmr_calculator = NDHDMRCalculator(H, weight = "avg", supports = 'ones')

'''
for key in empr_calculator.g_components.keys():
    print(key, " - shape :", empr_calculator.g_components[key].shape, "\n\t", empr_calculator.g_components[key])
'''
    
G = torch.rand(3, 4, 5)
empr_calculator = NDEMPRCalculator(G, supports = 'das')

'''
for key in empr_calculator.g_components.keys():
    print(key, " - shape :", empr_calculator.g_components[key].shape, "\n\t", empr_calculator.g_components[key])
'''

hdmr1 = hdmr_calculator.g0 * npouter(np.squeeze(np.outer(np.array(hdmr_calculator.support_vectors[0] * hdmr_calculator.weights[0]),
                                               np.transpose(np.array(hdmr_calculator.support_vectors[1]) * np.array(hdmr_calculator.weights[1])))),
                                               np.array(hdmr_calculator.support_vectors[2] * hdmr_calculator.weights[2]))

hdmr1 += npouter(np.outer(np.array(hdmr_calculator.g_components['g_1']),
                  np.transpose(np.array(hdmr_calculator.support_vectors[1]) * np.array(hdmr_calculator.weights[1]))),
                  np.array(hdmr_calculator.support_vectors[2] * hdmr_calculator.weights[2]))
hdmr1 += npouter(np.outer(hdmr_calculator.support_vectors[0] * hdmr_calculator.weights[0],
                  np.transpose(np.array(hdmr_calculator.g_components['g_2']))),
                  np.array(hdmr_calculator.support_vectors[2] * hdmr_calculator.weights[2]))
hdmr1 += npouter(np.outer(hdmr_calculator.support_vectors[0] * hdmr_calculator.weights[0],
                  np.transpose(np.array(hdmr_calculator.support_vectors[2] * hdmr_calculator.weights[2]))),
                  (np.array(hdmr_calculator.g_components['g_3'])).reshape( -1, 1))

empr1 = empr_calculator.g0 * npouter(np.squeeze(np.outer(np.array(empr_calculator.support_vectors[0] * empr_calculator.weights[0]),
                                               np.transpose(np.array(empr_calculator.support_vectors[1]) * np.array(empr_calculator.weights[1])))),
                                               np.array(empr_calculator.support_vectors[2] * empr_calculator.weights[2]))

empr1 += npouter(np.outer(np.array(empr_calculator.g_components['g_1']),
                  np.transpose(np.array(empr_calculator.support_vectors[1]))),
                  np.array(empr_calculator.support_vectors[2]))
empr1 += npouter(np.outer(empr_calculator.support_vectors[0],
                  np.transpose(np.array(empr_calculator.g_components['g_2']))),
                  np.array(empr_calculator.support_vectors[2]))
empr1 += npouter(np.outer(empr_calculator.support_vectors[0],
                  np.transpose(np.array(empr_calculator.support_vectors[2]))),
                  np.array(empr_calculator.g_components['g_3']))

norm_empr_hdmr = torch.norm(hdmr1-empr1)
norm_hdmr = torch.norm(empr1)

# Calculate relative error
relative_error = norm_empr_hdmr / norm_hdmr

print(1-relative_error)