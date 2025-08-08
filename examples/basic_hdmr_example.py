import hdmr_core as hdmr
import numpy as np

# Set backend (optional, default is numpy)
hdmr.set_backend('numpy')

tensor = np.random.rand(5, 5, 5)

model = hdmr.HDMR(tensor)
result = model.decompose(order=2)
print('HDMR decomposition result:', result) 