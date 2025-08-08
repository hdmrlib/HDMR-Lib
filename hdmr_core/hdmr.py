from .backends import get_backend_instance

class HDMR:
    def __init__(self, tensor, **kwargs):
        self.tensor = tensor
        self.kwargs = kwargs

    def decompose(self, order=2):
        backend = get_backend_instance()
        return backend.hdmr_decompose(self.tensor, order=order, **self.kwargs)

    def sensitivity(self):
        # (Optional) Return sensitivity indices
        pass 