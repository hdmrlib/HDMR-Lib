from backends import get_backend_instance

class HDMR:
    def __init__(self, tensor, **kwargs):
        self.tensor = tensor
        self.kwargs = kwargs

    def decompose(self, order=2):
        backend = get_backend_instance()
        return backend.hdmr_decompose(self.tensor, order=order, **self.kwargs)

    def components(self, max_order=None):
        backend = get_backend_instance()
        return backend.hdmr_components(self.tensor, max_order=max_order, **self.kwargs)