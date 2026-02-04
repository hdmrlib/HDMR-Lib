from .backends import get_backend_instance

class EMPR:
    def __init__(self, tensor, order=2, **kwargs):
        self.tensor = tensor
        self.kwargs = kwargs
        self.order = order
        # Get backend and create model instance for reconstruction
        backend = get_backend_instance()
        self._model = backend.get_empr_model(self.tensor, **self.kwargs)
        
        self.dimensions = self._model.dimensions
        self.weights = self._model.weights
        self.support_vectors = self._model.support_vectors
    
    def reconstruct(self, order=None):
        """Reconstruct/approximate the tensor using calculate_approximation with specified order"""
        if order is None:
            order = self.order
        return self._model.calculate_approximation(order)

    def components(self, elements=None):
        if elements is None:
            return dict(self._model.g_components)
        return {key: self._model.g_components[key] for key in elements}