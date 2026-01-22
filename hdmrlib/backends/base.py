from abc import ABC, abstractmethod

class BaseBackend(ABC):
    @abstractmethod
    def hdmr_decompose(self, tensor, order=2):
        pass

    @abstractmethod
    def empr_decompose(self, tensor, order=2):
        pass

    @abstractmethod
    def hdmr_components(self, tensor, max_order=None, **kwargs):
        """Return a dict of component arrays keyed like 'g1','g12',... up to max_order (or full)."""
        pass

    @abstractmethod
    def empr_components(self, tensor, max_order=None, **kwargs):
        """Return a dict of component arrays keyed like 'g1','g12',... up to max_order (or full)."""
        pass 