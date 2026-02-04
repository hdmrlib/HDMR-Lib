from abc import ABC, abstractmethod

class BaseBackend(ABC):

    @abstractmethod
    def get_hdmr_model(self, tensor, **kwargs):
        """Return an HDMR model instance for reconstruction."""
        pass

    @abstractmethod
    def get_empr_model(self, tensor, **kwargs):
        """Return an EMPR model instance for reconstruction."""
        pass 