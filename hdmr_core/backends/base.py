from abc import ABC, abstractmethod

class BaseBackend(ABC):
    @abstractmethod
    def hdmr_decompose(self, tensor, order=2):
        pass

    @abstractmethod
    def empr_decompose(self, tensor, order=2):
        pass 