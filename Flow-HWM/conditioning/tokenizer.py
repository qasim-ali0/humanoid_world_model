import abc

class Embedder(abc.ABC):
    @abc.abstractmethod
    def tokenize(self, captions):
        pass
    @abc.abstractmethod
    def get_embeddding(self, captions):
        pass
    @property
    def device(self):
        return self._device  # Use the internal variable
    @abc.abstractmethod
    def to(self, device):
        pass