from abc import ABC, abstractmethod
from typing import List


class EmbeddingModel(ABC):
    @abstractmethod
    def encode(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    def encode_query(self, query: str) -> List[float]:
        pass
