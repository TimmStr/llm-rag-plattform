from abc import ABC, abstractmethod
from typing import Any


class VectorStore(ABC):
    @abstractmethod
    def add_documents(self,
                      ids: list[str],
                      embeddings: list[list[float]],
                      metadatas: list[dict[str, Any]]) -> None:
        pass

    @abstractmethod
    def similarity_search(self,
                          query_embedding: list[float],
                          top_k: int = 5) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        pass
