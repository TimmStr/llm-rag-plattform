from abc import ABC, abstractmethod
from typing import Any

from apps.ingestion_service.domain.entities import Chunk


class VectorStore(ABC):
    @abstractmethod
    def add_documents(self, chunks: list[Chunk]) -> None:
        pass

    @abstractmethod
    def similarity_search(self,
                          query_embedding: list[float],
                          top_k: int = 5) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        pass
