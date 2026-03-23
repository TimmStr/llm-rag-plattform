from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from qdrant_client.models import PointStruct

from apps.core.config import Settings
from apps.core.logging_config import get_logger
from apps.core.vector_store.base import VectorStore

logger = get_logger(__name__)


class QdrantVectorStore(VectorStore):
    def __init__(self,
                 collection_name: str,
                 host: str = "localhost",
                 port: int = 6333):
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)
        self._ensure_collection(vector_size=Settings.embedding_dimension)
        logger.info(f"Initialized QdrantVectorStore collection={self.collection_name} host={host} port={port}")

    def _ensure_collection(self, vector_size: int) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size,
                                            distance=Distance.COSINE)
            )

    def add_documents(self,
                      ids: list[str],
                      embeddings: list[list[float]],
                      metadatas: list[dict[str, Any]]) -> None:
        logger.info(f"Upserting {len(ids)} documents into Qdrant collection={self.collection_name}")
        points = [
            PointStruct(
                id=ids[index],
                vector=embeddings[index],
                payload=metadatas[index]
            )
            for index in range(len(ids))
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def similarity_search(self,
                          query_embedding: list[float],
                          top_k: int = 5) -> list[dict[str, Any]]:
        logger.info(f"Similarity search with query_embedding={query_embedding} top_k={top_k}")
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k
        )
        return [
            {
                "id": point.id,
                "score": point.score,
                "metadata": point.payload
            }
            for point in results.points
        ]

    def delete(self, ids: list[str]) -> None:
        logger.info(f"Delete {len(ids)} documents from Qdrant collection={self.collection_name}")
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=ids
        )
