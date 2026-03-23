from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from qdrant_client.models import PointStruct
from sympy import resultant

from apps.core.vector_store.base import VectorStore
from apps.ingestion_service.embedding import load_embedding_config


class QdrantVectorStore(VectorStore):
    def __init__(self,
                 collection_name: str,
                 host: str = "localhost",
                 port: int = 6333):
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)
        self._ensure_collection(vector_size=load_embedding_config().dimension)

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
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=ids
        )
