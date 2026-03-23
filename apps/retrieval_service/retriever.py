from apps.core.vector_store.base import VectorStore
from apps.ingestion_service.embedding import EmbeddingService
from apps.retrieval_service.hybrid_search import HybridSearch
from apps.retrieval_service.reranker import Reranker


class Retriever:
    def __init__(self,
                 documents: list[dict],
                 vector_store: VectorStore,
                 embedding_service: EmbeddingService):
        self.hybrid_search = HybridSearch(documents=documents,
                                          vector_store=vector_store,
                                          embedding_service=embedding_service)
        self.reranker = Reranker()

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        candidates = self.hybrid_search.search(query, top_k=20)
        results = self.reranker.rerank(query, candidates, top_k=top_k)
        return results
