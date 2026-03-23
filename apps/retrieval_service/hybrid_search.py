from rank_bm25 import BM25Okapi

from apps.core.vector_store.base import VectorStore
from apps.ingestion_service.embedding import EmbeddingService
from apps.ingestion_service.ingest import hash_to_uuid


class HybridSearch:
    def __init__(self,
                 documents: list[dict],
                 vector_store: VectorStore,
                 embedding_service: EmbeddingService):
        self.documents = documents
        self.vector_store = vector_store
        self.embedding_service = embedding_service

        self.tokenized_corpus = [doc["text"].split() for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        tokenized_query = query.split()
        bm_25_scores = self.bm25.get_scores(tokenized_query)

        query_embedding = self.embedding_service.embed_query(query)
        dense_results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k
        )

        dense_score_map = {r["id"]: r["score"] for r in dense_results}
        results = []

        for doc in self.documents:
            doc_id = hash_to_uuid(doc["metadata"]["hash"])
            dense_score = dense_score_map.get(doc_id, 0.0)
            bm_25_score = bm_25_scores[self.documents.index(doc)]
            combined_score = float(bm_25_score + dense_score)
            results.append({
                "text": doc["text"],
                "score": combined_score,
                "metadata": doc["metadata"]
            })
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results[:top_k]
