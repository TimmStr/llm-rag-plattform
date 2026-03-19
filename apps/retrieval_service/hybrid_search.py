import numpy as np
from rank_bm25 import BM25Okapi

from apps.ingestion_service.embedding import EmbeddingService


class HybridSearch:
    def __init__(self, documents: list[dict]):
        self.documents = documents

        self.tokenized_corpus = [doc["text"].split() for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        self.embedding_model = EmbeddingService().model
        self.doc_embeddings = self.doc_embeddings = self.embedding_model.encode(
            [doc["text"] for doc in documents],
            normalize_embeddings=True
        )

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        tokenized_query = query.split()
        bm_25_scores = self.bm25.get_scores(tokenized_query)

        query_embedding = self.embedding_model.encode(query, normalize_embeddings=True)
        dense_scores = np.dot(self.doc_embeddings, query_embedding)

        combined_scores = bm_25_scores + dense_scores

        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            results.append({
                "text": doc["text"],
                "score": float(combined_scores[idx]),
                "metadata": doc.get("metadata", {})
            })
        return results
