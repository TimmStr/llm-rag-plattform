from os import path, getcwd

from apps.ingestion_service.ingest import ingest_pdf
from apps.retrieval_service.hybrid_search import HybridSearch
from apps.retrieval_service.reranker import Reranker


class Retriever:
    def __init__(self, documents: list[dict]):
        self.hybrid_search = HybridSearch(documents)
        self.reranker = Reranker()

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        candidates = self.hybrid_search.search(query, top_k=20)
        results = self.reranker.rerank(query, candidates, top_k=top_k)
        return results


if __name__ == "__main__":
    retriever = Retriever(ingest_pdf(path.join(getcwd(), "data", "raw", "_10-K-2025-As-Filed.pdf")))
    results = retriever.retrieve("European Commission State Aid Decision?", top_k=10)
    for r in results:
        print(r["text"])
        print(r["score"])
