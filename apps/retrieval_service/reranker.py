from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str, device: str):
        self.model = CrossEncoder(model_name=model_name,
                                  device=device)

    def rerank(self, query: str, documents: list[dict], top_k: int = 5):
        pairs = [(query, doc["text"]) for doc in documents]
        scores = self.model.predict(pairs)
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        documents = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        return documents[:top_k]
