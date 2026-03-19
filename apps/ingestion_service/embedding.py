from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, chunks: list[dict]) -> list[dict]:
        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb.tolist()
        return chunks
