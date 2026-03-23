from sentence_transformers import SentenceTransformer

from apps.core.config import get_settings

settings = get_settings()


class EmbeddingService:

    def __init__(self, model_name: str, device: str):
        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: list[str]) -> list[dict]:
        embeddings = self.model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            show_progress_bar=True,
            normalize_embeddings=settings.embedding_normalize
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        embedding = self.model.encode(
            query,
            normalize_embeddings=settings.embedding_normalize
        )
        return embedding.tolist()
