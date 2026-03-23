import os

from sentence_transformers import SentenceTransformer

from apps.core.embedding_config import EmbeddingConfig


def load_embedding_config() -> EmbeddingConfig:
    return EmbeddingConfig(
        model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        dimension=int(os.getenv("EMBEDDING_DIMENSION", "384"))
    )


class EmbeddingService:
    embedding_config: EmbeddingConfig = load_embedding_config()

    def __init__(self, model_name: str = embedding_config.model_name):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[dict]:
        embeddings = self.model.encode(
            texts,
            batch_size=self.embedding_config.batch_size,
            show_progress_bar=True,
            normalize_embeddings=self.embedding_config.normalize
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        embedding = self.model.encode(
            query,
            normalize_embeddings=True
        )
        return embedding.tolist()
