from sentence_transformers import SentenceTransformer

from apps.core.config import get_settings
from apps.core.logging_config import get_logger

settings = get_settings()

logger = get_logger(__name__)


class EmbeddingService:

    def __init__(self, model_name: str, device: str):
        self.model = SentenceTransformer(model_name, device=device)
        logger.info(f"Initializing EmbeddingService with model={model_name} device={device}")

    def embed_texts(self, texts: list[str]) -> list[dict]:
        logger.info(f"Embedding {len(texts)} texts")
        embeddings = self.model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            show_progress_bar=True,
            normalize_embeddings=settings.embedding_normalize
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        logger.info(f"Embedding query with length={len(query)}")
        embedding = self.model.encode(
            query,
            normalize_embeddings=settings.embedding_normalize
        )
        return embedding.tolist()
