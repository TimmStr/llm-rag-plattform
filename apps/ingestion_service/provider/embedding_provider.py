from sentence_transformers import SentenceTransformer

from apps.core.config import get_settings
from apps.core.logging_config import get_logger
from apps.ingestion_service.domain.exceptions import EmptyModelNameError, InvalidDeviceError, EmbeddingInitializationError, \
    EmbeddingFailedError, EmbeddingInferenceError

settings = get_settings()

logger = get_logger(__name__)


class EmbeddingService:

    def __init__(self, model_name: str, device: str):
        if not model_name:
            raise EmptyModelNameError(f"Invalid model name: {model_name}")
        if not device in ("cuda", "cpu", "mps", "npu"):
            raise InvalidDeviceError(f"Invalid device: {device}")
        logger.info(f"Initialize EmbeddingService with model=%s device=%s", model_name, device)
        try:
            self.model = SentenceTransformer(model_name, device=device)
        except Exception as exc:
            raise EmbeddingInitializationError(f"Failed to initialize model: {exc}") from exc

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not isinstance(texts, list):
            raise EmbeddingInferenceError("Texts must be a list of strings")
        if not texts:
            return []
        if any(not isinstance(text, str) for text in texts):
            raise EmbeddingInferenceError("All items in texts must be strings")
        logger.info(f"Embedding %s texts", len(texts))
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=settings.embedding_batch_size,
                show_progress_bar=True,
                normalize_embeddings=settings.embedding_normalize
            )
        except Exception as exc:
            raise EmbeddingFailedError(f"Failed to embed texts: {exc}") from exc
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        if not isinstance(query, str):
            raise EmbeddingInferenceError("Query must be a string")
        if len(query) == 0:
            raise EmbeddingInferenceError("Query must not be empty")
        logger.info(f"Embedding query with length=%s", len(query))
        try:
            embedding = self.model.encode(
                query,
                normalize_embeddings=settings.embedding_normalize
            )
        except Exception as exc:
            raise EmbeddingFailedError(f"Failed to embed query: {exc}") from exc
        return embedding.tolist()
