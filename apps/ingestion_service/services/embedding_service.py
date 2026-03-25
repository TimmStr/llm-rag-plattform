from apps.core.exceptions import EmbeddingError, IngestionError
from apps.core.logging_config import get_logger
from apps.ingestion_service.domain.entities import Chunk
from apps.ingestion_service.domain.validators import validate_chunks, validate_embeddings
from apps.ingestion_service.provider.embedding_provider import EmbeddingService

logger = get_logger(__name__)


def embed_chunks(chunks: list[Chunk],
                 embedding_service: EmbeddingService) -> list[Chunk]:
    validate_chunks(chunks)

    if embedding_service is None:
        raise EmbeddingError("Embedding service must not be None")

    logger.info(f"Creating embeddings for %s chunks", len(chunks))

    texts = [chunk.text for chunk in chunks]

    try:
        embeddings = embedding_service.embed_texts(texts)
    except IngestionError:
        raise
    except Exception as exc:
        logger.exception("Failed to create embeddings for %s chunks", len(chunks))
        raise EmbeddingError(f"Failed to create embeddings for {len(chunks)} chunks") from exc

    embeddings = validate_embeddings(embeddings, expected_count=len(chunks))

    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        try:
            chunk.embedding = embedding
        except Exception as exc:
            raise EmbeddingError(f"Failed to assign embedding to chunk at index {index}") from exc

    logger.info(f"Created embeddings for %s chunks", len(chunks))
    return chunks
