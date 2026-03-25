from apps.core.exceptions import IngestionError
from apps.core.logging_config import get_logger
from apps.core.vector_store.base import VectorStore
from apps.ingestion_service.domain.entities import Chunk
from apps.ingestion_service.domain.exceptions import VectorIndexingError
from apps.ingestion_service.domain.validators import validate_chunks

logger = get_logger(__name__)


def docs_to_vectorstore(chunks: list[Chunk],
                        vector_store: VectorStore) -> None:
    validate_chunks(chunks)

    if vector_store is None:
        raise VectorIndexingError("Vector store must not be None")

    for index, chunk in enumerate(chunks):
        if getattr(chunk, "embedding", None) is None:
            raise VectorIndexingError(f"Chunk at index {index} has no embedding")
        chunk_hash = getattr(chunk.metadata, "hash", None)
        if not isinstance(chunk_hash, str) or not chunk_hash:
            raise VectorIndexingError(f"Chunk at index {index} has missing or invalid metadata.hash")

    logger.info("Indexing %s chunks", len(chunks))

    try:
        vector_store.add_documents(chunks)
    except IngestionError:
        raise
    except Exception as exc:
        logger.exception("Failed to index %s chunks", len(chunks))
        raise VectorIndexingError(f"Failed to index {len(chunks)} chunks in vector store") from exc
