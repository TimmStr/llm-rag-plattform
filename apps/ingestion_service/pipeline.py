from apps.core.exceptions import IngestionError, IngestionPipelineError
from apps.core.logging_config import get_logger
from apps.core.vector_store.base import VectorStore
from apps.ingestion_service.provider.embedding_provider import EmbeddingService
from apps.ingestion_service.services.embedding_service import embed_chunks
from apps.ingestion_service.services.indexing_service import docs_to_vectorstore
from apps.ingestion_service.services.ingest_service import ingest_pdf
from apps.retrieval_service.vector_store.qdrant_store import QdrantVectorStore

logger = get_logger(__name__)


def ingest_and_index(
        file_path: str,
        vector_store: VectorStore,
        embedding_service: EmbeddingService) -> None:
    logger.info(f"Starting ingestion pipeline for file %s", file_path)

    try:
        chunks = ingest_pdf(file_path)
        chunks = embed_chunks(chunks, embedding_service)
        docs_to_vectorstore(chunks, vector_store)
    except IngestionError:
        raise  # Bekannte ingestion Fehler einfach weiterreichen und unbekannt Fehler sauber in IngestionError packen
    except Exception as exc:
        logger.exception("Unexpected pipeline failure for file=%s", file_path)
        raise IngestionPipelineError(f"Unexpected ingestion pipeline failure for file={file_path}") from exc
    logger.info("Finished ingestion pipeline for file=%s", file_path)


if __name__ == '__main__':
    from apps.core.config import get_settings

    settings = get_settings()
    vector_store = QdrantVectorStore(settings.qdrant_collection_name)
    embedding_service = EmbeddingService(
        settings.embedding_model,
        settings.embedding_device
    )

    ingest_and_index(
        file_path=settings.demo_document_path,
        vector_store=vector_store,
        embedding_service=embedding_service
    )
