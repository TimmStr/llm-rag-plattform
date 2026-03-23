import hashlib
import uuid
from datetime import datetime

from apps.core.logging_config import get_logger
from apps.core.vector_store.base import VectorStore
from apps.ingestion_service.chunking import extract_pages, split_pages
from apps.ingestion_service.embedding import EmbeddingService
from apps.retrieval_service.vector_store.qdrant_store import QdrantVectorStore

logger = get_logger(__name__)


def hash_to_uuid(hash_str: str) -> str:
    return str(uuid.UUID(hash_str[:32]))


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def enrich_metadata(chunks: list[dict], source) -> list[dict]:
    logger.info(f"Enriching metadata for {len(chunks)} chunks from source={source}")

    enriched = []
    for chunk in chunks:
        metadata = {
            **chunk["metadata"],
            "source": source,
            "hash": hash_text(chunk["text"]),
            "ingested_at": datetime.now().isoformat()
        }
        enriched.append({
            "text": chunk["text"],
            "metadata": metadata
        })
    return enriched


def deduplicate(chunks: list[dict]) -> list[dict]:
    logger.info(f"Deduplicating {len(chunks)} chunks", )
    seen = set()
    unique = []
    for chunk in chunks:
        h = chunk["metadata"]["hash"]
        if h not in seen:
            seen.add(h)
            unique.append(chunk)
    return unique


def ingest_pdf(file_path: str) -> list[dict]:
    logger.info(f"Starting PDF ingestion for file={file_path}")
    pages = extract_pages(file_path)
    logger.info(f"Extracted {len(pages)} pages")
    chunks = split_pages(pages)
    logger.info(f"Split pages into {len(chunks)} chunks")
    chunks = enrich_metadata(chunks, file_path)
    chunks = deduplicate(chunks)
    logger.info(f"Finished ingestion for file={file_path}")
    return chunks


def embed_chunks(chunks: list[dict],
                 embedding_service: EmbeddingService) -> list[dict]:
    logger.info(f"Creating embeddings for {len(chunks)} chunks")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_service.embed_texts(texts)

    enriched = []
    for chunk, embedding in zip(chunks, embeddings):
        enriched.append({
            **chunk,
            "embedding": embedding
        })
    logger.info(f"Created embeddings for {len(enriched)} chunks", )
    return enriched


def index_chunks(chunks: list[dict],
                 vector_store: VectorStore) -> None:
    ids = []
    embeddings = []
    metadatas = []
    for chunk in chunks:
        ids.append(hash_to_uuid(chunk["metadata"]["hash"]))
        embeddings.append(chunk["embedding"])

        metadatas.append({
            "text": chunk["text"],
            **chunk["metadata"]
        })
    vector_store.add_documents(ids=ids,
                               embeddings=embeddings,
                               metadatas=metadatas)


def ingest_and_index(
        file_path: str,
        vector_store: VectorStore,
        embedding_service: EmbeddingService) -> None:
    chunks = ingest_pdf(file_path)
    chunks = embed_chunks(chunks, embedding_service)
    index_chunks(chunks, vector_store)


if __name__ == '__main__':
    from apps.core.config import get_settings

    settings = get_settings()
    vector_store = QdrantVectorStore(settings.qdrant_collection_name)
    embedding_service = EmbeddingService(settings.embedding_model, settings.embedding_device)

    ingest_and_index(
        file_path=settings.demo_document_path,
        vector_store=vector_store,
        embedding_service=embedding_service
    )
