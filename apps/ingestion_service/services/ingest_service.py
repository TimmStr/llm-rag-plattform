from datetime import datetime, timezone

from apps.core.exceptions import IngestionError
from apps.core.logging_config import get_logger
from apps.ingestion_service.domain.entities import Chunk
from apps.ingestion_service.domain.exceptions import PdfIngestionError, ChunkValidationError, HashValidationError
from apps.ingestion_service.domain.validators import validate_chunks, validate_pdf_path
from apps.ingestion_service.infrastructure.chunking import extract_pages, split_pages
from apps.ingestion_service.infrastructure.hashing import hash_text

logger = get_logger(__name__)


def enrich_metadata(chunks: list[Chunk], file_path: str) -> list[Chunk]:
    validate_chunks(chunks)
    validate_pdf_path(file_path)
    logger.info("Enriching metadata for %s chunks from source=%s", len(chunks), file_path)

    for chunk in chunks:
        chunk.metadata.source = file_path
        chunk.metadata.hash = hash_text(chunk.text)
        chunk.metadata.ingested_at = datetime.now(timezone.utc).isoformat()
    return chunks


def deduplicate(chunks: list[Chunk]) -> list[Chunk]:
    validate_chunks(chunks)
    if not all(isinstance(chunk.metadata.hash, str) for chunk in chunks):
        raise HashValidationError(f"All hash values must be str")

    logger.info("Deduplicating %s chunks", len(chunks))
    seen: set[str] = set()
    unique: list[Chunk] = []
    for index, chunk in enumerate(chunks):
        chunk_hash = chunk.metadata.hash

        if chunk_hash not in seen:
            seen.add(chunk_hash)
            unique.append(chunk)
    return unique


def ingest_pdf(pdf_file_path: str) -> list[Chunk]:
    validate_pdf_path(pdf_file_path)
    logger.info("Starting PDF ingestion for file=%s", pdf_file_path)
    try:
        pages = extract_pages(pdf_file_path)
        logger.info("Extracted %s pages", len(pages))
        chunks = split_pages(pages)
        logger.info("Split pages into %s chunks", len(chunks))
        chunks = enrich_metadata(chunks, pdf_file_path)
        chunks = deduplicate(chunks)
        logger.info("Finished ingestion for file=%s with %s chunks", pdf_file_path, len(chunks))
        return chunks
    except IngestionError:
        raise
    except Exception as exc:
        logger.exception("PDF ingestion failed for file=%s", pdf_file_path)
        raise PdfIngestionError(f"Failed to ingest PDF: {pdf_file_path}") from exc
