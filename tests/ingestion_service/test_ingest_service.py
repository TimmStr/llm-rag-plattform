from datetime import datetime

import pytest

from apps.core.config import get_settings
from apps.ingestion_service.domain.entities import Chunk, Metadata
from apps.ingestion_service.domain.exceptions import PdfIngestionError, ChunkValidationError, HashValidationError, \
    IngestionError
from apps.ingestion_service.services.ingest_service import enrich_metadata, deduplicate
from apps.ingestion_service.services.ingest_service import ingest_pdf


def test_enrich_metadata_sets_source_hash_and_timestamp():
    settings = get_settings()
    chunks = [
        Chunk(text="Example test", metadata=Metadata(page_number=1, chunk_id=1)),
        Chunk(text="Example test 212", metadata=Metadata(page_number=2, chunk_id=2)),
    ]
    file_path = settings.demo_document_path
    enriched_chunks = enrich_metadata(chunks, file_path)
    assert enriched_chunks is chunks
    assert enriched_chunks[0] is chunks[0]
    assert enriched_chunks[1] is chunks[1]
    assert all(chunk.metadata.source == file_path for chunk in enriched_chunks)
    assert all(isinstance(chunk.metadata.hash, str) for chunk in enriched_chunks)
    assert chunks[0].metadata.hash != chunks[1].metadata.hash
    assert all(isinstance(chunk.metadata.ingested_at, str) for chunk in enriched_chunks)

    for chunk in enriched_chunks:
        assert isinstance(chunk.metadata.ingested_at, str)
        parsed = datetime.fromisoformat(chunk.metadata.ingested_at)
        assert parsed.tzinfo is not None


def test_enrich_metadata_raises_for_empty_chunks():
    with pytest.raises(ChunkValidationError):
        enrich_metadata([], "test.pdf")


@pytest.mark.parametrize("file_path", [5, "", "xyz.pdf", "apps/api", "test.png"])
def test_enrich_metadata_raises_for_invalid_file_path(file_path):
    chunks = [Chunk(text="Example test", metadata=Metadata(page_number=1, chunk_id=1))]

    with pytest.raises(PdfIngestionError):
        enrich_metadata(chunks, file_path)


def test_deduplicate_keeps_all_chunks_when_unique():
    chunks = [
        Chunk(text="A", metadata=Metadata(page_number=1, chunk_id=1, hash="h1")),
        Chunk(text="B", metadata=Metadata(page_number=2, chunk_id=2, hash="h2")),
    ]

    deduplicated = deduplicate(chunks)

    assert deduplicated == chunks
    assert deduplicated[0] is chunks[0]
    assert deduplicated[1] is chunks[1]


def test_deduplicate_removes_duplicates():
    chunks = [
        Chunk(text="A", metadata=Metadata(page_number=1, chunk_id=1, hash="same")),
        Chunk(text="B", metadata=Metadata(page_number=2, chunk_id=2, hash="same")),
    ]

    deduplicated = deduplicate(chunks)

    assert len(deduplicated) == 1
    assert deduplicated[0] is chunks[0]


def test_deduplicate_preserves_order():
    chunks = [
        Chunk(text="A", metadata=Metadata(page_number=1, chunk_id=1, hash="h1")),
        Chunk(text="B", metadata=Metadata(page_number=2, chunk_id=2, hash="h2")),
        Chunk(text="C", metadata=Metadata(page_number=3, chunk_id=3, hash="h1")),  # duplicate
    ]

    deduplicated = deduplicate(chunks)

    assert len(deduplicated) == 2
    assert deduplicated[0] is chunks[0]
    assert deduplicated[1] is chunks[1]


def test_deduplicate_multiple_duplicates():
    chunks = [
        Chunk(text="A", metadata=Metadata(page_number=1, chunk_id=1, hash="x")),
        Chunk(text="B", metadata=Metadata(page_number=2, chunk_id=2, hash="x")),
        Chunk(text="C", metadata=Metadata(page_number=3, chunk_id=3, hash="x")),
    ]

    deduplicated = deduplicate(chunks)

    assert len(deduplicated) == 1
    assert deduplicated[0] is chunks[0]


def test_deduplicate_raises_when_hash_missing():
    chunks = [
        Chunk(text="A", metadata=Metadata(page_number=1, chunk_id=1, hash=None)),
    ]

    with pytest.raises(HashValidationError):
        deduplicate(chunks)


def test_ingest_pdf_integration(tmp_path):
    settings = get_settings()
    pdf_path = settings.demo_document_path

    chunks = ingest_pdf(str(pdf_path))

    assert len(chunks) > 0
    assert all(chunk.metadata.hash for chunk in chunks)


def test_ingest_pdf_integration_ingestion_error(tmp_path):
    pdf_path = "xyz.pdf"
    with pytest.raises(IngestionError):
        chunks = ingest_pdf(str(pdf_path))
