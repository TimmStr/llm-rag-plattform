import hashlib
import os.path
from datetime import datetime

from apps.ingestion_service.chunking import extract_pages, split_pages
from apps.ingestion_service.embedding import EmbeddingService


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def enrich_metadata(chunks: list[dict], source) -> list[dict]:
    for chunk in chunks:
        chunk["metadata"].update(
            {
                "source": source,
                "hash": hash_text(chunk["text"]),
                "ingested_at": datetime.now().isoformat()
            }
        )
    return chunks


def deduplicate(chunks: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for chunk in chunks:
        h = chunk["metadata"]["hash"]
        if h not in seen:
            seen.add(h)
            unique.append(chunk)
    return unique


def ingest_pdf(file_path: str) -> list[dict]:
    # Extract
    pages = extract_pages(file_path)

    # Chunk
    chunks = split_pages(pages)

    # Metadata
    chunks = enrich_metadata(chunks, file_path)

    # Remove duplicates
    chunks = deduplicate(chunks)

    # Embedding
    embedder = EmbeddingService()
    chunks = embedder.embed(chunks)
    return chunks


if __name__ == '__main__':
    print(ingest_pdf(os.path.join(os.getcwd(), "data", "raw", "_10-K-2025-As-Filed.pdf")))
