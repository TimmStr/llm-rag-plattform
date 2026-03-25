from pathlib import Path

from apps.ingestion_service.domain.entities import Chunk
from apps.ingestion_service.domain.exceptions import PdfIngestionError, ChunkValidationError, EmbeddingError


def validate_pdf_path(file_path: str) -> Path:
    if not isinstance(file_path, str):
        raise PdfIngestionError(f"File path must be str, got {type(file_path).__name__}")
    if not file_path.strip():
        raise PdfIngestionError(f"File path must not be empty string")
    path = Path(file_path)

    if not path.exists():
        raise PdfIngestionError(f"File path does not exist: {file_path}")
    if not path.is_file():
        raise PdfIngestionError(f"File path is not a file: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise PdfIngestionError(f"File path is not a PDF file: {file_path}")
    return path


def validate_chunks(chunks: list[Chunk]) -> None:
    if not isinstance(chunks, list):
        raise ChunkValidationError(f"Chunks must be list, got {type(chunks).__name__}")
    if not any(isinstance(chunk, Chunk) for chunk in chunks):
        raise ChunkValidationError(f"Chunks must be list[Chunk], got {type(chunks).__name__}")

    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, Chunk):
            raise ChunkValidationError(f"Chunk at index {index} must be Chunk, got {type(chunk).__name__}")
        if not isinstance(chunk.text, str):
            raise ChunkValidationError(f"Chunk at index {index} must be str, got {type(chunk.text).__name__}")
        if chunk.metadata is None:
            raise ChunkValidationError(f"Chunk at index {index} has no metadata")


def validate_embeddings(embeddings: list[list[float]], expected_count: int) -> list[list[float]]:
    if embeddings is None:
        raise EmbeddingError("Embedding service returned None")

    if len(embeddings) != expected_count:
        raise EmbeddingError(f"Embedding count mismatch: expected {expected_count}, got {len(embeddings)}")

    for index, embedding in enumerate(embeddings):
        if embedding is None:
            raise EmbeddingError(f"Embedding at index {index} is None")
        if not isinstance(embedding, (list, tuple)):
            raise EmbeddingError(f"Embedding at index {index} must be a list or tuple, got {type(embedding).__name__}")
        if not embedding:
            raise EmbeddingError(f"Embedding at index {index} is empty")
    return embeddings
