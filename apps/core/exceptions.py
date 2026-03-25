class AppError(Exception):
    """Base application exception."""

    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code


class IngestionError(AppError):
    """Baseclass for ingestion errors."""

class IngestionPipelineError(IngestionError):
    """Base class for ingestion pipeline errors."""

class RetrievalError(AppError):
    """Raised when retrieval fails."""


class EmbeddingError(AppError):
    """Raised when embedding fails."""


class VectorStoreError(AppError):
    """Raised when vector store operations fail."""


class LLMGenerationError(AppError):
    """Raised when LLM generation fails."""
