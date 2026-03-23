class AppError(Exception):
    """Base application exception."""


class IngestionError(AppError):
    """Raised when document ingestion fails."""


class RetrievalError(AppError):
    """Raised when retrieval fails."""


class VectorStoreError(AppError):
    """Raised when vector store operations fail."""


class LLMGenerationError(AppError):
    """Raised when LLM generation fails."""
