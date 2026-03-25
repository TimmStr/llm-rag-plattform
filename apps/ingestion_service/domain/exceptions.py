from apps.core.exceptions import AppError, EmbeddingError


class IngestionError(AppError):
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message, status_code=status_code)


class PdfExtractionError(IngestionError):
    pass


class ChunkingError(IngestionError):
    pass


class EmptyModelNameError(EmbeddingError):
    pass


class LoadEmbeddingModelError(EmbeddingError):
    pass


class InvalidDeviceError(EmbeddingError):
    pass


class EmbeddingInitializationError(EmbeddingError):
    pass


class EmbeddingFailedError(EmbeddingError):
    pass


class EmbeddingInferenceError(EmbeddingError):
    pass


class InvalidChunkConfigError(IngestionError):
    def __init__(self, message: str):
        super().__init__(message, status_code=400)


class EmptyDocumentError(IngestionError):
    def __init__(self, message: str):
        super().__init__(message, status_code=422)


class PdfIngestionError(IngestionError):
    pass


class VectorIndexingError(IngestionError):
    pass


class ChunkValidationError(IngestionError):
    pass


class HashValidationError(IngestionError):
    pass


class EnrichMetadataError(IngestionError):
    pass
