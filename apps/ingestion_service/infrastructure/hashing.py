import hashlib
import uuid

from apps.ingestion_service.domain.exceptions import HashValidationError, ChunkValidationError


def hash_to_uuid(hash_str: str) -> str:
    try:
        return str(uuid.UUID(hash_str[:32]))
    except ValueError as exc:
        raise HashValidationError(f"Invalid hash string for UUID conversion: '{hash_str}'") from exc


def hash_text(text: str) -> str:
    if not isinstance(text, str):
        raise ChunkValidationError(f"Text must be str, got {type(text).__name__}")
    if not text.strip():
        raise ChunkValidationError("Text must not be empty")
    try:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    except Exception as exc:
        raise ChunkValidationError(f"Failed to hash text: {text}") from exc
