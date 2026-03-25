from dataclasses import dataclass
from typing import Any


@dataclass
class Chunk:
    text: str
    metadata: dict[str, Any]
    embedding: list[float] | None = None


@dataclass
class Page:
    page_number: int
    text: str


@dataclass
class Metadata:
    page_number: int
    chunk_id: int
    source: str | None = None
    hash: str | None = None
    ingested_at: str | None = None
    text: str | None = None


@dataclass
class Chunk:
    text: str
    metadata: Metadata
    embedding: list[float] | None = None
