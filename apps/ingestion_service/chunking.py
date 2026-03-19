from dataclasses import dataclass
from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


@dataclass
class ChunkConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50


@dataclass
class PageObject:
    page_number: int
    text: str


@dataclass
class PageMetadata:
    page: int
    chunk_id: int
    source: str | None = None
    hash: str | None = None
    ingested_at: datetime | None = None


@dataclass
class Chunk:
    text: str
    metadata: PageMetadata
    embedding: list | None = None


def clean_text(text: str) -> str:
    return " ".join(text.split())


def extract_pages(file_path: str) -> list[PageObject]:
    reader = PdfReader(file_path)
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append(PageObject(i, clean_text(text)))
    return pages


def split_pages(pages: list[PageObject]) -> list[Chunk]:
    chunk_config = ChunkConfig()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_config.chunk_size,
        chunk_overlap=chunk_config.chunk_overlap,
        separators=["\n\n", "\n", ".", ""],
    )
    chunks = []
    for page in pages:
        split_texts = splitter.split_text(page.text)
        for i, chunk in enumerate(split_texts):
            chunks.append(
                Chunk(
                    text=chunk,
                    metadata=PageMetadata(
                        page=page.page_number,
                        chunk_id=i)
                )
            )
    return chunks
