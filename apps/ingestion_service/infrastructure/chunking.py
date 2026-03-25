from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from apps.ingestion_service.domain.entities import Page, Metadata, Chunk
from apps.ingestion_service.domain.exceptions import InvalidChunkConfigError, PdfExtractionError, ChunkingError


@dataclass
class ChunkConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise InvalidChunkConfigError("Chunk size must be greater than 0")
        if self.chunk_overlap < 0:
            raise InvalidChunkConfigError("Chunk overlap must be greater than 0")
        if self.chunk_overlap >= self.chunk_size:
            raise InvalidChunkConfigError("Chunk overlap must be less than chunk size")


def clean_text(text: str) -> str:
    return " ".join(text.split())


def extract_pages(file_path: str) -> list[Page]:
    try:
        reader = PdfReader(file_path, strict=False)
    except FileNotFoundError as exc:
        raise PdfExtractionError(f"PDF file not found: {file_path}") from exc
    except PdfReadError as exc:
        raise PdfExtractionError(f"Invalid or unreadable PDF: {file_path}") from exc
    except Exception as exc:
        raise PdfExtractionError(f"Failed to open PDF: {file_path}") from exc

    pages = []

    try:
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages.append(
                    Page(page_number=i,
                         text=clean_text(text))
                )
    except Exception as exc:
        raise PdfExtractionError(f"Failed while extracting text from PDF: {file_path}") from exc
    return pages


def split_pages(pages: list[Page]) -> list[Chunk]:
    chunk_config = ChunkConfig()

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_config.chunk_size,
            chunk_overlap=chunk_config.chunk_overlap,
            separators=["\n\n", "\n", ".", ""],
        )
    except Exception as exc:
        raise ChunkingError("Failed to initialize text splitter") from exc

    chunks = []
    try:
        for page in pages:
            split_texts = splitter.split_text(page.text)

            for i, chunk_text in enumerate(split_texts):
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata=Metadata(
                            page_number=page.page_number,
                            chunk_id=i)
                    )
                )
    except ChunkingError:
        raise
    except Exception as exc:
        raise ChunkingError("Failed to split pages into chunks") from exc
    return chunks
