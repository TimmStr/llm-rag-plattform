from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


@dataclass
class ChunkConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50


def clean_text(text: str) -> str:
    return " ".join(text.split())


def extract_pages(file_path: str) -> list[dict]:
    reader = PdfReader(file_path)
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append(
                {
                    "page": i,
                    "text": clean_text(text)
                }
            )
    return pages


def split_pages(pages: list[dict]) -> list[dict]:
    chunk_config = ChunkConfig()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_config.chunk_size,
        chunk_overlap=chunk_config.chunk_overlap,
        separators=["\n\n", "\n", ".", ""],
    )

    chunks = []
    for page in pages:
        split_texts = splitter.split_text(page["text"])

        for i, chunk_text in enumerate(split_texts):
            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": {
                        "page": page["page"],
                        "chunk_id": i
                    }
                }
            )
    return chunks
