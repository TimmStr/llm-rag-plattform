from sentence_transformers import SentenceTransformer

from apps.ingestion_service.chunking import Chunk


class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, chunks: list[Chunk]) -> list[Chunk]:
        texts = [chunk.text for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        return chunks
